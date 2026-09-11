"""Static-shape decode path (one token per step) with per-GPU CUDA graphs.

Everything that depends on the position is expressed with device tensors and fixed-size buffers:
the window ring index pattern, the compressor's parity, the compressed-KV / index-key cache writes
(a dummy row absorbs the "no new group yet" case), the indexer's top-k over the whole cache with
future positions masked. Host work per step is reduced to: the Engram table gather (CPU), a few tiny
H2D/D2D copies between GPU segments, and one graph launch per GPU."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .cukern import fp4_gemv_pairs as cukern_fp4
from .fused import fake_quant_fp4, fake_quant_fp8, rmsnorm, rope_dev_, sparse_attn_decode_split as sparse_attn_decode2, swiglu_quant
from .model import Attention, Block, Transformer, _hc_post, _hc_pre, linear_fp8, select_candidate_blocks


class DecodeRuntime:
    def __init__(self, model: Transformer, use_graphs: bool = True):
        self.m = model
        args = model.args
        self.cfg = args.cfg
        self.win = self.cfg["window_size"]
        self.rd = self.cfg["rope_head_dim"]
        self.topk = self.cfg["index_topk"]
        self.use_graphs = use_graphs
        # contiguous device segments in layer order
        self.segments: list[tuple[torch.device, list[Block]]] = []
        for blk in model.blocks:
            if self.segments and self.segments[-1][0] == blk.device:
                self.segments[-1][1].append(blk)
            else:
                self.segments.append((blk.device, [blk]))
        self.devices = [d for d, _ in self.segments]
        hc, dim = self.cfg["hc_mult"], self.cfg["dim"]
        self.pos = {d: torch.zeros((), dtype=torch.int64, device=d) for d in self.devices}
        self.h_in = {d: torch.zeros(1, 1, hc, dim, dtype=torch.bfloat16, device=d) for d in self.devices}
        self.h_out = {d: torch.zeros(1, 1, hc, dim, dtype=torch.bfloat16, device=d) for d in self.devices}
        self.pre_in = {d: torch.zeros(1, 1, hc, dtype=torch.float32, device=d) for d in self.devices}
        self.pre_out = {d: torch.zeros(1, 1, hc, dtype=torch.float32, device=d) for d in self.devices}
        self.topk_buf = {d: torch.full((1, 1, self.topk), -1, dtype=torch.int32, device=d) for d in self.devices}
        n_cand = args.max_seq_len + 1
        self.cand_buf = {d: torch.zeros(1, 1, n_cand, dtype=torch.bool, device=d) for d in self.devices}
        self.tok = torch.zeros(1, 1, dtype=torch.int64, device=self.devices[0])
        self.logits = torch.zeros(1, self.cfg["vocab_size"], dtype=torch.float32, device=self.devices[-1])
        self.eng_in: dict[int, torch.Tensor] = {}
        for blk in model.blocks:
            if blk.engram is not None:
                cols = blk.engram.wkv.shape[1]
                self.eng_in[blk.layer_id] = torch.zeros(1, 1, cols, dtype=torch.bfloat16, device=blk.device)
        # owners: the row written this step (value + index), to propagate to mirrors on other devices
        self.kv_row: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ik_row: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for blk in model.blocks:
            if blk.attn.is_kv_source:
                d = blk.device
                self.kv_row[blk.layer_id] = (torch.zeros(1, 1, self.cfg["head_dim"], dtype=torch.bfloat16, device=d), torch.zeros(1, dtype=torch.int64, device=d))
                self.ik_row[blk.layer_id] = (torch.zeros(1, 1, self.cfg["index_head_dim"], dtype=torch.bfloat16, device=d), torch.zeros(1, dtype=torch.int64, device=d))
        self.arange_win = {d: torch.arange(self.win, device=d) for d in self.devices}
        self.graphs: dict[torch.device, torch.cuda.CUDAGraph] = {}
        self.kv_owner = -1
        self.index_owner = -1

    # ------------------------------------------------------------------ attention (static)
    def attention(self, A: Attention, x: torch.Tensor, d: torch.device) -> torch.Tensor:
        pos = self.pos[d]
        rd, eps = self.rd, A.eps
        qr = rmsnorm(linear_fp8(x, A.wq_a), A.q_norm_w, eps)
        q = linear_fp8(qr, A.wq_b).unflatten(-1, (A.n_heads, A.head_dim))
        rope_dev_(q, rd, A.cos, A.sin, pos)
        # sliding window: write this token's KV into the ring, build the ring index pattern
        kv = rmsnorm(linear_fp8(x, A.wkv), A.kv_norm_w, eps)
        rope_dev_(kv, rd, A.cos, A.sin, pos)
        kv = fake_quant_fp8(kv, 32)
        slot = torch.remainder(pos, self.win)
        A.window_kv_cache.index_copy_(1, slot.view(1), kv)
        widx = torch.remainder(self.arange_win[d] + slot + 1, self.win)
        widx = torch.where(widx > pos, -1, widx).to(torch.int32).view(1, 1, self.win)
        if A.ratio:
            ratio = A.ratio
            compress_len = torch.div(pos + 1, ratio, rounding_mode="floor")
            latent = None
            if A.is_kv_source:
                self.kv_owner = A.layer_id
                latent, should = self.compressor(A, x, pos)
                cache = self.m.shared.compress_kv[(A.layer_id, d)]
                row = torch.where(should, compress_len - 1, torch.full_like(compress_len, cache.shape[1] - 1))
            if A.is_index_source:
                idxs = self.indexer(A, x, qr, latent, pos, compress_len, d, row if A.is_kv_source else None)
                self.topk_buf[d].copy_(idxs)
            else:
                idxs = self.topk_buf[d]
            if latent is not None:
                latent = latent.contiguous()
                rope_dev_(latent, rd, A.cos, A.sin, pos, add=1 - ratio)
                latent = fake_quant_fp4(latent, 16, scale_e4m3=True)
                cache.index_copy_(1, row, latent)
                val, idx = self.kv_row[A.layer_id]
                val.copy_(latent)
                idx.copy_(row)
            ckv = self.m.shared.compress_kv[(self.kv_owner, d)]
            o = sparse_attn_decode2(q, A.window_kv_cache, ckv, A.attn_sink, torch.cat([widx, idxs], dim=-1), A.softmax_scale)
        else:
            o = sparse_attn_decode2(q, A.window_kv_cache, None, A.attn_sink, widx, A.softmax_scale)
        rope_dev_(o, rd, A.cos, A.sin, pos, inverse=True)
        o = o.view(1, 1, A.n_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, A.wo_a)
        return linear_fp8(o.flatten(2), A.wo_b)

    def compressor(self, A: Attention, x: torch.Tensor, pos: torch.Tensor):
        C = A.compressor
        if C.ratio == 1:
            latent = rmsnorm(F.linear(x, C.wkv), C.norm_w, C.eps)
            return latent, torch.ones((), dtype=torch.bool, device=x.device)
        xf = x.float()
        kv, score = F.linear(xf, C.wkv), F.linear(xf, C.wgate)
        slot = torch.remainder(pos, C.ratio).view(1)
        C.kv_state.index_copy_(1, slot, kv)
        C.score_state.index_copy_(1, slot, score)
        pooled = (C.kv_state * C.score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        should = torch.remainder(pos + 1, C.ratio) == 0
        return rmsnorm(pooled.to(x.dtype), C.norm_w, C.eps), should

    def indexer(self, A: Attention, x, qr, latent, pos, compress_len, d, row):
        I = A.indexer
        ratio, rd = I.ratio, self.rd
        if I.owns_k:
            self.index_owner = A.layer_id
            k = rmsnorm(F.linear(latent, I.wk), I.k_norm_w, I.eps).contiguous()
            rope_dev_(k, rd, I.cos, I.sin, pos, add=1 - ratio)
            k = fake_quant_fp4(k, 32)
            cache = self.m.shared.index_k[(A.layer_id, d)]
            cache.index_copy_(1, row, k)
            val, idx = self.ik_row[A.layer_id]
            val.copy_(k)
            idx.copy_(row)
        q = linear_fp8(qr, I.wq_b).unflatten(-1, (I.n_heads, I.head_dim))
        rope_dev_(q, rd, I.cos, I.sin, pos)
        q = fake_quant_fp4(q, 32)
        index_k = self.m.shared.index_k[(self.index_owner, d)]  # [1, max_c + 1, 128] (last row = dummy)
        weights = F.linear(x, I.weights_proj) * (I.softmax_scale * I.n_heads**-0.5)
        score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        score = (score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)  # [1, 1, max_c + 1]
        n_pos = score.shape[-1]
        score = score.masked_fill(torch.arange(n_pos, device=d) >= compress_len, -torch.inf)
        if I.is_candidate_source:
            cand = select_candidate_blocks(score, compress_len, I.candidate_topk_blocks, I.candidate_block_size)
            self.cand_buf[d][..., :n_pos].copy_(cand)
        elif I.uses_candidates:
            score = score.masked_fill(~self.cand_buf[d][..., :n_pos], -torch.inf)
        idxs = score.topk(self.topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_len, idxs + self.win, -1).to(torch.int32)

    # ------------------------------------------------------------------ block / segment
    def block(self, blk: Block, x: torch.Tensor, pre_mix: torch.Tensor):
        residual = x
        attn_pre, attn_post, attn_comb = blk.hc_mixes(x, *blk.hc_attn)
        x = _hc_pre(x, pre_mix)
        x = rmsnorm(x, blk.attn_norm_w, blk.eps)
        x = self.attention(blk.attn, x, blk.device)
        x = _hc_post(x, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(x, *blk.hc_ffn)
        x = _hc_pre(x, attn_pre)
        x = rmsnorm(x, blk.ffn_norm_w, blk.eps)
        x = blk.ffn(x)
        x = _hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def run_segment(self, si: int):
        d, blocks = self.segments[si]
        with torch.cuda.device(d):
            if si == 0:
                h = F.embedding(self.tok, self.m.embed).unsqueeze(2).repeat(1, 1, self.cfg["hc_mult"], 1)
                pre = torch.zeros(1, 1, self.cfg["hc_mult"], dtype=torch.float32, device=d)
                pre[:, :, 0] = 1.0
            else:
                h, pre = self.h_in[d], self.pre_in[d]
            for blk in blocks:
                if blk.engram is not None:
                    h = blk.engram.apply(h, self.eng_in[blk.layer_id])
                h, pre = self.block(blk, h, pre)
            if si == len(self.segments) - 1:
                hh = _hc_pre(h, pre)[:, -1]
                hh = rmsnorm(hh, self.m.norm_w, self.cfg["norm_eps"])
                self.logits.copy_(F.linear(hh, self.m.head).float())
            else:
                self.h_out[d].copy_(h)
                self.pre_out[d].copy_(pre)

    def _propagate(self, si: int):
        """After a segment: push cache rows written by its owners and the index/candidate buffers to
        the devices of later segments (plain D2D copies, outside the graphs)."""
        d, blocks = self.segments[si]
        later = self.devices[si + 1 :]
        if not later:
            return
        for blk in blocks:
            lid = blk.layer_id
            if lid in self.kv_row:
                val, idx = self.kv_row[lid]
                for dd in later:
                    self.m.shared.compress_kv[(lid, dd)].index_copy_(1, idx.to(dd, non_blocking=True), val.to(dd, non_blocking=True))
                val, idx = self.ik_row[lid]
                for dd in later:
                    self.m.shared.index_k[(lid, dd)].index_copy_(1, idx.to(dd, non_blocking=True), val.to(dd, non_blocking=True))
        nd = later[0]
        self.topk_buf[nd].copy_(self.topk_buf[d], non_blocking=True)
        self.cand_buf[nd].copy_(self.cand_buf[d], non_blocking=True)
        self.h_in[nd].copy_(self.h_out[d], non_blocking=True)
        self.pre_in[nd].copy_(self.pre_out[d], non_blocking=True)

    def capture(self):
        for si, (d, _) in enumerate(self.segments):
            with torch.cuda.device(d):
                s = torch.cuda.Stream(d)
                s.wait_stream(torch.cuda.current_stream(d))
                with torch.cuda.stream(s):
                    for _ in range(2):  # warm up (Triton compiles, allocator)
                        self.run_segment(si)
                torch.cuda.current_stream(d).wait_stream(s)
                torch.cuda.synchronize(d)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    self.run_segment(si)
                self.graphs[d] = g
        torch.cuda.synchronize()

    @torch.inference_mode()
    def step(self, token: int, pos: int) -> torch.Tensor:
        """One decode step for the token at `pos`; returns logits [1, vocab] on the last device."""
        self.tok.fill_(token)
        for d in self.devices:
            self.pos[d].fill_(pos)
        # the owner bookkeeping mirrors the eager path: sources set themselves as they run (in layer order)
        self.kv_owner = -1
        self.index_owner = -1
        if self.m.engram_hash is not None:
            hashes = self.m.engram_hash(self.tok, pos)
            for blk in self.m.blocks:
                if blk.engram is not None:
                    emb = blk.engram.table.lookup(hashes[:, :, blk.engram.layer_hash_index, :], blk.device).flatten(-2)
                    self.eng_in[blk.layer_id].copy_(emb)
        for si, (d, _) in enumerate(self.segments):
            if self.use_graphs and d in self.graphs:
                self.graphs[d].replay()
            else:
                self.run_segment(si)
            self._propagate(si)
        return self.logits


class OffloadDecodeRuntime(DecodeRuntime):
    """Single-GPU expert-offload decode: each layer is two CUDA graphs with the host-side expert
    gather (gate result -> DMA from pinned RAM into the staging buffers) in between. Only the dense
    part of the layer is captured, so the per-layer host sync no longer drains ~40 kernel launches."""

    def __init__(self, model: Transformer, use_graphs: bool = True):
        super().__init__(model, use_graphs=False)
        assert len(self.devices) == 1, "offload mode runs all layers on one GPU"
        self.d = self.devices[0]
        self.use_graphs = use_graphs
        hc, dim = self.cfg["hc_mult"], self.cfg["dim"]
        d = self.d
        self.h_buf = torch.zeros(1, 1, hc, dim, dtype=torch.bfloat16, device=d)
        self.pre_buf = torch.zeros(1, 1, hc, dtype=torch.float32, device=d)
        self.x_buf = torch.zeros(1, dim, dtype=torch.bfloat16, device=d)
        self.res_buf = torch.zeros(1, 1, hc, dim, dtype=torch.bfloat16, device=d)
        self.ffn_pre = torch.zeros(1, 1, hc, dtype=torch.float32, device=d)
        self.ffn_post = torch.zeros(1, 1, hc, dtype=torch.float32, device=d)
        self.ffn_comb = torch.zeros(1, 1, hc, hc, dtype=torch.float32, device=d)
        topk = self.cfg["n_activated_experts"]
        self.gate_w = torch.zeros(1, topk, dtype=torch.float32, device=d)
        self.gate_idx = torch.zeros(1, topk, dtype=torch.int64, device=d)
        self.g1: dict[int, torch.cuda.CUDAGraph] = {}
        self.g2: dict[int, torch.cuda.CUDAGraph] = {}
        self.g3: dict[int, torch.cuda.CUDAGraph] = {}
        self.cpu_experts = model.blocks[0].ffn.host is not None
        self.prof = None  # set to a defaultdict(float) to collect per-stage seconds
        self.y_gpu = torch.zeros(1, dim, dtype=torch.float32, device=d)
        self.y_shared = torch.zeros(1, dim, dtype=torch.float32, device=d)

    # ---- the two halves of a layer, on static buffers
    def part1(self, blk: Block):
        h = self.h_buf
        if blk.engram is not None:
            h = blk.engram.apply(h, self.eng_in[blk.layer_id])
        residual = h
        attn_pre, attn_post, attn_comb = blk.hc_mixes(h, *blk.hc_attn)
        x = _hc_pre(h, self.pre_buf)
        x = rmsnorm(x, blk.attn_norm_w, blk.eps)
        x = self.attention(blk.attn, x, blk.device)
        h = _hc_post(x, residual, attn_post, attn_comb)
        self.res_buf.copy_(h)
        ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(h, *blk.hc_ffn)
        self.ffn_pre.copy_(ffn_pre)
        self.ffn_post.copy_(ffn_post)
        self.ffn_comb.copy_(ffn_comb)
        x = _hc_pre(h, attn_pre)
        x = rmsnorm(x, blk.ffn_norm_w, blk.eps).reshape(-1, blk.ffn.dim)
        self.x_buf.copy_(x)
        w, idx = blk.ffn.gate(x)
        self.gate_w.copy_(w)
        self.gate_idx.copy_(idx)

    def part2_cpu_shared(self, blk: Block):
        """GPU work that overlaps the CPU expert computation: the shared expert."""
        self.y_shared.copy_(blk.ffn.shared_expert(self.x_buf))

    def part2_cpu_post(self, blk: Block):
        moe = blk.ffn
        y = (self.y_gpu + self.y_shared).to(torch.bfloat16).view(1, 1, moe.dim)
        h = _hc_post(y, self.res_buf, self.ffn_post, self.ffn_comb)
        self.h_buf.copy_(h)
        self.pre_buf.copy_(self.ffn_pre)

    def _cpu_experts(self, blk: Block):
        moe = blk.ffn
        xq = fake_quant_fp8(self.x_buf, 32)
        moe.x_host.copy_(xq.view(-1))  # sync: waits for part1 on the GPU
        ids = self.gate_idx.flatten().tolist()
        wts = self.gate_w.flatten().tolist()
        y = moe.host.forward(moe.x_host, ids, wts, float(moe.swiglu_limit))
        moe.y_host.copy_(y)
        self.y_gpu.copy_(moe.y_host, non_blocking=True)

    def part2(self, blk: Block):
        moe = blk.ffn
        n_tok, n_pairs = 1, moe.topk
        tok, pair_rows, ones = moe._pair_tables(n_tok, self.d)
        b = moe._staging(n_pairs, "decode")
        xq = fake_quant_fp8(self.x_buf, 32)
        local = torch.arange(n_pairs, device=self.d, dtype=torch.int32)
        gu = cukern_fp4(xq, b["w13"], b["s13"], tok, local, ones, n_pairs)
        hq = swiglu_quant(gu, self.gate_w.flatten().contiguous(), moe.inter, moe.swiglu_limit)
        y2 = cukern_fp4(hq, b["w2"], b["s2"], pair_rows, local, ones, n_pairs)
        y = y2.view(n_tok, moe.topk, moe.dim).sum(dim=1)
        y += moe.shared_expert(self.x_buf)
        y = y.to(torch.bfloat16).view(1, 1, moe.dim)
        h = _hc_post(y, self.res_buf, self.ffn_post, self.ffn_comb)
        self.h_buf.copy_(h)
        self.pre_buf.copy_(self.ffn_pre)

    def _gather(self, moe):
        b = moe._staging(moe.topk, "decode")
        for i, e in enumerate(self.gate_idx.flatten().tolist()):  # the one host sync per layer
            for gk, src in (("w13", moe.w13), ("s13", moe.s13), ("w2", moe.w2), ("s2", moe.s2)):
                b[gk][i].copy_(src[e], non_blocking=True)

    def capture(self):
        if not self.use_graphs:
            return
        with torch.cuda.device(self.d):
            s = torch.cuda.Stream(self.d)
            s.wait_stream(torch.cuda.current_stream(self.d))
            for blk in self.m.blocks:
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self.part1(blk)
                        if self.cpu_experts:
                            self.part2_cpu_shared(blk)
                            torch.cuda.synchronize(self.d)
                            self._cpu_experts(blk)
                            self.part2_cpu_post(blk)
                        else:
                            self._gather(blk.ffn)
                            self.part2(blk)
                torch.cuda.synchronize(self.d)
                g1 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g1, stream=s):
                    self.part1(blk)
                if self.cpu_experts:
                    g2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g2, stream=s):
                        self.part2_cpu_shared(blk)
                    g3 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g3, stream=s):
                        self.part2_cpu_post(blk)
                    self.g3[blk.layer_id] = g3
                else:
                    g2 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g2, stream=s):
                        self.part2(blk)
                self.g1[blk.layer_id], self.g2[blk.layer_id] = g1, g2
            torch.cuda.current_stream(self.d).wait_stream(s)
            torch.cuda.synchronize(self.d)

    @torch.inference_mode()
    def step(self, token: int, pos: int) -> torch.Tensor:
        d = self.d
        self.tok.fill_(token)
        self.pos[d].fill_(pos)
        self.kv_owner = -1
        self.index_owner = -1
        if self.m.engram_hash is not None:
            hashes = self.m.engram_hash(self.tok, pos)
            for blk in self.m.blocks:
                if blk.engram is not None:
                    emb = blk.engram.table.lookup(hashes[:, :, blk.engram.layer_hash_index, :], d).flatten(-2)
                    self.eng_in[blk.layer_id].copy_(emb)
        self.h_buf.copy_(F.embedding(self.tok, self.m.embed).unsqueeze(2).repeat(1, 1, self.cfg["hc_mult"], 1))
        self.pre_buf.zero_()
        self.pre_buf[:, :, 0] = 1.0
        prof = self.prof
        for blk in self.m.blocks:
            lid = blk.layer_id
            t0 = time.perf_counter() if prof is not None else 0.0
            if self.use_graphs:
                self.g1[lid].replay()
            else:
                self.part1(blk)
            if self.cpu_experts:
                if self.use_graphs:
                    self.g2[lid].replay()  # shared expert on the GPU, overlapping the CPU experts
                else:
                    self.part2_cpu_shared(blk)
                if prof is not None:
                    torch.cuda.synchronize(d); t1 = time.perf_counter(); prof["gpu dense+attn+shared"] += t1 - t0; t0 = t1
                self._cpu_experts(blk)
                if prof is not None:
                    t1 = time.perf_counter(); prof["cpu experts (+sync)"] += t1 - t0; t0 = t1
                if self.use_graphs:
                    self.g3[lid].replay()
                else:
                    self.part2_cpu_post(blk)
                if prof is not None:
                    torch.cuda.synchronize(d); prof["gpu post"] += time.perf_counter() - t0
                continue
            self._gather(blk.ffn)
            if self.use_graphs:
                self.g2[lid].replay()
            else:
                self.part2(blk)
        hh = _hc_pre(self.h_buf, self.pre_buf)[:, -1]
        hh = rmsnorm(hh, self.m.norm_w, self.cfg["norm_eps"])
        self.logits.copy_(F.linear(hh, self.m.head).float())
        return self.logits
