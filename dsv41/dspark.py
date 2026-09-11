"""DSpark (DeepSeek-V4.1's multi-token prediction head): 3 extra blocks that draft `block_size` (5) tokens
in one parallel pass from the last sampled token and the attention inputs of the target layers 37-39.

Reference semantics (inference/model.py, DSparkBlock / forward_spec):
  main_x  = main_norm(main_proj(cat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39))))          [1, dim]
  the draft blocks keep a sliding-window cache of kv(main_x) for the main positions;
  x       = embed([t, noise, noise, noise, noise]) (t = the token just sampled) at positions pos+1 .. pos+5
  each draft block: window attention of the 5 drafts over the main window + the 5 drafts themselves
  (non-causal), then a 128-expert top-3 MoE, all with hyper-connections like the main blocks;
  head:  logits of the 5 positions + a first-order Markov bias (prev token -> vocab) applied left to right,
  greedy pick, and a confidence score per draft position.
This module is the eager version (measurement / correctness); the static-graph version follows."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .fused import fake_quant_fp8, rmsnorm, rope_
from .load import _dense, load_layer
from .model import Block, _hc_post, _hc_pre, linear_fp8
from .w8 import linear_w, oproj_a


class DSpark:
    def __init__(self, ckpt, args, device, embed: torch.Tensor, head: torch.Tensor, shared, n_layers: int):
        cfg = args.cfg
        self.device = device
        self.dim = cfg["dim"]
        self.hc = cfg["hc_mult"]
        self.block = cfg["dspark_block_size"]
        self.noise = cfg["dspark_noise_token_id"]
        self.targets = list(cfg["dspark_target_layer_ids"])
        self.embed, self.head = embed.to(device), head  # the embedding table lives on the first GPU: a copy (1.3 GB) here
        self.eps = cfg["norm_eps"]
        self.blocks: list[Block] = []
        self.ws = []
        for i in range(cfg["n_mtp_layers"]):
            w = load_layer(ckpt, n_layers + i, device, prefix=f"mtp.{i}.")
            blk = Block(args, n_layers + i, w, device, shared)
            blk.ffn.topk = cfg["dspark_n_activated_experts"]
            blk.ffn.n_experts = cfg["dspark_n_routed_experts"]
            self.blocks.append(blk)
            self.ws.append(w)
        w0, wl = self.ws[0], self.ws[-1]
        self.main_proj = w0["main_proj.weight"]
        self.main_norm_w = w0["main_norm.weight"]
        self.norm_w = wl["norm.weight"]
        self.markov_embed = wl["markov_head.embed.weight"]  # bf16 [vocab, 256]
        self.markov_head = wl["markov_head.head.weight"]  # bf16 [vocab, 256]
        self.conf_w = wl["confidence_head.proj.weight"].float()  # [1, dim + 256]
        self.rd = cfg["rope_head_dim"]
        self.win = cfg["window_size"]

    # ---- main-side state: kv of the projected target hidden for every main position
    @torch.no_grad()
    def write_main(self, main_hidden: torch.Tensor, start_pos: int):
        """main_hidden: bf16 [T, 3*dim] for main positions start_pos .. start_pos+T-1."""
        T = main_hidden.shape[0]
        main_x = rmsnorm(linear_fp8(main_hidden.to(self.device), self.main_proj), self.main_norm_w, self.eps).view(1, T, self.dim)
        for blk in self.blocks:
            A = blk.attn
            kv = rmsnorm(linear_fp8(main_x, A.wkv), A.kv_norm_w, A.eps).contiguous()
            rope_(kv, self.rd, A.cos, A.sin, start_pos)
            kv = fake_quant_fp8(kv, 32)
            for j in range(T):
                A.window_kv_cache[0, (start_pos + j) % self.win] = kv[0, j]

    def _attention(self, A, x: torch.Tensor, pos: int) -> torch.Tensor:
        """x: [1, B, dim] draft inputs at positions pos+1 .. pos+B; attends to the main window (<= pos) and all drafts."""
        B = x.shape[1]
        qr = rmsnorm(linear_fp8(x, A.wq_a), A.q_norm_w, A.eps)
        q = linear_fp8(qr, A.wq_b).view(1, B, A.n_heads, A.head_dim).contiguous()
        rope_(q, self.rd, A.cos, A.sin, pos + 1)
        kv = rmsnorm(linear_fp8(x, A.wkv), A.kv_norm_w, A.eps).contiguous()
        rope_(kv, self.rd, A.cos, A.sin, pos + 1)
        kv = fake_quant_fp8(kv, 32)
        win = A.window_kv_cache[0]  # [win, head_dim]
        K = torch.cat([win, kv[0]], dim=0).float()  # [win + B, d]
        valid = torch.ones(self.win + B, dtype=torch.bool, device=x.device)
        if pos + 1 < self.win:
            valid[pos + 1 : self.win] = False
        s = torch.einsum("bhd,td->bht", q[0].float(), K) * A.softmax_scale  # [B, H, T]
        s = s.masked_fill(~valid, float("-inf"))
        m = s.amax(dim=-1, keepdim=True)
        p = torch.exp(s - m)
        l = p.sum(dim=-1, keepdim=True) + torch.exp(A.attn_sink.view(1, -1, 1) - m)
        o = (torch.einsum("bht,td->bhd", p, K) / l).to(torch.bfloat16).view(1, B, A.n_heads, A.head_dim).contiguous()
        rope_(o, self.rd, A.cos, A.sin, pos + 1, inverse=True)
        o = oproj_a(o.view(1, B, A.n_groups, -1), A.wo_a, A.n_groups, A.o_lora_rank)
        return linear_fp8(o, A.wo_b)

    @torch.no_grad()
    def draft(self, token: int, pos: int):
        """Drafts for positions pos+2 .. pos+1+block given t_{pos+1} = token and the main state up to pos.
        Returns (ids [block], confidence [block] fp32)."""
        B = self.block
        ids = torch.tensor([token] + [self.noise] * (B - 1), device=self.device)
        h = F.embedding(ids, self.embed).view(1, B, 1, self.dim).repeat(1, 1, self.hc, 1)
        pre = h.new_zeros(1, B, self.hc, dtype=torch.float32)
        pre[:, :, 0] = 1.0
        for blk in self.blocks:
            residual = h
            attn_pre, attn_post, attn_comb = blk.hc_mixes(h, *blk.hc_attn)
            x = rmsnorm(_hc_pre(h, pre), blk.attn_norm_w, blk.eps)
            a = self._attention(blk.attn, x, pos)
            h = _hc_post(a, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = blk.hc_mixes(h, *blk.hc_ffn)
            x = rmsnorm(_hc_pre(h, attn_pre), blk.ffn_norm_w, blk.eps)
            y = blk.ffn(x)
            h = _hc_post(y, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        x = _hc_pre(h, pre)[0]  # [B, dim]
        logits = F.linear(rmsnorm(x, self.norm_w, self.eps), self.head).float()  # [B, vocab]
        out = [token]
        embeds = []
        for i in range(B):
            e = self.markov_embed[out[i]]
            embeds.append(e)
            logits[i] += F.linear(e.float(), self.markov_head.float())
            out.append(int(logits[i].argmax()))
        conf = F.linear(torch.cat([x.float(), torch.stack(embeds).float()], dim=-1), self.conf_w).view(-1)
        return out[1:], conf
