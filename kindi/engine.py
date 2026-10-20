"""Kindi engine: Ternary-Bonsai-2-2B (Qwen3.5-2B hybrid: 18 Gated-DeltaNet layers + 6 attention layers, ternary
weights) on one Kepler SMX (Quadro K420, sm_30, 1 GB).

Per token: embedding row -> 24 x (GDN or attention block + SwiGLU FFN) -> output norm -> tied head.
Every matrix product is a ternary GEMV (k_g4, lookup tables, see docs/HOW-IT-WORKS.md); the activations entering it
are Hadamard-folded (the transform Prism ML folded into the weights), fused into the preceding norm.
Arithmetic follows llama.cpp's reference for this model (fp64 norm sums, fp16 K/V cache), so greedy output matches
it token for token (kindi verify)."""
import json
import os
import time

import numpy as np
import cupy as cp

from . import layout

D, NL, NFF = 2048, 24, 6144
NH, NHKV, HD = 8, 2, 256
NK, NV, HK = 16, 16, 128
CONV_CH = 6144
VOCAB = 248320
ROPE_BASE = 1e7
INV_FREQ = np.float64(ROPE_BASE) ** (-2.0 * np.arange(32) / 64.0)
EOS = (248044, 248046)
CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kindi.cu")
KERNELS = ("k_g4", "k_g4l", "k_fold_fwht", "k_rmsfold", "k_smfold", "k_l2w", "k_rmsw", "k_gdn_gates",
           "k_gdn_conv", "k_gdn_delta", "k_rope", "k_deq_emb3", "k_attn_s4w", "k_softmax_rows", "k_attn_v4p",
           "k_attn_comb", "k_argmax", "k_expand", "k_mask_neg", "k_band", "k_topk")


class Kernels(object):
    def __init__(self):
        mod = cp.RawModule(code=open(CU, encoding="utf-8").read())
        for k in KERNELS:
            setattr(self, k, mod.get_function(k))


class Engine(object):
    """One sequence at a time. step(tok, pos) -> next token (greedy) with the logits left in self.ylog;
    feed(tok, pos) processes a prompt token without the output head."""

    def __init__(self, pack, ctx=4096, clusters=None, log=print):
        t0 = time.time()
        self.log = log
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        if man.get("pack_format") != 1:
            raise SystemExit("kindi: pack format %r is not supported - re-run kindi convert" % man.get("pack_format"))
        self.k = Kernels()
        self.idx = json.load(open(os.path.join(pack, "index.json")))
        self.bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        bc = np.fromfile(os.path.join(pack, "bcodes.bin"), dtype=np.uint8)
        dp = np.fromfile(os.path.join(pack, "dpm.bin"), dtype=np.uint16)
        self.hsl = None
        if clusters is not None:
            bc, dp, self.hsl = layout.reorder_head(self.bi, bc, dp, clusters)
        dims = layout.repack(self.bi, bc, dp)
        self.bc, self.dp = cp.asarray(bc), cp.asarray(dp)
        del bc, dp
        self.w = {n: (self.bc[self.bi[n]["bc_off"]:], self.dp[self.bi[n]["dpm_off"] // 2:], np.int32(R), np.int32(C))
                  for n, (R, C) in dims.items()}
        self.fl = cp.asarray(np.fromfile(os.path.join(pack, "f32.bin"), dtype=np.float32))
        self.s2048 = cp.asarray(np.fromfile(os.path.join(pack, "signs_2048.f32"), dtype=np.float32))
        self.s6144 = cp.asarray(np.fromfile(os.path.join(pack, "signs_6144.f32"), dtype=np.float32))
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
        Rh = VOCAB if self.hsl is None else self.hsl["Rp"]
        self.y, self.xp, self.h, self.xn, self.zv = z(Rh), z(NFF), z(D), z(D), z(D)
        self.qkv, self.cs, self.qkbuf, self.qkvbuf = z(CONV_CH), z(CONV_CH), z(4096), z(512)
        self.kn, self.qn, self.qr, self.kro, self.fo, self.fo6 = z(512), z(4096), z(2048), z(512), z(2048), z(NFF)
        self.outgdn, self.att, self.ffg, self.ffu = z(D), z(NH * HD), z(NFF), z(NFF)
        self.gb, self.gg, self.gout = z(NV), z(NV), z(NV * HK)
        self.sbuf = z(NH * self.ctx)
        self.apart = z(((self.ctx + 255) // 256) * NH * HD)   # attention partials: NH x HD per 256-position chunk
        self.aout = cp.zeros(1, dtype=cp.int32)
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * INV_FREQ[None, :]
        self.cos_t = cp.asarray(np.cos(th).astype(np.float32))
        self.sin_t = cp.asarray(np.sin(th).astype(np.float32))
        # recurrent (GDN) state for the 18 GDN layers; fp16 K/V cache for the 6 attention layers
        self.M = [cp.zeros((NV, HK, HK) if il % 4 != 3 else (1,), dtype=cp.float32) for il in range(NL)]
        self.conv = [z(CONV_CH * 3) for _ in range(NL)]
        self.Kc = [cp.zeros((self.ctx, NHKV, HD), dtype=cp.float16) for _ in range(6)]
        self.Vc = [cp.zeros((self.ctx, NHKV, HD), dtype=cp.float16) for _ in range(6)]
        self.shortlist = 0                 # > 0: fast head over this many clusters
        if self.hsl is not None:
            m = self.hsl
            self.hs_cent = cp.asarray(m["cent"])
            self.hs_permtok, self.hs_permtok_h, self.hs_permpos = cp.asarray(m["permtok"]), m["permtok"], m["permpos"]
            self.hs_gstart, self.hs_gcount = cp.asarray(m["gstart"]), cp.asarray(m["gcount"])
            self.hs_glist = cp.zeros(m["Rp"] // 16, dtype=cp.int32)
            self.hs_yc = z(m["Rp"])
            self.hs_tokc = cp.zeros(m["Rp"], dtype=cp.int32)
        self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
        log("engine ready: ctx %d, head %s (%.1fs)" % (self.ctx, "reordered" if self.hsl else "plain", time.time() - t0))

    # ------------------------------------------------------------------------------------------- helpers
    def f32(self, name):
        rec = self.idx["tensors"][name]
        off = rec["f32_off"] // 4
        return self.fl[off:off + rec["rows"] * rec["ne0"]]

    def gemv(self, names, x, outs):
        """Up to three ternary tensors with the same input in one launch."""
        a = [self.w[n] for n in names]
        args = []
        for i in range(3):
            if i < len(a):
                args += [a[i][0], a[i][1], outs[i], a[i][2]]
            else:
                args += [a[0][0], a[0][1], outs[0], np.int32(0)]
        self.k.k_g4((1,), (1024,), tuple(args) + (x, a[0][3]))

    def rmsfold(self, x, wname):
        self.k.k_rmsfold((1,), (512,), (x, self.f32(wname), self.s2048, self.xn, self.xp))
        return self.xn

    def rms_rows(self, x, wname, rows, n, out):
        self.k.k_rmsw(((rows + 7) // 8,), (256,), (x, self.f32(wname), out, np.int32(n), np.int32(rows)))
        return out

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc:
            a.fill(0)

    def embed(self, tok):
        a = self.w["token_embd.weight"]
        row = int(self.hs_permpos[tok]) if self.hsl is not None else int(tok)
        self.k.k_deq_emb3((1,), (1024,), (a[0], a[1], self.s2048, self.h, np.int32(row)))
        return self.h

    # ------------------------------------------------------------------------------------------- blocks
    def gdn(self, il, h):
        p = "blk.%d." % il
        xn = self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_qkv.weight", p + "attn_gate.weight"], self.xp, [self.qkv, self.zv])
        self.k.k_gdn_gates((NV,), (256,), (xn, self.f32(p + "ssm_beta.weight"), self.f32(p + "ssm_alpha.weight"),
                                           self.f32(p + "ssm_dt.bias"), self.f32(p + "ssm_a"), self.gb, self.gg,
                                           np.int32(D)))
        self.k.k_gdn_conv(((CONV_CH + 255) // 256,), (256,),
                          (self.qkv, self.conv[il], self.f32(p + "ssm_conv1d.weight"), self.cs, np.int32(CONV_CH)))
        self.k.k_l2w((4,), (256,), (self.cs, self.qkbuf, np.int32(HK), np.int32(32)))
        self.k.k_gdn_delta((NV,), (HK,), (self.M[il], self.qkbuf[:NK * HK], self.qkbuf[NK * HK:], self.cs[4096:],
                                          self.gg, self.gb, self.gout, np.float32(1.0 / np.sqrt(HK))))
        outn = self.rms_rows(self.gout, p + "ssm_norm.weight", NV, HK, self.outgdn)
        self.k.k_smfold(((NV * HK) // 512,), (512,), (self.zv, outn, self.s2048, self.xp))
        self.gemv([p + "ssm_out.weight"], self.xp, [self.y])
        return h + self.y[:D]

    def attn(self, il, h, pos, kvi):
        p = "blk.%d." % il
        self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], self.xp,
                  [self.qn, self.kro, self.qkvbuf])
        Kn = self.rms_rows(self.kro, p + "attn_k_norm.weight", NHKV, HD, self.kn)
        qc = self.qn.reshape(NH, 2 * HD)[:, :HD].copy()
        q = self.rms_rows(qc, p + "attn_q_norm.weight", NH, HD, self.fo)
        cs, sn = self.cos_t[pos], self.sin_t[pos]
        self.k.k_rope((NH,), (256,), (q, cs, sn, self.qr, np.int32(HD)))
        self.k.k_rope((NHKV,), (256,), (Kn, cs, sn, self.kro, np.int32(HD)))
        self.Kc[kvi][pos] = self.kro.reshape(NHKV, HD)
        self.Vc[kvi][pos] = self.qkvbuf.reshape(NHKV, HD)
        n = pos + 1
        nch = (n + 255) // 256
        Kall, Vall = self.Kc[kvi][:n], self.Vc[kvi][:n]
        self.k.k_attn_s4w((NHKV, min(32, (n + 31) // 32)), (256,), (Kall, self.qr, self.sbuf, np.int32(n)))
        self.k.k_softmax_rows((NH,), (256,), (self.sbuf, np.int32(n)))
        self.k.k_attn_v4p((NHKV, 1, nch), (128,), (self.sbuf, Vall, self.apart, np.int32(n)))
        self.k.k_attn_comb((NH,), (256,), (self.apart, self.qn, self.att, np.int32(nch)))
        self.k.k_fold_fwht((D // 512,), (512,), (self.att, self.s2048, self.xp))
        self.gemv([p + "attn_output.weight"], self.xp, [self.y])
        return h + self.y[:D]

    def ffn(self, il, h):
        p = "blk.%d." % il
        self.rmsfold(h, p + "post_attention_norm.weight")
        self.gemv([p + "ffn_gate.weight", p + "ffn_up.weight"], self.xp, [self.ffg, self.ffu])
        self.k.k_smfold((NFF // 512,), (512,), (self.ffg, self.ffu, self.s6144, self.fo6))
        self.gemv([p + "ffn_down.weight"], self.fo6, [self.y])
        return h + self.y[:D]

    def body(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        h = self.embed(tok)
        kvi = 0
        for il in range(NL):
            if il % 4 == 3:
                h = self.attn(il, h, pos, kvi)
                kvi += 1
            else:
                h = self.gdn(il, h)
            h = self.ffn(il, h)
        return h

    def feed(self, tok, pos):
        """A prompt token whose logits are not needed."""
        self.body(tok, pos)

    def step(self, tok, pos):
        h = self.body(tok, pos)
        self.rmsfold(h, "output_norm.weight")
        return self.head()

    # ------------------------------------------------------------------------------------------- output head
    def head(self):
        """-> greedy next token id. Leaves the logits in self.ylog[:self.ylog_n], their argmax slot in self.aout and
        (reordered head) each slot's token id in self.ylog_map."""
        a = self.w["token_embd.weight"]
        if self.hsl is None:
            self.gemv(["token_embd.weight"], self.xp, [self.y])
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(VOCAB), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
            return int(self.aout[0])
        Rp = self.hsl["Rp"]
        M = self.shortlist
        if M <= 0 or M >= self.hsl["K"]:
            self.gemv(["token_embd.weight"], self.xp, [self.y])
            self.k.k_mask_neg((64,), (256,), (self.y, self.hs_permtok, np.int32(Rp)))
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(Rp), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, Rp, self.hs_permtok
            return int(self.hs_permtok_h[int(self.aout[0])])
        cs = self.hs_cent @ self.xp[:D]
        sel = cp.argsort(-cs)[:M].astype(cp.int32)
        cnt = self.hs_gcount[sel]
        off = (cp.cumsum(cnt) - cnt).astype(cp.int32)
        ng = int(cnt.sum())
        self.k.k_expand((M,), (128,), (sel, self.hs_gstart, self.hs_gcount, off, self.hs_glist))
        zr = np.int32(0)
        self.k.k_g4l((1,), (1024,), (a[0], a[1], self.hs_yc, a[2], a[0], a[1], self.hs_yc, zr, a[0], a[1], self.hs_yc,
                                     zr, self.xp, a[3], self.hs_glist, np.int32(ng), self.hs_permtok, self.hs_tokc))
        n = ng * 16
        self.k.k_mask_neg((16,), (256,), (self.hs_yc, self.hs_tokc, np.int32(n)))
        self.k.k_argmax((1,), (1024,), (self.hs_yc, np.int32(n), self.aout))
        self.ylog, self.ylog_n, self.ylog_map = self.hs_yc, n, self.hs_tokc
        return int(self.hs_tokc[int(self.aout[0])])

    def generate(self, ids, n):
        """Greedy: the n tokens after the prompt ids (fresh state)."""
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out
