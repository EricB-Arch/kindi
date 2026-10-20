"""Load-time weight layouts (host side, before upload).

repack(): every ternary tensor from the pack layout (bytes [slot][row], scales [phase][row]) to the GEMV layout:
  weights: 4 rows interleaved per 16 bytes, [R/4][C/16][4 rows][4 slots] - one 128-bit load gives a lane one
           16-column word of 4 rows;
  scales:  per (16-row group, 512-column phase), the 64 fp16 values the 32 lanes of a warp need, in lane order, so
           a warp reads them with one coalesced 128-byte load (lane l holds rows base(l), base(l)+1 of its group,
           base(l) = (l&1)*8 + (l>>1&1)*4 + (l>>2&1)*2, for the 128-column group of lane l / 8).
reorder_head(): optional, for the fast head: the head/embedding rows in cluster order, each cluster padded to whole
  16-row groups (padding rows: zero weights, token id -1)."""
import numpy as np


def _base_row(lane):
    return (lane & 1) * 8 + ((lane >> 1) & 1) * 4 + ((lane >> 2) & 1) * 2


def scales16(Sm):
    """Sm: [R, P] uint16 (fp16 bits) -> the GEMV scale layout, flat [R/16 * NPH * 32 * 2]."""
    R, P = Sm.shape
    NPH = P // 4
    lanes = np.arange(32)
    rows = np.array([_base_row(l) for l in lanes])
    grp = lanes >> 3
    g16 = np.arange(R // 16)[:, None, None, None]
    ph = np.arange(NPH)[None, :, None, None]
    r = g16 * 16 + rows[None, None, :, None] + np.arange(2)[None, None, None, :]
    p = ph * 4 + grp[None, None, :, None]
    return np.ascontiguousarray(Sm[r, p]).reshape(-1)


def repack(bi, bc, dp):
    """In place on the host arrays. Returns {name: (rows, ne0)}."""
    done = {}
    for name, rec in bi.items():
        R, C = rec["rows"], rec["ne0"]
        o, d = rec["bc_off"], rec["dpm_off"] // 2
        if R % 16 or C % 512 or o % 16 or d % 2:
            raise ValueError("tensor %s (%d x %d) does not fit the GEMV layout" % (name, R, C))
        ns, P = C // 4, C // 128
        w = bc[o:o + ns * R].reshape(ns, R).T.reshape(R // 4, 4, ns // 4, 4).transpose(0, 2, 1, 3)
        bc[o:o + ns * R] = np.ascontiguousarray(w).reshape(-1)
        dp[d:d + P * R] = scales16(np.ascontiguousarray(dp[d:d + P * R].reshape(P, R).T))
        done[name] = (R, C)
    return done


def reorder_head(bi, bc, dp, clusters):
    """Before repack(): rebuild the head region (the first tensor, token_embd) in cluster order, padded, and shift
    every other tensor's offsets. clusters: dict with 'assign' [R] and 'centroids' [K, C].
    Returns (bc, dp, meta)."""
    assign = np.asarray(clusters["assign"]).astype(np.int64)
    cent = np.asarray(clusters["centroids"]).astype(np.float32)
    rec = bi["token_embd.weight"]
    R, C = rec["rows"], rec["ne0"]
    ns, P = C // 4, C // 128
    if rec["bc_off"] != 0 or rec["dpm_off"] != 0 or len(assign) != R:
        raise ValueError("cluster file does not match this pack")
    K = cent.shape[0]
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=K)
    padded = (counts + 15) // 16 * 16
    Rp = int(padded.sum())
    starts = np.concatenate([[0], np.cumsum(padded)[:-1]])
    src_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    permtok = np.full(Rp, -1, dtype=np.int32)
    for k in range(K):
        permtok[starts[k]:starts[k] + counts[k]] = order[src_start[k]:src_start[k] + counts[k]]
    real = permtok >= 0
    permpos = np.empty(R, dtype=np.int32)
    permpos[permtok[real]] = np.nonzero(real)[0]
    nb = np.full((ns, Rp), 40, dtype=np.uint8)            # 40 = four zero trits
    nsc = np.zeros((P, Rp), dtype=np.uint16)
    nb[:, real] = bc[:ns * R].reshape(ns, R)[:, permtok[real]]
    nsc[:, real] = dp[:P * R].reshape(P, R)[:, permtok[real]]
    dbc, ddp = (Rp - R) * ns, (Rp - R) * P
    bc = np.concatenate([nb.reshape(-1), bc[ns * R:]])
    dp = np.concatenate([nsc.reshape(-1), dp[P * R:]])
    for name, r in bi.items():
        if name != "token_embd.weight":
            r["bc_off"] += dbc
            r["dpm_off"] += ddp * 2
    rec["rows"] = Rp
    meta = {"R": R, "Rp": Rp, "K": K, "cent": cent, "permtok": permtok, "permpos": permpos,
            "gstart": (starts // 16).astype(np.int32), "gcount": (padded // 16).astype(np.int32)}
    return bc, dp, meta
