"""Prompt cache for a recurrent-hybrid model.

Bonsai's Gated-DeltaNet layers carry a recurrent state that cannot be rewound, so prefix reuse works on checkpoints:
  * checkpoints: the GDN state is copied to host RAM every `checkpoint_every` tokens and at the end of each prompt;
    a request resumes from the last checkpoint at or before its first differing token (the attention K/V cache is
    position-indexed and stays valid for any prefix);
  * parked conversations: when a request is unrelated to the active conversation (shares < 50 % of it), the active
    one - tokens, checkpoints, end state and K/V - is moved to host RAM (`park_max` kept) and resumed when its next
    turn arrives (agents alternate between their main prompt and side prompts);
  * saved openings: when two requests share >= `prefix_min` tokens and then differ, the full state at that checkpoint
    is written to disk (`prefix_max` kept, least recently used dropped), so the next session that starts the same way
    (same system prompt and tools) loads it in a fraction of a second, also after a restart.
All of it is exact: resuming gives the same tokens as reading the prompt from scratch."""
import hashlib
import json
import os
import shutil
import time

import numpy as np
import cupy as cp


class Cancelled(Exception):
    """The client disconnected; the engine state is consistent (every processed token is in Runner.tokens)."""


def _lcp(a, b):
    n = min(len(a), len(b))
    L = 0
    while L < n and a[L] == b[L]:
        L += 1
    return L


class Openings(object):
    """Saved prompt openings on disk: tokens[:p] and the engine state after them (GDN state + K/V below p)."""

    def __init__(self, eng, root, keep, log):
        self.eng, self.root, self.keep, self.log = eng, root, keep, log
        self.entries = {}
        try:
            os.makedirs(root, exist_ok=True)
            for key in os.listdir(root):
                d = os.path.join(root, key)
                try:
                    meta = json.load(open(os.path.join(d, "meta.json")))
                    if meta["p"] < eng.ctx:
                        self.entries[key] = {"p": meta["p"], "used": meta.get("used", 0),
                                             "tokens": np.load(os.path.join(d, "tokens.npy"))}
                except Exception:  # noqa: BLE001 - a damaged entry is skipped, not fatal
                    continue
        except OSError as e:
            log("saved openings disabled: %s" % e)
        log("saved openings: %d in %s" % (len(self.entries), root))

    @staticmethod
    def key_of(tokens):
        return hashlib.sha1(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:20]

    def has(self, tokens):
        return self.key_of(tokens) in self.entries

    def best(self, T, lo, hi):
        """The longest saved opening with lo < p <= hi that T starts with: (key, p) or None."""
        if not self.entries:
            return None
        Ta = np.asarray(T[:hi], dtype=np.int32)
        best = None
        for key, e in self.entries.items():
            p = e["p"]
            if lo < p <= hi and (best is None or p > best[1]) and np.array_equal(e["tokens"], Ta[:p]):
                best = (key, p)
        return best

    def save(self, tokens, Ms, Cs):
        p, key = len(tokens), self.key_of(tokens)
        if key in self.entries or self.keep <= 0:
            return
        t0 = time.time()
        d = os.path.join(self.root, key)
        try:
            os.makedirs(d, exist_ok=True)
            arrs = {}
            for i, m in enumerate(Ms):
                arrs["M%d" % i] = m
            for i, c in enumerate(Cs):
                arrs["C%d" % i] = c
            for i in range(len(self.eng.Kc)):
                arrs["K%d" % i] = cp.asnumpy(self.eng.Kc[i][:p])
                arrs["V%d" % i] = cp.asnumpy(self.eng.Vc[i][:p])
            np.savez(os.path.join(d, "state.npz"), **arrs)
            toks = np.asarray(tokens, dtype=np.int32)
            np.save(os.path.join(d, "tokens.npy"), toks)
            now = time.time()
            json.dump({"p": p, "used": now, "created": now}, open(os.path.join(d, "meta.json"), "w"))
            self.entries[key] = {"p": p, "used": now, "tokens": toks}
            self.log("saved an opening of %d tokens (%.1fs)" % (p, time.time() - t0))
        except Exception as e:  # noqa: BLE001 - a full disk must not break serving
            self.log("could not save an opening: %s" % e)
            shutil.rmtree(d, ignore_errors=True)
            return
        while len(self.entries) > self.keep:
            old = min(self.entries, key=lambda k: self.entries[k]["used"])
            self.entries.pop(old, None)
            shutil.rmtree(os.path.join(self.root, old), ignore_errors=True)

    def load(self, key):
        t0 = time.time()
        eng = self.eng
        d = os.path.join(self.root, key)
        z = np.load(os.path.join(d, "state.npz"))
        p = self.entries[key]["p"]
        Ms = [z["M%d" % i] for i in range(len(eng.M))]
        Cs = [z["C%d" % i] for i in range(len(eng.conv))]
        for a, h in zip(eng.M, Ms):
            a.set(h)
        for a, h in zip(eng.conv, Cs):
            a.set(h)
        for i in range(len(eng.Kc)):
            eng.Kc[i][:p] = cp.asarray(z["K%d" % i])
            eng.Vc[i][:p] = cp.asarray(z["V%d" % i])
        self.entries[key]["used"] = time.time()
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
            meta["used"] = time.time()
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        except Exception:  # noqa: BLE001
            pass
        self.log("loaded a saved opening of %d tokens (%.1fs)" % (p, time.time() - t0))
        return p, Ms, Cs


class Runner(object):
    """The engine plus the cache. self.tokens = the tokens whose effect is in the engine state."""

    def __init__(self, eng, cfg, pack_id, log, progress=None):
        self.eng, self.log = eng, log
        self.every = max(16, int(cfg["checkpoint_every"]))
        self.park_max = int(cfg["park_max"])
        self.prefix_min = int(cfg["prefix_min"])
        self.progress = progress or {}
        self.tokens, self.ckpt, self.parked = [], {}, []
        self.openings = Openings(eng, os.path.join(cfg["state_dir"], "openings", pack_id), int(cfg["prefix_max"]), log)

    # ---- state copies
    def _save_ckpt(self, pos):
        if pos not in self.ckpt:
            self.ckpt[pos] = ([cp.asnumpy(m) for m in self.eng.M], [cp.asnumpy(c) for c in self.eng.conv])

    def _set_state(self, Ms, Cs):
        for a, h in zip(self.eng.M, Ms):
            a.set(h)
        for a, h in zip(self.eng.conv, Cs):
            a.set(h)

    def _park(self):
        n = len(self.tokens)
        if n < self.prefix_min or self.park_max <= 0:
            return
        toks = list(self.tokens)
        self.parked = [P for P in self.parked if _lcp(P["tokens"], toks) < 0.9 * len(P["tokens"])]
        self.parked.insert(0, {
            "tokens": toks, "ckpt": dict(self.ckpt),
            "end": ([cp.asnumpy(m) for m in self.eng.M], [cp.asnumpy(c) for c in self.eng.conv]),
            "K": [cp.asnumpy(k[:n]) for k in self.eng.Kc], "V": [cp.asnumpy(v[:n]) for v in self.eng.Vc]})
        del self.parked[self.park_max:]

    def _resume_in(self, P, T):
        L = _lcp(P["tokens"], T)
        if L == len(P["tokens"]) and L < len(T):
            return L
        c = [p for p in P["ckpt"] if p <= min(L, len(T) - 1)]
        return max(c) if c else 0

    def _unpark(self, i, pos):
        P = self.parked.pop(i)
        self._set_state(*(P["end"] if pos == len(P["tokens"]) else P["ckpt"][pos]))
        for l in range(len(self.eng.Kc)):
            self.eng.Kc[l][:pos] = cp.asarray(P["K"][l][:pos])
            self.eng.Vc[l][:pos] = cp.asarray(P["V"][l][:pos])
        self.tokens = P["tokens"][:pos]
        self.ckpt = {p: v for p, v in P["ckpt"].items() if p <= pos}

    # ---- the request path
    def prepare(self, T, boundary, opening_end=None, cancel=None):
        """Make the engine state = T processed. Returns (greedy next token, tokens reused).
        boundary: where the generation prompt starts (checkpointed too).
        opening_end: where T's first message (system prompt + tools) ends. An opening is saved only when the two
        requests part before that point (+ a few tokens): a new session, or an agent that edits its system prompt -
        not the next turn of the same conversation, which always parts near its own end."""
        if opening_end is None:
            opening_end = len(T)
        S = self.tokens
        nS = len(S)
        L = _lcp(S, T)
        if L == nS and L < len(T):
            a_start = L
        else:
            c = [p for p in self.ckpt if p <= min(L, len(T) - 1)]
            a_start = max(c) if c else 0
        best_i, best_p = None, a_start
        for i, P in enumerate(self.parked):
            p = self._resume_in(P, T)
            if p > best_p:
                best_i, best_p = i, p
        if L < nS and L < 0.5 * nS:                                  # an unrelated prompt: park this conversation
            chosen = self.parked[best_i] if best_i is not None else None
            self._park()
            if chosen is not None:
                best_i = next((i for i, P in enumerate(self.parked) if P is chosen), None)
        if best_i is not None:
            self._unpark(best_i, best_p)
            nS = len(self.tokens)
            L = _lcp(self.tokens, T)
        if L == len(self.tokens) and L < len(T):
            start = L
        else:
            c = [p for p in self.ckpt if p <= min(L, len(T) - 1)]
            if c:
                start = max(c)
                self._set_state(*self.ckpt[start])
                del self.tokens[start:]
                if (start >= self.prefix_min and L < nS and L <= opening_end + 16
                        and not self.openings.has(self.tokens[:start])):
                    self.openings.save(self.tokens[:start], *self.ckpt[start])
            else:
                self.eng.reset()
                self.tokens = []
                start = 0
        for p in [p for p in self.ckpt if p > start]:
            del self.ckpt[p]
        hit = self.openings.best(T, start, len(T) - 1)
        if hit is not None:
            p, Ms, Cs = self.openings.load(hit[0])
            self.tokens = list(T[:p])
            self.ckpt = {p: (Ms, Cs)}
            start = p
        nxt = None
        self.progress.update(phase="reading prompt", cached=start, read=0)
        for pos in range(start, len(T)):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client went away after %d of %d prompt tokens" % (pos - start, len(T) - start))
            if pos > start and (pos % self.every == 0 or pos == boundary):
                self._save_ckpt(pos)
            if pos < len(T) - 1:
                self.eng.feed(T[pos], pos)
            else:
                nxt = self.eng.step(T[pos], pos)
            self.tokens.append(T[pos])
            self.progress["read"] = pos + 1 - start
        self._save_ckpt(len(T))
        return nxt, start

    def step(self, tok):
        nxt = self.eng.step(tok, len(self.tokens))
        self.tokens.append(tok)
        return nxt
