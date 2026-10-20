"""kindi verify: greedy output vs llama.cpp's reference for the same model file (reference.json: 4 prompts, the first
40 tokens llama.cpp generated from them at temperature 0). Every token must match."""
import json
import os
import time


def run(cfg, log=print):
    import cupy as cp
    from .engine import Engine
    import numpy as np
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")))
    pack = cfg["pack"]
    cl = os.path.join(pack, "head_clusters.npz")
    eng = Engine(pack, ctx=1024, clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log)
    ok_all, tok, dt = True, 0, 0.0
    for i, r in enumerate(ref["prompts"]):
        cp.cuda.Device().synchronize()
        t0 = time.time()
        got = eng.generate(r["ids"], len(r["expected"]))
        cp.cuda.Device().synchronize()
        t = time.time() - t0
        n = sum(1 for a, b in zip(got, r["expected"]) if a == b)
        ok = got == r["expected"]
        ok_all &= ok
        tok += len(r["ids"]) + len(got)
        dt += t
        log("prompt %d %-60s %s  %d/%d tokens match  (%.1fs)" % (i + 1, repr(r["prompt"])[:60], "PASS" if ok else "FAIL",
                                                                  n, len(r["expected"]), t))
        if not ok:
            j = next(k for k, (a, b) in enumerate(zip(got, r["expected"])) if a != b)
            log("   first difference at token %d" % j)
    log("%s - %.1f tokens/s over the run" % ("ALL PASS: token-exact with llama.cpp" if ok_all else "FAIL", tok / dt))
    return 0 if ok_all else 1
