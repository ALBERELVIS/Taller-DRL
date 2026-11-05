"""Lanza muchos `run_job` en paralelo (subprocesos, 1 hilo de BLAS cada uno)."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(algo, gen, seed, cfg, out, test=False, save_model=None):
    return dict(algo=algo, gen=gen, seed=seed, cfg=cfg, out=out, test=test, save_model=save_model)


def _run(j):
    if os.path.exists(j["out"]):
        return j["out"], 0.0, "cached"
    cmd = [sys.executable, "-m", "experiments.run_job", "--algo", j["algo"], "--gen", j["gen"],
           "--seed", str(j["seed"]), "--cfg", json.dumps(j["cfg"]), "--out", j["out"]]
    if j["test"]:
        cmd.append("--test")
    if j["save_model"]:
        cmd += ["--save-model", j["save_model"]]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return j["out"], time.time() - t0, "ERROR: " + p.stderr[-1500:]
    return j["out"], time.time() - t0, "ok"


def run_all(jobs, max_parallel=30):
    t0 = time.time()
    n = len(jobs)
    errors = []
    with ThreadPoolExecutor(max_parallel) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out, dt, status = f.result()
            if status.startswith("ERROR"):
                errors.append((out, status))
            print(f"[{i}/{n}] {time.time() - t0:7.1f}s {dt:6.1f}s {status[:200]} {os.path.basename(out)}", flush=True)
    for out, st in errors:
        print("FAILED", out, st)
    return errors
