"""E8 (uso interno): evalúa los modelos publicados por otros dos grupos sobre NUESTRAS 100 series de test.

Se reconstruyen sus redes y su normalización de la entrada a partir de sus ficheros (sin reentrenar)
y se simulan con nuestro simulador (idéntico al entorno). Así la comparación es pareada: mismas
series, mismas métricas.  Salida: results/peers.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "Taller_RL2 otro intento")
P2 = os.path.join(ROOT, "Taller_RL2 otro intento mas")
if "g1" in sys.argv:  # su paquete debe importarse antes que el nuestro (si no, el proceso se cae en Windows)
    sys.path.insert(0, P1)
    from rl_trader.prices import GENERADORES  # noqa: E402,F401
    from rl_trader.redes import Normalizador, crear_red  # noqa: E402,F401
sys.path.insert(0, ROOT)
from rl_trading.env import make_set  # noqa: E402
from rl_trading.evaluate import metrics, simulate  # noqa: E402

GEN_P1 = {"ciclo": "ciclo", "gbm": "browniano", "reversion": "reversion", "regimenes": "regimenes"}
GEN_P2 = {"ciclo": "tendencia_ciclo", "gbm": "browniano", "reversion": "reversion_media", "regimenes": "regimenes"}


class Peer1Policy:
    """Grupo 1: entrada [clip(tendencia/sigma, ±5), 2*pos-1]; RedQ/RedDuelo (Q) o actor-crítico (logits)."""

    def __init__(self, model, norm):
        self.model, self.norm = model, norm

    def __call__(self, obs, rng=None):
        x = torch.from_numpy(self.norm.entrada(obs[:, 0], obs[:, 1]))
        with torch.no_grad():
            out = self.model(x)
        out = out[0] if isinstance(out, tuple) else out
        return out.numpy().argmax(1)


class Peer2Policy:
    """Grupo 2: entrada [tendencia/sigma, pos]; DQN 2-64-3 o actor-crítico de una capa."""

    def __init__(self, net, scale, kind, mask=False):
        self.net, self.scale, self.kind, self.mask = net, scale, kind, mask

    def __call__(self, obs, rng=None):
        x = obs.astype(np.float32).copy()
        x[:, 0] /= self.scale
        with torch.no_grad():
            out = self.net(torch.from_numpy(x))
        v = (out[0] if self.kind == "a3c" else out).numpy().copy()
        if self.mask:  # su "double_canonical"/"mask_redundant": sin activo no vende, con activo no compra
            no_asset = obs[:, 1] < 0.5
            v[no_asset, 0] = -np.inf
            v[~no_asset, 2] = -np.inf
        return v.argmax(1)


class AC2(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.common = nn.Linear(2, hidden)
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.common(x))
        return self.pi(h), self.v(h)


def evaluate(pol, test):
    m = metrics(simulate(test, pol))
    return {k: m[k].tolist() for k in ("log_wealth", "sum_reward", "trades", "ret_pct", "max_dd", "exposure")}


def peers1(tests, out):
    d = torch.load(os.path.join(P1, "results", "modelos.pt"), weights_only=False, map_location="cpu")
    for gen, pg in GEN_P1.items():
        norm = Normalizador.ajustar(GENERADORES[pg])
        for key, v in d.items():
            g, algo, seed = key.split("|")
            if g != pg:
                continue
            pol = Peer1Policy(crear_red(v["tipo_red"], v["ocultas"], v["estado"]).eval(), norm)
            out.setdefault(f"G1|{algo}|mejor|{gen}", []).append(evaluate(pol, tests[gen]))


def peers2(tests, out):
    for gen, pg in GEN_P2.items():
        for path in sorted(glob.glob(os.path.join(P2, "results", "models", f"*_{pg}_s*.pt"))):
            name = os.path.basename(path)[:-3]
            algo = name.split("_")[0]
            variant = "mejor" if "mejor_punto" in name else "ultimo"
            d = torch.load(path, weights_only=True)
            cfg = d["cfg"]
            if algo == "dqn":
                net = nn.Sequential(nn.Linear(2, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 3))
                mask = cfg.get("double_canonical", False)
            else:
                net = AC2(cfg["hidden"])
                mask = cfg.get("mask_redundant", False)
            net.load_state_dict(d["state"])
            scale = d["scales"][0] if cfg.get("normalize_obs", True) else 1.0
            pol = Peer2Policy(net.eval(), scale, algo, mask)
            out.setdefault(f"G2|{algo}|{variant}|{gen}", []).append(evaluate(pol, tests[gen]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = {g: make_set(g, "test") for g in GEN_P1}
    path = os.path.join(ROOT, "results", "peers.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    if "g1" in sys.argv:
        peers1(tests, out)
    else:
        peers2(tests, out)
    with open(path, "w") as f:
        json.dump(out, f)
    for k, runs in sorted(out.items()):
        sr = np.mean([np.mean(r["sum_reward"]) for r in runs])
        lw = np.mean([np.mean(r["log_wealth"]) for r in runs])
        tr = np.mean([np.mean(r["trades"]) for r in runs])
        print(f"{k:35s} n={len(runs):2d}  Σr={sr:+.3f}  logW={lw:+.3f}  trades={tr:.1f}")
