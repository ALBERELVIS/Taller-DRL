"""Batería de pruebas: python -m pytest -q tests/

Comprueba que las extensiones no alteran el problema del enunciado y que no hay fugas de
información del futuro.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trading.baselines import buy_and_hold  # noqa: E402
from rl_trading.env import GENERATORS, CostTradingEnv, SimpleTradingEnv, make_series, make_set  # noqa: E402
from rl_trading.evaluate import metrics, simulate  # noqa: E402
from rl_trading.features import SETS, FeatureEncoder, raw_features  # noqa: E402
from rl_trading.ppo import VecTradingEnv  # noqa: E402
from rl_trading.rules import Rule  # noqa: E402

GENS = list(GENERATORS)


def run_env(env, actions):
    obs, _ = env.reset()
    R, B, O = [], [env.balance], [obs]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        R.append(r), B.append(env.balance), O.append(obs)
        if term:
            break
    return np.array(R), np.array(B), np.array(O[:-1])


@pytest.mark.parametrize("gen", GENS)
@pytest.mark.parametrize("cost", [0.001, 0.0, 0.005])
def test_simulator_equals_env(gen, cost):
    """Simulador vectorizado == entorno (original para c=0.1%, CostTradingEnv para el resto)."""
    rng = np.random.default_rng(1)
    p = make_series(gen, 321)
    acts = rng.integers(0, 3, size=len(p) - 6)
    env = SimpleTradingEnv(p) if cost == 0.001 else CostTradingEnv(p, cost)
    r_env, b_env, o_env = run_env(env, acts)
    it = iter(acts)
    sim = simulate([p], lambda o, _r: [next(it)], cost=cost)
    assert np.allclose(r_env, sim["rewards"][0], atol=1e-12)
    assert np.allclose(b_env, sim["balances"][0], rtol=1e-12)
    assert np.allclose(o_env, sim["obs"][0])


@pytest.mark.parametrize("gen", GENS)
def test_cost_env_reproduces_original(gen):
    p = make_series(gen, 11)
    acts = np.random.default_rng(2).integers(0, 3, size=len(p) - 6)
    a = run_env(SimpleTradingEnv(p), acts)
    b = run_env(CostTradingEnv(p, 0.001), acts)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("gen", GENS)
def test_vec_env_equals_env(gen):
    """Entorno vectorizado de PPO == entorno original, serie a serie."""
    P = [make_series(gen, 500 + i) for i in range(4)]
    rng = np.random.default_rng(3)
    acts = rng.integers(0, 3, size=(len(P[0]) - 6, 4))
    venv = VecTradingEnv(P)
    o = venv.reset()
    R, B, O = [], [venv.balance.copy()], [o]
    done = False
    k = 0
    while not done:
        o, r, done = venv.step(acts[k])
        R.append(r), B.append(venv.balance.copy()), O.append(o)
        k += 1
    R, B, O = np.array(R), np.array(B), np.array(O[:-1])
    for i in range(4):
        r_env, b_env, o_env = run_env(SimpleTradingEnv(P[i]), acts[:, i])
        assert np.allclose(R[:, i], r_env, atol=1e-12)
        assert np.allclose(B[:, i], b_env, rtol=1e-12)
        assert np.allclose(O[:, i], o_env)


def test_commission_arithmetic():
    """Comprar y vender inmediatamente cuesta exactamente (1-c)^2 del capital (precio constante)."""
    p = np.full(20, 100.0)
    for c in (0.0, 0.001, 0.005):
        acts = iter([2, 0] + [1] * 20)
        sim = simulate([p], lambda o, _r: [next(acts)], cost=c)
        assert np.isclose(sim["balances"][0, -1], 1000 * (1 - c) ** 2)
        assert np.isclose(sim["rewards"][0].sum(), -2 * c)


@pytest.mark.parametrize("feats", SETS)
def test_features_are_causal(feats):
    """Cambiar precios futuros (> t) no cambia las variables en t."""
    P = np.array([make_series("regimenes", 77)])
    for t in (5, 12, 40, 300, 493):
        f0 = raw_features(P, t, feats)
        Q = P.copy()
        Q[:, t + 1:] *= 1.7
        assert np.array_equal(f0, raw_features(Q, t, feats))


@pytest.mark.parametrize("feats", ["ventana", "ma_vol"])
def test_feature_encoder_env_vs_batch(feats):
    """La codificación paso a paso (entrenamiento) coincide con la del simulador (evaluación)."""
    enc = FeatureEncoder("reversion", feats, n_series=3)
    p = make_series("reversion", 9)
    env = SimpleTradingEnv(p)
    obs, _ = env.reset()
    for _ in range(50):
        x_env = enc.from_env(obs, env)
        x_bat = enc.encode(obs[None], p[None], env.current_step)[0]
        assert np.allclose(x_env, x_bat)
        obs, *_ = env.step(1)


def test_reproducible_series():
    for g in GENS:
        assert np.array_equal(make_series(g, 5), make_series(g, 5))
    assert not np.array_equal(make_set("gbm", "val")[0], make_set("gbm", "test")[0])


def test_buy_and_hold_matches_notebook():
    for g in GENS:
        p = make_series(g, 7)
        bh = (1000.0 * 0.999) / p[5] * p[-1]
        sim = simulate([p], buy_and_hold)
        assert abs(sim["balances"][0, -1] - bh) / bh < 1e-9


def test_rules_logic():
    obs = np.array([[2.0, 0], [2.0, 1], [-2.0, 1], [-0.5, 1], [-0.5, 0]], np.float32)
    h = Rule(1.0, "mom_h", 1.0, -1.0)(obs)
    assert list(h) == [2, 1, 0, 1, 1]  # compra >1, vende <-1, mantiene en la banda
    b = Rule(1.0, "band_h", -1.0, 1.0, -1.5, 2.5)(np.array([[0.0, 0], [2.0, 0], [2.0, 1], [3.0, 1]], np.float32))
    assert list(b) == [2, 1, 1, 0]


def test_expected_sarsa_equals_qlearning_without_exploration():
    from rl_trading.qlearning import train_qlearning
    cfg = dict(episodes=3, eps_start=0.0, eps_min=0.0, n_bins=6)
    q = train_qlearning("reversion", {**cfg, "target": "q"}, 0)["Q"]
    s = train_qlearning("reversion", {**cfg, "target": "expected_sarsa"}, 0)["Q"]
    assert np.array_equal(q, s)
    e = train_qlearning("reversion", {**cfg, "eps_start": 0.5, "eps_min": 0.5, "target": "expected_sarsa"}, 0)["Q"]
    assert not np.array_equal(q, e)


def test_metrics_trades_and_exposure():
    p = np.linspace(100, 110, 30)
    acts = iter([2] * 5 + [0] * 5 + [2] * 30)
    m = metrics(simulate([p], lambda o, _r: [next(acts)]))
    assert m["trades"][0] == 3


# --------------------------------------------------------------------- evaluación estadística global
def test_verdict_rule_on_synthetic_effects():
    from rl_trading.stats import contrast, verdict
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 100)                    # efecto común de la serie (pareado)
    noise = lambda s: rng.normal(0, s, (5, 100))  # noqa: E731
    base = series + noise(0.005)
    same = contrast(series + noise(0.005), base, delta=0.02)
    assert same["equivalent"] and verdict({**same, "p_adj": same["p"]}) == "equivalente"
    up = contrast(base + 0.5 + noise(0.1), base, delta=0.02)
    assert verdict({**up, "p_adj": up["p"]}) == "gana" and up["ci_low"] > 0
    down = contrast(base - 0.5 + noise(0.1), base, delta=0.02)
    assert verdict({**down, "p_adj": down["p"]}) == "pierde"
    vague = contrast(base + noise(1.0), base, delta=0.02)
    assert verdict({**vague, "p_adj": vague["p"]}) == "no concluyente"


def test_friedman_detects_known_ordering():
    from rl_trading.stats import friedman_nemenyi
    rng = np.random.default_rng(1)
    M = rng.normal(0, 1, (100, 4)) + np.array([0.0, 0.3, 0.6, 0.9])
    f = friedman_nemenyi(M)
    assert f["p"] < 1e-6
    assert list(np.argsort(f["avg_rank"])) == [3, 2, 1, 0]
    assert f["avg_rank"][0] - f["avg_rank"][3] > f["cd"]
    null = friedman_nemenyi(rng.normal(0, 1, (100, 4)))
    assert null["p"] > 0.01


def test_nemenyi_constants_match_scipy():
    from scipy.stats import studentized_range
    from rl_trading.stats import Q_NEMENYI
    for k, q in Q_NEMENYI.items():
        assert abs(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) - q) < 2e-3


def test_gbm_hidden_oracle_is_buy_and_hold():
    from rl_trading.baselines import HiddenOracle
    P = [make_series("gbm", 100_000 + i) for i in range(20)]
    a, b = metrics(simulate(P, HiddenOracle("gbm"))), metrics(simulate(P, buy_and_hold))
    assert np.allclose(a["sum_reward"], b["sum_reward"]) and np.allclose(a["log_wealth"], b["log_wealth"])


def test_regime_states_follow_the_generator():
    from rl_trading.baselines import regime_states
    vol = {0: [], 1: []}
    for seed in range(5):
        make_series("regimenes", seed)
        after_series = np.random.rand()
        s = regime_states(seed)
        assert np.random.rand() == after_series  # mismas extracciones aleatorias que el generador
        P = make_series("regimenes", seed)
        r = P[1:] / P[:-1] - 1
        for k in (0, 1):
            vol[k] += list(r[s[1:] == k])
    assert np.std(vol[1]) > 2 * np.std(vol[0])  # bajista: sigma 2.5% frente a 1%


def test_single_seed_ensemble_equals_that_seed():
    import glob
    if not glob.glob("results/models/dqn_reversion_s*.pt"):
        pytest.skip("sin modelos guardados")
    from experiments.extras_peers import Ensemble, load_policies
    pols, _ = load_policies("dqn", "reversion")
    P = [make_series("reversion", 100_000 + i) for i in range(20)]
    one, ens = metrics(simulate(P, pols[0])), metrics(simulate(P, Ensemble(pols[:1], actor=False)))
    assert np.array_equal(one["sum_reward"], ens["sum_reward"])
