"""Pruebas de sanidad: simulador vectorizado == SimpleTradingEnv original; B&H == notebook."""
import numpy as np

from rl_trading.env import SimpleTradingEnv, make_series, GENERATORS
from rl_trading.evaluate import simulate


def run_env(prices, actions):
    env = SimpleTradingEnv(prices)
    obs, _ = env.reset()
    rewards, balances, observations = [], [env.balance], [obs]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        balances.append(env.balance)
        observations.append(obs)
        if term:
            break
    return np.array(rewards), np.array(balances), np.array(observations[:-1])


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    for gen in GENERATORS:
        for s in range(5):
            p = make_series(gen, 123 + s)
            T = len(p) - 6
            acts = rng.integers(0, 3, size=T)
            r_env, b_env, o_env = run_env(p, acts)
            it = iter(acts)
            sim = simulate([p], lambda o, _r: [next(it)])
            assert len(r_env) == T == sim["rewards"].shape[1]
            worst = max(worst, np.abs(r_env - sim["rewards"][0]).max(),
                        np.abs(b_env - sim["balances"][0]).max() / b_env.max(),
                        np.abs(o_env - sim["obs"][0]).max())
        # buy & hold del notebook
        p = make_series(gen, 7)
        bh_nb = (1000.0 * 0.999) / p[5] * p[-1]
        sim = simulate([p], lambda o, _r: np.full(len(o), 2))
        assert abs(sim["balances"][0, -1] - bh_nb) / bh_nb < 1e-9, (gen, sim["balances"][0, -1], bh_nb)
    # reproducibilidad de las series
    assert np.allclose(make_series("regimenes", 5), make_series("regimenes", 5))
    print("OK. Máxima discrepancia simulador vs entorno:", worst)


if __name__ == "__main__":
    main()
