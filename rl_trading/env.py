"""Entorno y generadores de precios del taller.

`SimpleTradingEnv` y las cuatro funciones generadoras están copiadas literalmente de
`taller_trader2.ipynb` (el profesor pidió explícitamente no modificar el entorno).
Lo único que se añade es la forma reproducible de generar series (`make_series`), que
fija la semilla global de NumPy antes de llamar al generador original, y un cargador
de series reales del S&P 500 (extensión opcional).
"""
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ----------------------------------------------------------------------------------
# Entorno original (literal del notebook)
# ----------------------------------------------------------------------------------
class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices):
        super(SimpleTradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices) - 1

        # Acciones: 0 = Vender, 1 = Mantener, 2 = Comprar
        self.action_space = spaces.Discrete(3)

        # Estado: [tendencia_precio (-1 a 1), posicion_actual (0 o 1)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 5  # Empezar con suficiente historia para la media
        self.position = 0      # 0: sin activo, 1: con activo

        # Gestión del capital
        self.initial_balance = 1000.0
        self.balance = self.initial_balance  # Valor total del portafolio (Efectivo + Activos)
        self.cash = self.initial_balance     # Dinero en efectivo disponible
        self.shares = 0.0                    # Cantidad de acciones poseídas

        return self._get_obs(), {}

    def _get_obs(self):
        # Calcular tendencia simple basada en los últimos 5 pasos
        window = self.prices[self.current_step - 5 : self.current_step + 1]
        trend = (window[-1] - np.mean(window)) / np.mean(window)
        return np.array([np.clip(trend, -1.0, 1.0), self.position], dtype=np.float32)

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]

        reward = 0.0
        cost_rate = 0.001  # Comisión del 0.1% por transacción

        # 1. Ejecutar acciones y actualizar efectivo/acciones
        if action == 2 and self.position == 0:  # Comprar
            # Usar todo el efectivo disponible para comprar acciones al precio anterior
            commission = self.cash * cost_rate
            effective_cash = self.cash - commission
            self.shares = effective_cash / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= cost_rate

        elif action == 0 and self.position == 1:  # Vender
            # Vender todas las acciones al precio anterior
            sale_value = self.shares * prev_price
            commission = sale_value * cost_rate
            self.cash = sale_value - commission
            self.shares = 0.0
            self.position = 0
            reward -= cost_rate

        # 2. Calcular recompensa por variación del precio si mantiene la posición
        if self.position == 1:
            price_change = (curr_price - prev_price) / prev_price
            reward += price_change

        # 3. Actualizar el balance total (Net Worth) al finalizar el paso
        if self.position == 1:
            self.balance = self.cash + (self.shares * curr_price)
        else:
            self.balance = self.cash

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Opcional: puedes pasar el balance actual en el diccionario de info
        info = {"balance": self.balance}

        return self._get_obs(), reward, terminated, truncated, info


class CostTradingEnv(SimpleTradingEnv):
    """EXTENSIÓN (no es el entorno del enunciado): idéntica dinámica con comisión configurable.

    Solo se usa en el experimento de sensibilidad a la comisión; con cost_rate=0.001 reproduce
    exactamente `SimpleTradingEnv` (comprobado en tests/)."""

    def __init__(self, prices, cost_rate=0.001):
        super().__init__(prices)
        self.cost_rate = cost_rate

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]
        reward = 0.0
        c = self.cost_rate
        if action == 2 and self.position == 0:
            self.shares = (self.cash - self.cash * c) / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= c
        elif action == 0 and self.position == 1:
            sale_value = self.shares * prev_price
            self.cash = sale_value - sale_value * c
            self.shares = 0.0
            self.position = 0
            reward -= c
        if self.position == 1:
            reward += (curr_price - prev_price) / prev_price
            self.balance = self.cash + (self.shares * curr_price)
        else:
            self.balance = self.cash
        terminated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, False, {"balance": self.balance}


def make_env(prices, cost=0.001):
    """Entorno original salvo que se pida otra comisión (extensión)."""
    return SimpleTradingEnv(prices) if cost == 0.001 else CostTradingEnv(prices, cost)


# ----------------------------------------------------------------------------------
# Generadores originales (literales del notebook)
# ----------------------------------------------------------------------------------
def generar_precios(n_steps=1000):
    # Combinar una tendencia suave (drift) + componente cíclico + ruido
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)

    # Precio base acumulado
    retornos = drift + ciclo + ruido
    precios = 100 * np.cumprod(1 + retornos)
    return precios


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    """
    Genera precios usando Movimiento Browniano Geométrico.
    - mu: Tendencia o deriva esperada (drift).
    - sigma: Volatilidad (desviación estándar de los retornos).
    """
    dt = 1.0
    # Retornos logarítmicos distribuidos normalmente
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    precios = start_price * np.cumprod(np.exp(retornos))
    return precios


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    """
    Genera precios que tienden a revertir hacia un valor medio.
    - mean_price: El precio objetivo o de equilibrio.
    - kappa: Velocidad de reversión (cuánto más alto, más rápido vuelve al valor medio).
    - volatility: Magnitud del ruido aleatorio añadido en cada paso.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    for t in range(1, n_steps):
        # Fuerza de atracción hacia la media + ruido aleatorio
        delta = kappa * (mean_price - precios[t-1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta

        # Evitar precios negativos o cero por seguridad
        precios[t] = max(precios[t], 0.1)

    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    """
    Genera precios alternando entre regímenes alcistas y bajistas de forma estocástica.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    # Estado inicial: 0 = Alcista, 1 = Bajista
    estado = 0

    retornos = []
    for _ in range(n_steps):
        # Probabilidad de cambiar de régimen en cada paso
        if np.random.rand() < 0.02:
            estado = 1 - estado

        if estado == 0:  # Régimen alcista
            ret = np.random.normal(0.001, 0.01)
        else:            # Régimen bajista (caídas con más ruido)
            ret = np.random.normal(-0.0015, 0.025)

        retornos.append(ret)

    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)

    # Asegurar que no haya valores negativos
    precios = np.maximum(precios, 0.1)
    return precios


# ----------------------------------------------------------------------------------
# Generación reproducible de series y particiones de semillas
# ----------------------------------------------------------------------------------
N_STEPS = 500

GENERATORS = {
    "ciclo": generar_precios,
    "gbm": generar_precios_browniano,
    "reversion": generar_precios_reversion_media,
    "regimenes": generar_precios_regimenes,
}
GEN_LABELS = {
    "ciclo": "Tendencia + ciclo",
    "gbm": "Browniano (GBM)",
    "reversion": "Reversión a la media",
    "regimenes": "Regímenes alcista/bajista",
    "sp500": "S&P 500 (real)",
}

# Semillas disjuntas: entrenamiento, validación (ajuste) y test (evaluación final)
VAL_SEED0, N_VAL = 50_000, 40
TEST_SEED0, N_TEST = 100_000, 100


def train_seed(run_seed, episode):
    return 1_000_000 + 100_000 * run_seed + episode


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "Referencias profesor", "03-trading")
_SP500_CACHE = {}


def _load_close(name):
    if name not in _SP500_CACHE:
        import pandas as pd
        df = pd.read_csv(os.path.join(_DATA_DIR, name))
        _SP500_CACHE[name] = df["Close"].to_numpy(dtype=float)
    return _SP500_CACHE[name]


def sp500_train_series(seed, n_steps=N_STEPS):
    """Ventana aleatoria de `n_steps` días del S&P 500 2001-2010 (entrenamiento)."""
    close = _load_close("GSPC_2001-10.csv")
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(close) - n_steps)
    return close[start:start + n_steps].copy()


def sp500_test_series():
    """Año 2011 completo del S&P 500 (test fuera de muestra, posterior al entrenamiento)."""
    return _load_close("GSPC_2011.csv").copy()


def make_series(gen, seed, n_steps=N_STEPS):
    if gen == "sp500":
        return sp500_train_series(seed, n_steps)
    np.random.seed(seed)
    return GENERATORS[gen](n_steps)


MULTI_GENS = ("ciclo", "gbm", "reversion", "regimenes")


def train_series(gen, run_seed, ep):
    """Serie de entrenamiento del episodio `ep`. gen='multi' alterna los 4 generadores."""
    if gen == "multi":
        return make_series(MULTI_GENS[ep % 4], train_seed(run_seed, ep))
    return make_series(gen, train_seed(run_seed, ep))


def make_set(gen, kind):
    """Conjunto fijo de series: kind in {'val', 'test'}. Idéntico para todos los agentes."""
    if gen == "sp500":
        if kind == "test":
            return [sp500_test_series()]
        # validación: ventanas de 2001-10 con semillas reservadas
        return [sp500_train_series(VAL_SEED0 + i, 252) for i in range(N_VAL)]
    seed0, n = (VAL_SEED0, N_VAL) if kind == "val" else (TEST_SEED0, N_TEST)
    return [make_series(gen, seed0 + i) for i in range(n)]
