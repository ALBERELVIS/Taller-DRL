# Taller RL Trader 2

Agentes de aprendizaje por refuerzo profundo que invierten en un único activo sobre el entorno `SimpleTradingEnv` del enunciado. El entorno no se modifica. DQN y A3C reciben el estado continuo `(tendencia, posición)` y se comparan, en los cuatro generadores de precios, con el agente aleatorio, *buy & hold*, Q-learning tabular y reglas de umbral.

Máster MIAX · Aprendizaje por Refuerzo · noviembre 2026.

Probado con Python 3.13 y PyTorch 2.6 (CPU).

## Entregables

| Fichero | Qué es |
|---|---|
| [`memoria/memoria.pdf`](memoria/memoria.pdf) | Memoria de 2 páginas. Las cifras salen de `results/*.csv`; no se escriben a mano. |
| [`solucion_taller_trader2.ipynb`](solucion_taller_trader2.ipynb) | Notebook de la solución: teoría, protocolo, resultados y ampliación. No reentrena. |
| [`defensa.md`](defensa.md) | Guion de la defensa oral (15 min) y cifras que hay que saberse. |

## Qué hace el trabajo

El estado del enunciado es la tendencia de los últimos 6 precios y si hay posición abierta. Las acciones son vender, mantener y comprar. La comisión es del 0,1 % y la recompensa es el rendimiento de la posición menos el coste de operar. Cada serie tiene 500 precios.

Los cuatro mercados son tendencia con ciclo, movimiento browniano geométrico, reversión a la media y regímenes alcista/bajista. Donde hay señal, la estrategia esperada es *momentum*, nada, contraria y una banda (U invertida), respectivamente. En el browniano no hay señal por encima de la comisión.

Los hiperparámetros se eligen en validación (40 series). El test son 100 series comunes a todos los agentes, con 5 semillas nuevas. La métrica principal es la suma de recompensas (Σr). Cada contraste usa Wilcoxon pareado y un intervalo bootstrap jerárquico, con una regla única de veredicto: gana, pierde, equivalente o no concluyente.

Más allá de los mínimos hay reglas de histéresis ajustadas por búsqueda exhaustiva, PPO, estado enriquecido (ventana de rendimientos, volatilidad y media móvil), comisión y riesgo, Dueling DQN, un agente para los cuatro mercados, Expected SARSA y el S&P 500 (entrenamiento 2001–2010, test 2011).

## Puesta en marcha

En PowerShell, desde esta carpeta:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q tests/
.\.venv\Scripts\python tests_sanity.py
```

`tests_sanity.py` comprueba que el simulador vectorizado coincide con `SimpleTradingEnv` y que *buy & hold* coincide con el notebook del enunciado. `tests/test_core.py` cubre comisión variable, causalidad de las features, la regla de veredicto y el oráculo.

El notebook solo lee `results/`. Para abrirlo:

```powershell
.\.venv\Scripts\python -m pip install jupyter
.\.venv\Scripts\python -m jupyter lab solucion_taller_trader2.ipynb
```

## Reproducir

Los entrenamientos ya están en `results/` (JSON por semilla, modelos en `results/models/`). El lanzador salta un trabajo si el JSON de salida existe. Cada proceso usa un hilo de BLAS. El segundo argumento es el número de trabajos en paralelo.

```powershell
# Referencias y ajuste (solo validación)
python -m experiments.baselines_eval
python -m experiments.tune dqn,qlearning 20
python -m experiments.tune a3c 4

# Evaluación final: 5 semillas × 4 mercados
python -m experiments.main final 20

# Ablaciones, más episodios, workers de A3C, S&P 500 y tiempos
python -m experiments.main ablations 14 dqn
python -m experiments.main iteration 5 dqn
python -m experiments.main iteration_a3c 5 a3c
python -m experiments.main iteration_a3c_reg 5 a3c
python -m experiments.main workers 1 a3c
python -m experiments.main sp500 6
python -m experiments.sp500_null
python -m experiments.main timing 1
python -m experiments.main terminal 12 dqn

# Tablas y figuras de los mínimos
python -m experiments.analysis
```

Ampliación (un cambio cada vez, sobre los hiperparámetros ya elegidos):

```powershell
python -m experiments.rules_eval
python -m experiments.tune ppo 8
python -m experiments.main final 4 ppo
python -m experiments.main ext_dqn 20 dqn
python -m experiments.main ext_a3c 2 a3c
python -m experiments.main ext_ppo_feat 4 ppo
python -m experiments.main ext_ppo 4 ppo
python -m experiments.main ext_ppo_snr 4 ppo
python -m experiments.main ext_onpolicy 6 qlearning
python -m experiments.main ext_dqn_feat_long 6 dqn
python -m experiments.analysis_ext
```

Informe y comprobaciones. `make_memoria.py` escribe el HTML y lo imprime a PDF con Edge en modo headless:

```powershell
python -m experiments.extras_peers
python -m experiments.stats_report
python build_notebook.py
python make_memoria.py
python check_memoria.py
```

`check_memoria.py` falla si aparece una cifra escrita a mano, si un veredicto no cuadra con la regla, si falta algún experimento de `results/inventory.csv` o si el PDF no tiene exactamente 2 páginas.

El S&P 500 se lee de `Referencias profesor/03-trading/` (`GSPC_2001-10.csv` y `GSPC_2011.csv`).

## Estructura

```
rl_trading/          entorno literal, codificación, agentes y estadística
  env.py             SimpleTradingEnv y los 4 generadores (sin modificar) + series por semilla
  encoding.py        tendencia / σ de entrenamiento y posición, sin discretizar
  dqn.py             replay, red objetivo, Double DQN, Huber, ε por episodio
  a3c.py             A3C asíncrono (procesos, Hogwild, Adam compartido, ventaja a n pasos)
  ppo.py             PPO con entornos vectorizados (ampliación)
  qlearning.py       Q-learning tabular del taller anterior
  baselines.py       aleatorio, buy & hold, momentum, contraria, oráculos
  rules.py           reglas de 1–4 umbrales
  features.py        estado enriquecido (causal)
  evaluate.py        simulador vectorizado y métricas
  stats.py           Wilcoxon, bootstrap, TOST, Holm, BH, Friedman–Nemenyi
experiments/         ajuste, trabajos en paralelo, análisis y tablas
results/             JSON, CSV, figuras y modelos
memoria/             memoria.html y memoria.pdf
memoria_text.py      texto de la memoria; las cifras son plantillas
make_memoria.py      HTML → PDF
check_memoria.py     consistencia de la memoria
build_notebook.py    genera el notebook desde notebook_text.py
defensa.md           guion oral
tests/               pytest
```

Las carpetas `Taller_RL2 otro intento` y `Taller_RL2 otro intento mas` son borradores anteriores. El trabajo actual es el de esta raíz.

## Protocolo, en corto

- Entrenamiento: una serie nueva por episodio. Semilla `1_000_000 + 100_000 * run_seed + episodio`.
- Validación: semillas desde 50 000 (40 series). Ahí se elige la configuración y el mejor checkpoint.
- Test: semillas desde 100 000 (100 series), las mismas para todos. El test no elige nada.
- Σr es lo que optimiza el agente. El log-riqueza y el cociente de capital son lo que ve el inversor: sumar rendimientos aritméticos no es lo mismo que ganar dinero.
- Familia confirmatoria (mínimos del enunciado): Holm. Familia exploratoria (el resto): Benjamini–Hochberg.
- δ de equivalencia = 0,02 de Σr.
