# Guion de defensa oral (15 min) — Taller RL Trader 2

## 1. Estructura de la presentación (≈10 min + 5 de preguntas)

1. **(1 min) Problema y restricciones.** Mismo entorno que el taller de Luis, sin modificar; estado continuo
   (tendencia, posición), sin discretizar; 4 generadores; DQN y A3C; comparación con aleatorio, B&H y Q-learning.
2. **(2 min) Antes de entrenar: ¿qué se puede aprender?** Figura de predictibilidad por deciles:
   ciclo = momentum, GBM = nada, reversión = contraria, regímenes = U invertida (no lineal).
   Esto da la "estrategia esperada" contra la que juzgar a los agentes.
3. **(2 min) Metodología.** Simulador verificado ≡ entorno; semillas disjuntas train/val/test; ajuste en validación;
   5 semillas finales; 100 series de test idénticas para todos (pareado); Wilcoxon + IC bootstrap.
4. **(3 min) Resultados.** Tabla principal + curvas de aprendizaje + mapas de política.
5. **(2 min) Más allá de los mínimos.**
   - Mensaje 1: el techo con el estado base es una regla de histéresis, y la igualamos.
   - Mensaje 2: el cuello de botella es la observación. Con vol./MA, PPO supera a la regla en tres mercados.
   - Mensaje 3: con señal débil, los métodos on-policy colapsan a "siempre fuera". Se demuestra con Expected SARSA frente a Q-learning.
   - Mensaje 4: con una única regla de veredicto (gana / pierde / equivalente / no concluyente), DQN frente a
     Q-learning es "no concluyente" en ciclo y reversión, no un empate; hacen falta más semillas.
   - Resto en una frase: comisión, riesgo, agente único, parada temprana, ensemble.
6. **(0.5 min) Conclusiones y limitaciones.**

## 1bis. Cifras que hay que saberse (test, 100 series × 5 semillas; Σr = recompensa total media)

| Mercado | Q-learning | DQN | A3C | B&H | Mejor regla |
|---|---|---|---|---|---|
| Tendencia+ciclo | 7.989 | 7.902 | 7.638 | 6.014 | oráculo miope 7.773 |
| GBM | −0.043 | 0.013 | −0.006 | 0.014 | B&H |
| Reversión | 0.271 | 0.282 | 0.224 (0.278 con 3000 ep) | 0.105 | contraria 0.122 |
| Regímenes | 0.077 | 0.040 (0.059 con 1000 ep) | 0.000 ("siempre fuera"; 0.013 con 3000 ep) | −0.110 | siempre fuera 0.000 |

- **Ciclo**: los tres agentes baten a B&H en el 99% de las series (Wilcoxon p≈4e-18) y superan al
  oráculo miope porque aprenden histéresis (compran con x>0, venden por debajo de ≈−0.5σ): ~30–50 operaciones
  frente a 82 del momentum puro.
- **GBM**: DQN = B&H (1 operación; *equivalente* por TOST, IC90% [−0.005, +0.002] dentro de ±0.02). Es lo
  correcto: no hay señal, cualquier operación cuesta comisión.
- **Reversión**: DQN rinde +24.3% mediano frente a −0.4% de B&H y +8.6% de la contraria; opera 25 veces frente a 127.
- **Regímenes**: la señal (~0.3‰/paso) es menor que el ruido; Q-learning y DQN aprenden una banda (U invertida)
  y baten a B&H en ~80% de las series; A3C se queda en "siempre fuera" (óptimo local seguro).
- **DQN frente a Q-learning** (Σr, familia confirmatoria con Holm):
  - GBM: DQN gana (+0.056, IC [+0.002, +0.102]; Q-learning sobreopera);
  - regímenes: DQN pierde (−0.037, IC [−0.069, −0.006], p de Holm = 0.011);
  - ciclo (−0.086) y reversión (+0.011): **no concluyente**, no empate. En reversión el IC90% [−0.025, +0.045]
    se sale de ±δ = 0.02. El 61–83% de la incertidumbre viene de las semillas; el efecto mínimo detectable en
    reversión es 0.062 con 5 semillas y 0.038 con 20, aún mayor que δ: hacen falta más semillas, no más series.
  
  DQN converge en muchos menos episodios (106–236 frente a 265–1250), pero cada episodio cuesta ≈31 veces más
  (media de los 4 mercados; ≈22 en el ciclo). Tiempos **sin carga, en serie, los 4 mercados** (Tabla 4):
  DQN 159–203 s / 400 ep, Q-learning 24–47 s / 2000 ep, A3C 17–55 s / 1200 ep con 4 workers. En segundos
  hasta converger, A3C es el más rápido.
- **Determinismo**: DQN y Q-learning repiten exactamente el mismo checkpoint con y sin carga; A3C no
  (el orden de las actualizaciones asíncronas depende del planificador del SO).
- **Ablaciones**: sin red objetivo, DQN en reversión llega a buenos *checkpoints* pero termina en 0.107 de
  validación, frente a 0.228 con red objetivo y 0.273 con Double. γ=0 basta en ciclo y reversión (la señal
  predice el siguiente paso) y es el peor en regímenes (persistencia). γ=0.99 opera de más.
- **A3C con workers**: sin contar el arranque, de 42 s a 7.6 s (×5.5) en el ciclo con 16 workers, y de 49 s a
  12 s en reversión (satura a partir de 4–8). Arrancar 16 procesos con *spawn* en Windows cuesta ≈10 s.
- **Reproducibilidad**: dos ejecuciones independientes con la misma configuración y semilla
  (`gamma_0.5_ciclo` y `dqnvar_double_ciclo`) dan resultados idénticos.
- **S&P 500 2011 (fuera de muestra)**: DQN +12.8% ± 7.5 (3 semillas) frente a −1.1% de B&H; es una sola serie,
  así que el resultado es indicativo. Contraste con 20 000 estrategias aleatorias de igual exposición y nº de
  operaciones: DQN p = 0.038 / 0.145 / 0.014 por semilla; Q-learning y A3C no se distinguen del azar.
  Con Benjamini-Hochberg sobre toda la familia exploratoria solo 1 de las 9 ejecuciones (DQN) sigue siendo
  significativa: indicio, no evidencia.

## 1ter. Ampliación (§9 del notebook): cifras clave

- **Regla óptima** (búsqueda exhaustiva de 1–4 umbrales en 100 series de *entrenamiento*), Σr en test:
  - ciclo: histéresis momentum (compra z > 0.25σ, vende z < −0.75σ) 8.120;
  - reversión: histéresis contraria (compra z < −1.25σ, vende z > 1.5σ) 0.277;
  - regímenes: banda con histéresis 0.016 (sobreajusta: 0.069 en el ajuste);
  - GBM: ≈ B&H.
  
  Frente a ella:
  - en ciclo DQN pierde (−0.22 en Σr; IC jerárquico [−0.36, −0.08]); Q-learning (−0.13) es no concluyente;
  - PPO es *equivalente* a la regla por TOST (−0.041; con 4800 episodios +0.011, también equivalente);
  - en reversión y GBM, no concluyente para DQN, Q-learning y PPO; A3C pierde en reversión (−0.053);
  - en regímenes Q-learning gana (+0.061 en Σr, IC [+0.020, +0.103]).
- **PPO** (GAE + *clip*, 16 entornos vectorizados; 5 semillas, 2400 episodios):
  - ciclo: 8.079 ± 0.027, mejor que DQN (+0.18, IC jerárquico [+0.04, +0.32]); con 4800 episodios, 8.131 (= regla);
  - reversión: 0.289 (= DQN);
  - regímenes: "siempre fuera".
  
  Es el agente más estable entre semillas y entrena en ≈1 min (DQN ≈9 min).
- **Regímenes y gradiente de política**: A3C y PPO acaban en "siempre fuera".
  - Con entropía 0, la política estocástica pasa de estar invertida el 40% del tiempo a estarlo el 0.003%: colapso de exploración.
  - Con entropía 0.2 sigue explorando (47%), pero la política greedy tampoco aprende la banda (Σr −0.010).
  - Por tanto el colapso es un síntoma; la causa es la baja relación señal/ruido de la ventaja (0.3‰ de señal frente a 1–2.5% de ruido por paso).
  - Q-learning promedia muchas muestras por estado y sí la encuentra.
  - **Evidencia a favor** (no aísla todas las diferencias con A3C/PPO, que además usan red y gradiente de política): misma tabla y mismos hiperparámetros, cambiando solo el objetivo de Q-learning (max) a Expected SARSA (esperanza bajo ε-greedy, on-policy):
    - regímenes: 0.078 → 0.001 (IC [−0.12, −0.04]);
    - ciclo: la misma política.
    - Si preguntan por ε = 0.3 (0.041 frente a 0.059, n.s.): SARSA sigue cayendo en "siempre fuera" mucho más que Q-learning (29–39 de 80 evaluaciones a 0, frente a 9–11), pero la exploración persistente le saca a ratos y la parada temprana elige uno de esos *checkpoints*. Con ε final 0.02, 67–70 de 80 evaluaciones dan 0.
  - Es el *cliff walking* de Sutton y Barto: on-policy valora comprar bajo su propia exploración, que paga comisiones al deshacer la posición.
- **Estado enriquecido** (vol. 20 días y P/MA30 − 1; ventana de 10 log-rendimientos; causales, test de causalidad):
  - **PPO + ventana + vol./MA es el mejor agente del trabajo**: ciclo 8.690, reversión 0.448, regímenes 0.141, GBM = B&H.
  - Frente a la mejor regla: +0.57, +0.17 y +0.13, los tres *gana* (p ajustada < 0.001).
  - Con vol./MA, PPO **deja de colapsar** en regímenes (0.118): la volatilidad delata el régimen bajista (σ 2.5% frente a 1%).
  - Aquí la red sí aventaja a la tabla, porque con 4–14 dimensiones discretizar es inviable.
  - DQN con la ventana de 10 rendimientos sobreopera (≈100 operaciones), incluso con el triple de episodios: limitación reconocida.
- **Agente único para 4 mercados**:
  - con el estado base es peor en todo (POMDP: la misma observación exige acciones opuestas);
  - con vol./MA iguala o mejora al especializado en reversión (0.401) y GBM, pero no en ciclo ni regímenes.
- **Comisión** (DQN reentrenado con 0 / 0.05 / 0.1 / 0.5%, 3 semillas):
  - reversión: de 57 a 7 operaciones; zona de no operar de 2.0σ a 4.2σ;
  - ciclo: de 58 a 39 operaciones; zona de 0.38σ a 1.0σ, igual que la regla óptima (38 operaciones, Σr indistinguible);
  - regímenes con 0.5%: DQN aprende "siempre fuera" y bate a la regla re-ajustada, que sobreajusta (+0.10, IC [+0.04, +0.16]);
  - **predicción refutada**: sin comisión la histéresis no desaparece (DQN 2.0σ, regla 2.75σ en reversión). La observación no es markoviana y la posición actúa como memoria.
- **Riesgo** (recompensa r − λr² o log; 3 semillas; misma evaluación):
  - regímenes con λ = 2: filtro de régimen. La penalización casi duplica la pérdida esperada del régimen bajista (−0.15% → −0.28% por paso) y apenas toca el alcista (+0.10% → +0.08%). Exposición del 63% al 44%, *drawdown* del 28% al 20%, Σr 0.039 → 0.088, log-riqueza +0.065 (IC [+0.01, +0.12]);
  - ciclo con λ = 10: sale del mercado (*drawdown* 71% → 11%, Σr 7.89 → 0.41), porque con ≈10% de volatilidad por paso la penalización supera la señal;
  - log-riqueza como recompensa, log(B_t/B_{t−1}) calculada del balance: nada significativo (≈ ρ − ρ²/2 salvo al
    operar, donde la comisión es multiplicativa: 0.999(1+u) frente a 1+u−0.001; equivale a λ ≈ 0.5);
  - reversión y GBM: no cambian. Lección: λ debería fijarse como λσ² por mercado.
- **Último paso terminal (sin bootstrap)**, DQN, 3 semillas: equivalente en el ciclo, no concluyente en el resto.
- **Dueling DQN** (5 semillas): sin diferencias demostradas con DQN (equivalente en el ciclo, no concluyente en
  los demás). Con 3 acciones, separar V y A apenas aporta.

## 1quater. Evaluación estadística global y balance (§3–§4 de la memoria)

- **Regla única** para los 252 contrastes: Wilcoxon pareado + IC95% *bootstrap* jerárquico (semillas × series).
  Gana/pierde si p ajustada < 0.05 y el IC excluye el 0; equivalente si el IC90% cae en ±δ (TOST);
  si no, no concluyente. δ = 0.02 de Σr (0.12 en el ciclo, 2% de B&H).
- **Familias**: confirmatoria = mínimos (20 contrastes, Holm); exploratoria = el resto (232, Benjamini-Hochberg).
  ¿Por qué? Los mínimos son las hipótesis fijadas de antemano por el enunciado; el resto es exploración, y ahí
  controlar la tasa de falsos descubrimientos es lo razonable.
- **Confirmatorio**: 14 de 20 ganan. DQN frente a B&H: cociente de capital ×14.6 (ciclo), ×1.26 (reversión),
  ×1.22 (regímenes). A3C frente a B&H en regímenes (+0.110) no resiste a Holm (p = 0.11).
- **Friedman** rechaza la igualdad de los 11 agentes en los 4 mercados (p ≤ 3e-30); diferencia crítica de
  Nemenyi 1.51 rangos.
- **Oráculo con información oculta** (idea del primer grupo; banda de reversión ajustada en entrenamiento):
  DQN captura el 37%, 32% y 18% del margen (ciclo, reversión, regímenes); PPO + ventana + vol./MA, 52%, 62% y 63%.
- **Parada temprana frente a modelo final** (ambos grupos): DQN +0.076 en reversión y +0.055 en regímenes.
- **Ensemble de 5 semillas** (segundo grupo): A3C en el ciclo +0.35 (anula la semilla que colapsó; no concluyente
  por esa misma semilla); DQN en regímenes −0.040: promedia bandas distintas y acaba "siempre fuera".
- **Comprobación automática**: `check_memoria.py` verifica que no hay cifras escritas a mano, que cada
  "equivalente" tiene TOST, que todo el inventario aparece y que el PDF tiene 2 páginas.

## 2. Preguntas probables y respuestas

### Sobre la ampliación

**¿Por qué no ganáis a una regla de dos umbrales en el ciclo?** Porque, con la observación (z, posición), la
política óptima entre las que solo dependen de la observación **es** una regla de histéresis de dos umbrales
cuando la ventaja de estar dentro es monótona en z. La búsqueda exhaustiva optimiza directamente la Σr dentro
de esa familia; el agente de RL tiene que descubrir la forma *y* los umbrales a partir de recompensas
ruidosas. Que Q-learning quede a 0.13 y PPO (4800 episodios) a +0.01 de ese techo dice que los agentes
capturan prácticamente todo lo alcanzable. La ventaja del RL aparece donde la regla no se puede ajustar bien
(regímenes: la regla de 4 umbrales sobreajusta) o cuando el estado crece (§9.2: con ventana y vol./MA, PPO
supera a la regla en ciclo, reversión y regímenes, los tres con p ajustada < 0.001).

**¿Por qué no usasteis PPO desde el principio?** El enunciado pide DQN y A3C. PPO es la ampliación natural de
A3C (mismo actor-crítico, datos síncronos, objetivo recortado que permite reutilizar cada lote varias épocas).
Lo hemos implementado y verificado: es el más estable entre semillas y el más rápido. Pero en regímenes
falla igual que A3C, lo que indica que el problema es del gradiente de política en sí con esta relación
señal/ruido, no de la implementación de A3C.

**¿Cómo sabéis que el colapso a "siempre fuera" no es un bug?** Por cuatro motivos:
- PPO y A3C aprenden bien en los otros tres mercados con el mismo código.
- En el ciclo, el mismo experimento de control mantiene la exposición en el 65%.
- PPO aprende la banda en regímenes en cuanto la observación incluye la volatilidad (0.118–0.141).
- El efecto se reproduce sin redes: Expected SARSA tabular (on-policy) también se queda en ≈0 donde Q-learning obtiene 0.078.

**¿Entonces qué recomendaríais para trading real?** Algo off-policy y basado en valor, o PPO con una
observación rica. Y siempre evaluado contra la mejor regla simple ajustada en entrenamiento: es la
referencia que de verdad hay que batir, no B&H.

### Sobre el entorno y el diseño

**¿Habéis modificado el entorno?** No. `rl_trading/env.py` contiene el código literal del notebook. Solo
añadimos generación reproducible (fijamos `np.random.seed(seed)` antes de llamar al generador original).
Lo que sí cambiamos es la *codificación* de la observación para la red: dividimos la tendencia por su
desviación típica (estimada solo con series de entrenamiento). Esto no cambia el MDP: es una
transformación biyectiva de la observación.

**¿Por qué normalizar la tendencia?** En GBM/reversión/regímenes la tendencia tiene σ≈0.024, frente a la
posición ∈{0,1}. Con entradas de escalas tan distintas la red tarda mucho en distinguir estados (los
gradientes respecto a la tendencia son ~40 veces menores). En el ciclo σ≈0.15. Normalizar por generador
hace que el mismo hiperparámetro de lr funcione en todos.

**¿Por qué escaláis la recompensa ×100?** Los rendimientos por paso son ~1e-2–1e-3; los Q serían ~1e-2 y el
error de Bellman minúsculo frente a la inicialización. Multiplicar por una constante positiva no cambia la
política óptima (Q* escala igual) y deja los valores en O(1). Es la recomendación del profesor
("escalar los rewards").

**¿Cómo tratáis el final del episodio?** El entorno devuelve `terminated=True`, pero nosotros decidimos
tratarlo como límite de tiempo (500 precios): la serie no acaba por nada que el agente observe y el estado no
contiene t, así que el mismo estado puede ser "terminal" o no. Hacemos *bootstrap* también en la última
transición (Pardo et al., 2018, *Time Limits in RL*; es la distinción terminated/truncated de Gymnasium).
Reconocemos que es una decisión sobre el objetivo, no una reproducción literal, y la contrastamos: DQN sin
*bootstrap* en el último paso (3 semillas por mercado) es equivalente en el ciclo y no concluyente en el resto,
sin ninguna diferencia significativa. Es lo esperable: afecta a 1 de 494 transiciones y con γ = 0.5 el término
de *bootstrap* pesa poco.

**¿Cómo se agregan las semillas en Wilcoxon?** Se promedian las semillas en cada serie y Wilcoxon se aplica a las
100 diferencias pareadas (no se apilan semilla × serie, que serían dependientes). Esa p habla de *estos*
modelos entrenados. La incertidumbre entre entrenamientos la recoge el IC *bootstrap* jerárquico (remuestrea
semillas y series), y la regla exige las dos cosas: p ajustada < 0.05 **y** IC que excluya el 0.

**¿Por qué la posición es imprescindible en el estado?** Por la comisión: el valor de estar dentro depende
de si ya lo estás (cambiar cuesta 0.1%). La política óptima tiene una banda de histéresis: compra con más
señal de la que necesita para mantener. Se ve en los mapas de política (curvas de "compra" y "mantiene"
distintas).

**¿La recompensa mide el dinero ganado?** No exactamente. Σr suma rendimientos *aritméticos*; la riqueza es
multiplicativa. log W_T ≈ Σr − ½Σρ² − n·c. Por eso reportamos Σr (lo que optimiza el agente) y log-riqueza
(lo que le importa al inversor). Es justo la advertencia del profesor: "G sube pero el PnL no".

### Sobre los algoritmos

**¿Por qué replay en DQN?** Las transiciones consecutivas de una serie de precios están muy
correlacionadas (la tendencia de 6 precios cambia poco de un paso a otro). Entrenar con ellas en orden
viola el supuesto i.i.d. del SGD y provoca olvido. El replay mezcla experiencias de muchas series.

**¿Por qué red objetivo y Double DQN?** El target r+γ max Q(s') depende de la red que se está
entrenando: "target dinámico" → oscilaciones. La red objetivo lo congela C pasos. Double DQN evita la
sobreestimación de max sobre estimaciones ruidosas: aquí la recompensa es muy ruidosa (σ del 1–10% por paso)
y max Q sesga hacia arriba la acción "estar dentro". Lo verificamos con la ablación.

**¿Qué es exactamente vuestro A3C?** Actor-crítico con red de dos cabezas (actor softmax, crítico lineal)
y tronco común, como el notebook del profesor, pero con: (1) retornos a n pasos con bootstrap del crítico
(ventaja A = R⁽ⁿ⁾ − V), (2) bonus de entropía, (3) N procesos worker que actualizan de forma asíncrona
los pesos globales en memoria compartida (Hogwild) con un Adam de momentos compartidos. Es A3C real, no A2C.

**¿Por qué A3C no necesita replay?** Porque es on-policy (el gradiente de política requiere datos de la
política actual) y la decorrelación la aportan los workers en paralelo, cada uno en una serie distinta.

**¿Por qué evaluáis A3C en modo greedy?** Para comparar políticas deterministas con DQN y Q-learning y
eliminar el ruido de muestreo. La política estocástica se usa solo para explorar durante el entrenamiento.

**¿Cómo exploráis en cada uno?** DQN: ε-greedy con ε_k = max(0.02, λ^k), λ ajustado. A3C: muestreo de π
más entropía β. Q-learning: ε-greedy y α decreciente.

**¿Por qué el profesor esperaba que DQN ≥ Q-learning?** Porque la red generaliza entre estados vecinos y no
pierde información al discretizar. Con un estado de 1 dimensión continua + 1 binaria, Q-learning con bins
equiprobables es muy competitivo: la discretización apenas pierde información. Veredicto empírico: DQN gana
en GBM, pierde en regímenes y es no concluyente en ciclo y reversión (no un empate: con 5 semillas no hay
potencia para demostrar equivalencia); ventaja clara de DQN en eficiencia muestral. Las
ventajas de la red aparecerían con un estado de mayor dimensión (varias ventanas, volatilidad...).

**¿Por qué A3C queda por debajo?** El gradiente de política es más ruidoso que el target de Q-learning y
aquí la recompensa tiene una relación señal/ruido bajísima. En reversión era falta de episodios (con 3000
episodios llega a 0.278, igual que DQN). En regímenes cae en "siempre fuera", que es un óptimo local:
gana a B&H en el 71% de las series pero no captura la banda. Una semilla del ciclo colapsó a B&H por baja
entropía (β=0.001).

**¿Por qué el agente no siempre maximiza Σr en test?** Porque se elige el *checkpoint* por validación
(parada temprana), no por test; y porque Σr mide rendimientos aritméticos. Además la política greedy final
es determinista, y el ruido en Q cerca de la frontera de decisión cambia operaciones concretas.

### Sobre la metodología

**¿Cómo evitáis sobreajustar a los datos de test?** Tres conjuntos de semillas disjuntos. Los
hiperparámetros y la parada temprana solo usan validación. El test (100 series) se mira una única vez con
la configuración ya elegida. Las semillas de la ejecución final (10–14) son distintas de las del ajuste (0–1).

**¿Por qué pareado?** Todos los agentes se evalúan en las mismas 100 series: la diferencia agente−B&H
elimina la varianza común de la serie (common random numbers). Wilcoxon no asume normalidad (la riqueza
tiene colas gruesas).

**¿Cómo medís la convergencia?** Primer episodio en el que la curva de validación (media móvil de 3
evaluaciones) alcanza el 90% de su mejora total; también en segundos reales, que es lo justo para comparar
A3C (paralelo) con DQN (serie).

**¿Por qué 5 semillas?** RL profundo tiene alta varianza entre ejecuciones (Henderson et al., 2018);
reportamos media ± sd entre semillas. El profesor: "no fiarse de una sola tirada".

## 3. Limitaciones que conviene reconocer antes de que las pregunten

- Datos sintéticos estacionarios: en mercados reales la estructura cambia (no estacionariedad).
- El estado (tendencia de 6 precios + posición) es muy pobre: limita lo que cualquier agente puede aprender
  (p. ej. en regímenes la señal es débil frente a la comisión). §9.2 lo demuestra y lo corrige en parte.
- La recompensa aritmética no penaliza el riesgo. En §9.5 lo probamos (r − λr²), pero λ no es comparable entre
  mercados porque depende de la escala de los rendimientos; habría que fijar λσ².
- Comisión fija y ejecución al precio anterior sin deslizamiento: optimista frente a la realidad.
- Búsqueda de hiperparámetros limitada por cómputo (12/10 configuraciones). Las variantes de la ampliación
  (variables nuevas, comisión, riesgo) heredan los hiperparámetros del estado base y usan 3 semillas.
- El test se usa para muchas comparaciones (252). Nunca para elegir nada, y el error se controla sobre todo el
  trabajo (Holm en los mínimos, Benjamini-Hochberg en el resto). Pero muchas variantes usan 2–5 semillas:
  potencia baja para efectos pequeños, por eso hay tantos "no concluyente".
