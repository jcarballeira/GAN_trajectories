# CLAUDE.md — cGAN de trayectorias articulares (NIMBLE)

## Contexto

Repositorio del TFG de Manuel Guerrero Sancho (tutor: Juan Carballeira López, URJC, grupo BeST).
Modelo: WGAN-GP condicionada que genera un ciclo de marcha (T = 100 muestras) de 4 grados de libertad
(Hip_flex, Knee_flex, Ankle_flex, Pelvis_List) a partir de:

- condiciones clínicas `Xcond = [L, H]` (longitud y altura de paso, en metros),
- antropometría `Smeas` (altura, fémur, tibia, longitud de pierna, ancho de pelvis, ancho de rodilla),
- ruido latente AR(1).

Componentes: generador (MLP de condiciones → 2×GRU → cabezas por articulación), crítico (Bi-GRU + MLP,
con feature matching), predictor auxiliar P (Conv1D + MLP → L̂, Ĥ). Evaluación cinemática con el modelo
directo de NIMBLE en MATLAB (`executeKinematicModel`, `gaitFeatureExtraction`), reconstruyendo la pierna
contralateral con un desfase del 50 %.

## Objetivo actual

Convertir el TFG en un **artículo de revista de nivel medio** (no Q1) con **dedicación mínima**.
El artículo es un estudio de viabilidad: el modelo genera, en milisegundos, trayectorias periódicas y
diversas condicionadas por metas clínicas, validadas estadística y cinemáticamente en **sujetos no vistos**.

**Fuera de alcance** (no implementar salvo indicación expresa):
FK diferenciable propia (solo se reutiliza la existente, ver T1.5), baseline cVAE/difusión,
experimentos de aumento de datos, datos de pacientes, comparación con IWO o con el MLP de NIMBLE
(trabajo no publicado).

**Opción en reserva (no es la dirección actual):** artículo comparativo cGAN vs DiT (Benasco et al.,
ICNR 2026) con el mismo split y métricas. Requiere coordinación con los coautores del CSIC; no empezar
sin confirmación de Juan. Mientras tanto, mantener el código compatible (mismo formato de splits y
métricas) para que la comparación sea barata si se decide.

## Trabajos relacionados y posicionamiento

- **Benasco et al., ICNR 2026** (CSIC + BeST): difusión (MDM y DiT con AdaLN-Zero y classifier-free
  guidance) sobre 4.590 ciclos de 22 sujetos, 21 ángulos bilaterales, condicionado por **dos clases** de
  longitud de paso. Incluye pérdidas de periodicidad y de trayectoria del pie por FK. Su trabajo futuro
  declarado es condicionamiento continuo por velocidad y antropometría, que es lo que aporta este modelo.
- **Trombin et al., IEEE TSMC 2026** (review de environment-adaptive gait planning): taxonomía
  analíticos vs imitativos. Los modelos generativos profundos (GAN, difusión) no aparecen; este trabajo
  encaja en "imitative methods → other imitative methods" y cubre limitaciones señaladas por la review
  (ciclo completo con doble apoyo, plano frontal vía Pelvis_List, coste computacional).
- Mensaje diferencial del artículo: condicionamiento **continuo** por metas clínicas (L, H) y
  antropometría, en **una sola pasada**, como interfaz entre percepción del entorno y generación de marcha.

## Reglas de trabajo

- No modificar nunca los datos originales (.mat de NIMBLE). Todo derivado va a `data/processed/`.
- Toda partición train/val/test se hace **por sujeto** (`subject_id`). Nunca por ciclo.
- Los splits se guardan en fichero (`data/splits/*.json`) y se reutilizan en todos los experimentos.
- Cada experimento tiene un config (`configs/<exp>.yaml`) y escribe en `results/<exp>/`:
  `config.yaml`, `checkpoint_best.pt`, `metrics.json`, `figures/`.
- Semillas fijadas (torch, numpy, random). Las comparaciones entre variantes usan ≥ 3 semillas.
- Las métricas se calculan con scripts, nunca a mano. Una tabla o figura del artículo debe poder
  regenerarse con un único comando.
- Informar siempre de la tasa de muestras descartadas; no ocultar resultados sobre el conjunto filtrado.
- Antes de cambiar la arquitectura, confirmar que el modelo base reproduce los números de la memoria.

## Plan de tareas

### Fase 0 — Comprobación (semana 1). Decide si sigue el artículo.

- [ ] **T0.0 Corregir el pipeline (antes de reentrenar, 1–2 días).** Fallos detectados al revisar el código:
      1. `sample_cgan.py`: normalizar L, H y S con las estadísticas del checkpoint (`m_LH`, `s_LH`,
         `m_S`, `s_S`); ahora entra L/H en metros y S con las estadísticas del fichero muestreado.
      2. `LH_from_VICON_angles.m` lee siempre la pierna izquierda (`*_L`, `LHEE`), también en
         `Normalized_R`, así que cada ciclo aparece dos veces y la copia "izquierda" lleva Pelvis_List
         invertida. Usar la pierna derecha real (`*_R`, `RHEE`) en `Normalized_R` e invertir Pelvis_List
         en los ciclos derechos (la base del modelo cinemático es `hipL`), tras verificarlo con un gráfico
         de Pelvis_List por pierna.
      3. Guardar `subject_id` y nivel de velocidad en la BD; no saltar el primer sujeto (`i = 2`).
      4. Suelo de error: pasar los ciclos reales por `LH_from_Y_single` y comparar con `X_cond`
         (etiquetas y evaluación usan parámetros de pie distintos). Guardar en `results/fk_floor.json`.
      5. `eval_clinica.py`: métricas principales sin filtro de Tukey; informar del % descartado en cada filtro.
- [ ] **T0.1 Inventario de datos.** Tabla con nº de sujetos, ciclos por sujeto, por pierna, por nivel de
      velocidad, longitud y altura. Guardar en `results/data_inventory.csv`.
- [ ] **T0.2 Partición por sujetos.** Hold-out de sujetos de test (≈ 20 %) + validación por sujetos para
      early stopping. Si hay pocos sujetos, k-fold agrupado (`GroupKFold`). Guardar splits.
- [ ] **T0.3 Reentrenar el modelo actual sin cambios** con el nuevo split y el pipeline de T0.0, y recalcular las métricas de la
      memoria (RMSE de curvas medias, error L/H por FK, % descartado por periodicidad).
- [ ] **T0.4 Baseline determinista.** Mismo generador sin pérdida adversarial ni feature matching
      (z = 0, entrenado con MSE + pérdidas biomecánicas + consistencia).
- [ ] **Punto de decisión** (reunión con Juan): si el error de L en test supera ~10 cm de mediana, o si la
      cGAN no muestra diversidad frente al baseline, replantear antes de seguir.

### Fase 1 — Mejoras de bajo coste (semanas 2–3)

- [ ] **T1.1 Salida periódica por construcción.** Las cabezas predicen K armónicos (a0, ak, bk; K = 5–8)
      por articulación y se reconstruye la señal sobre T = 100. El cierre de ciclo queda garantizado;
      eliminar la pérdida de cierre y el filtro de similitud ≥ 95 %. Opcional: padding circular en el crítico.
- [ ] **T1.2 Velocidad como condición.** Añadir velocidad o cadencia a `Xcond` (normalizada).
- [ ] **T1.3 Criterio de early stopping.** Sustituir el MSE frente a trayectorias reales (favorece el
      colapso a la media) por error de L/H estimado + MMD en validación.
- [ ] **T1.4 Revisar el desfase contralateral.** Verificar el tratamiento de signo de Pelvis_List al
      desplazar medio ciclo y documentarlo (la corrección del signo por pierna se hace en T0.0).
- [ ] **T1.5 Reutilizar el término FK del DiT (condicional).** Si el código del grupo CSIC está disponible,
      integrar su pérdida de trayectoria del pie por cinemática directa (con curriculum) en lugar de, o
      junto a, el predictor P. Es la mejora con más probabilidad de reducir el error en L. Si el código no
      está disponible o no es compatible, no implementar una FK propia: dejarlo como trabajo futuro.

### Fase 2 — Evaluación (semanas 3–4), siempre en el test de sujetos no vistos

- [ ] **T2.1 Error por muestra.** RMSE por articulación entre cada ciclo generado y su ciclo real emparejado
      (no solo entre curvas medias). Reportar mediana e IQR, también por sujeto.
- [ ] **T2.2 Diversidad.** M = 20 muestras por condición con distinto z. Comparar la dispersión
      intra-condición con la dispersión real de ciclos con condiciones similares (vecinos en L, H, S).
- [ ] **T2.3 Distribución.** MMD (kernel RBF) y precision/recall o coverage k-NN en espacio PCA.
      Mantener las curvas medias, matrices de acoplamiento y PCA de la memoria como apoyo.
- [ ] **T2.4 Control clínico por FK.** Error |L|, |H| (p50, p75, p95) por tramos de condición; % de ciclos
      con valores no fisiológicos; % con penetración del suelo (Z < 0).
- [ ] **T2.5 Tiempo de inferencia.** Batch 1, CPU y GPU, media ± SD sobre 1000 llamadas.
- [ ] **T2.6 Ablaciones.** Variantes: completo / sin FM / sin consistencia / sin biomecánicas /
      sin adversarial (= T0.4). Mismo split, ≥ 3 semillas.
- [ ] **T2.7 Estadística.** Media ± SD entre semillas; Wilcoxon pareado entre variantes; SPM1d opcional
      para curvas.

### Fase 3 — Entregables

- [ ] Script `make_paper_figures.py` que genere todas las tablas y figuras desde `results/`.
- [ ] Tabla resumen de hiperparámetros (capas, tamaños, α de cada pérdida, λGP, lr, batch, épocas,
      n_critic) para la sección de métodos.
- [ ] README con instrucciones de reproducción.

## Comandos

Completar cuando estén definidos (entrenamiento, muestreo, evaluación Python, evaluación FK en MATLAB).
