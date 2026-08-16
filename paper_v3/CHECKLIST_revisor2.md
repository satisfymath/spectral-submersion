# Checklist final del Revisor 2 — autoevaluación (Tarea 6)

| # | Pregunta | Sí/No | Evidencia / pendiente |
|---|----------|-------|------------------------|
| 1 | ¿Cada claim tiene nivel C1/C2/C3 asignado y visible? | **Sí** | Marcas \claim{} en resultados; captions con nivel; F10 con carriles |
| 2 | ¿Algún lugar sugiere "traducción" sin comillas ni nivel? | **Sí (limpio)** | Grep de "translat" revisado: siempre "translator/system" con estatus C2 declarado en Secs. 5–7 |
| 3 | ¿Baselines y ablaciones con las mismas métricas? | **Sí** | Tab. 1 (3 baselines + v5 + 3 ablaciones decode); App. C ablaciones reentrenadas (noclass/noaug) — **parcial hasta que terminen los runs** |
| 4 | ¿Todos los números con IC o test? | **Mayormente** | VR/BA/b g con CI bootstrap; JS/D2 punto (CI sesgada por pooling — declarado en caption); posicional con q-values BH; sweep con CIs. Métricas de corpus (5460 tokens etc.) son censos, no estimaciones |
| 5 | ¿7.31 < 8.26 explicado por P6 en el texto? | **Sí** | Prop. P6 en Sec. 3, guard en caption de Tab. 1, F5 y F6 |
| 6 | ¿Contaminación del held-out auditada y reportada? | **Sí** | 0 n-gramas n≥5 compartidos; Sec. 4 + JSON en repo |
| 7 | ¿Debate Sproat citado y respondido punto por punto? | **Sí** | Subsección propia en related work; F3 negativo como respuesta operativa |
| 8 | ¿Snyder/Luo–Barzilay citados y diferenciados? | **Sí** | Related work: su anclaje = K de T2; sin K, techo C2 |
| 9 | ¿Figuras con anti-conclusiones en caption? | **Sí** | F1–F10, todas |
| 10 | ¿Sección de ética presente? ¿Licencia declarada? | **Sí / parcial** | Sec. 9; licencia exacta de la transcripción phspaelti **[VERIFICAR antes de envío]** — compromiso de retiro si hay restricción |
| 11 | ¿Reproducibilidad: seeds, SHA-256, DOI, scripts? | **Sí / DOI pendiente** | Seed 42 global; manifiestos SHA-256 en repo; scripts nombrados en Sec. 10; DOI Zenodo [CALCULAR al aceptar] |
| 12 | ¿La dedicatoria salió de Results? | **Sí** | Vive en Acknowledgments (una línea); demo en Sec. 7 como case study con métricas |
| 13 | ¿El abstract promete exactamente lo que el paper entrega? | **Sí** | Estructurado; cada frase mapea a una sección; sin la palabra "decipherment" como logro |
| 14 | ¿Alguna cifra no regenerable con un comando? | **No** | Todas salen de scripts con seed en el repo; pendientes marcados [CALCULAR]: N(n) órbita empírica, histograma de márgenes T4, DOI, ablaciones reentrenadas, jueces ciegos |

## Pendientes explícitos antes del envío real
1. Swap a `acl.sty` oficial + ajuste fino a 8 páginas de cuerpo.
2. Verificar TODAS las referencias contra fuente primaria (regla 4) — hoy
   todas llevan `[VERIFY primary source]` en el .bib.
3. Confirmar licencia de la transcripción RR (phspaelti).
4. ~~Correr `fig_margin_distribution.py` y `estimate_empirical_orbit.py`~~ ✅
   Ejecutados: T4 con resultado negativo (umbral vacuo, AUC 0.536, reportado);
   órbita: log N ≥ 3727 nats, piso Fano 0.9998, precio de K ≈ 3354 nats.
5. Completar App. C cuando terminen los reentrenamientos noclass/noaug.
6. Ejecutar el test de jueces ciegos con k≥3 humanos.
7. Mint del DOI Zenodo en camera-ready.
