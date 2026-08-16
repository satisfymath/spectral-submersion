# Diff conceptual v2 → v3 (con justificación de revisor)

**Venue:** ML4AL @ ACL · **Idioma:** inglés · **Formato:** ACL 2-col emulado
(swap a `acl.sty` oficial antes del envío).

## Cambios de fondo

| # | Cambio | Justificación de revisor |
|---|--------|--------------------------|
| 1 | **Reencuadre del título y la contribución #1**: de "traductor a Rongorongo" a "generación de hipótesis consciente de identificabilidad; el protocolo C1/C2/C3 como teorema (D7/P7)" | El Revisor 2 ataca cualquier olor a claim de desciframiento. La novedad defendible es epistemológica y está demostrada, no declarada. |
| 2 | **T1 formalizado como acción de grupo** (antes Teorema 2.2 informal) + corolario data-processing | Cierra el hueco "¿y si un post-procesamiento astuto rompe la simetría?" — no puede (Cor. 1.1). |
| 3 | **T2 Fano + órbita ε(n)-empírica + precio de K** | Convierte "no se puede" en "cuánto costaría": I(θ;K) ≥ (1−δ)log N(n) − log 2. Resultado citable. |
| 4 | **T3 Bernstein matricial + lema PPMI con clipping + Wedin** | El bootstrap (CV 0.65%) pasa de observación a corolario; **predice también los sanity checks fallidos** (régimen ε(n) ≳ δ_k). Un solo teorema explica éxitos y fracasos reportados. |
| 5 | **T4 Procrustes con anclajes ruidosos + condición de margen** | Programa teorema↔figura EJECUTADO con resultado negativo honesto: umbral suficiente vacuo (0% certificado vs 68.9% observado; AUC del margen 0.536). Reportado como gap teoría-práctica, no escondido. |
| 6 | **P6 sesgo de selección del fusion decoding** | Blinda 7.31 < 8.26 ANTES de que el revisor pregunte. Verificado empíricamente: sweep λ monótono (8.03→5.37). |
| 7 | **Baselines obligatorios**: template sin aprendizaje, léxico+bigram counts, LSTM pequeño | "Sin baselines el Transformer es invendible." El bigram baseline además demuestra que BA=1.00 es hackeable → justifica la batería de 6 métricas. |
| 8 | **Ablaciones**: λ=0, sin rep-penalty, beam=1 (+ noclass y noaug reentrenadas, apéndice) | Contribución por componente con CIs. |
| 9 | **Tests estadísticos formales**: permutación within-inscription + BH (FDR≤0.05) para sesgo posicional Indus | Antes: descriptivo. Ahora: P324/P086/P217/P000 inicial y P385/P378 final con q-values. |
| 10 | **Auditoría de contaminación del held-out**: 0 n-gramas compartidos (n≥5) entre {D,F} y {A,B,C,E} | Pregunta segura del revisor de NLP; respondida con tabla antes de que la haga. |
| 11 | **Resultado negativo promovido a primera línea**: el espectro PPMI del corpus RR real NO se separa de controles freq-matched (F3) | Es la respuesta operativa a Sproat: reportamos lo que falla con la misma prominencia. |
| 12 | **Related work completo** con subsección propia para el debate Rao–Sproat, y diferenciación explícita de Snyder/Luo–Barzilay (ellos tienen lengua emparentada = anclaje K; nosotros no → techo C2) | Cobertura mínima exigida; todas las citas marcadas [VERIFY]. |
| 13 | **Sección de Ética y patrimonio** | Rongorongo como patrimonio vivo rapanui; sin claim de desciframiento; procedencia y licencia de la transcripción; sin contacto comunitario aún — declarado como trabajo futuro, no como casilla marcada. |
| 14 | **Dedicatoria movida a Acknowledgments**; el §7 de v2 queda como case study técnico con métricas | "El Revisor 2 no debe encontrar una carta de amor en Results." La versión romántica vive en el repo (consolidado v2 español). |
| 15 | **Figuras rehechas** F1–F10: Okabe–Ito, PDF vectorial, captions con anti-conclusión, un script con seed por figura | Marca de la casa C1/C2/C3 en cada caption. |
| 16 | **Abstract estructurado** (Problema/Método/Resultados/Límites, ≤250 palabras) + 4 contribuciones numeradas | Formato del venue. |
| 17 | **Los 2 tipos no cubiertos identificados**: `_` (laguna) y `000!` (ilegible, 95 tokens) — exclusión por diseño documentada | Cabo suelto de v2 cerrado. |
| 18 | **Protocolo de jueces ciegos pre-registrado** (40 ítems, llave sellada, plan de análisis AUC/κ/CI) | Métrica de realismo complementaria; declarado pendiente de ejecución humana — sin resultado inventado. |

## Claims: qué NO cambió
Ningún claim subió de nivel. v3 *baja* dos claims respecto de v2:
- La organización del mapa de embeddings por series Barthel se reporta como
  **parcial** (F1 muestra organización, no clusters limpios).
- El espectro real RR se reporta como **no separado** de controles (F3),
  cosa que v2 no analizaba sobre el corpus real.
