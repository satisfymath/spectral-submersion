# Rebuttal preventivo — las 10 objeciones más probables (ML4AL @ ACL)

**R1. "Esto es numerología sobre un corpus minúsculo; Sproat ya mostró que
estas estadísticas no prueban lengua."**
De acuerdo — y esa es la tesis del paper, no su refutación. La Sección 3
demuestra (T1/P7) que ninguna estadística interna decide un claim semántico;
por eso todo resultado se etiqueta C1 (falsable por controles) o C2
(rankeable). La subsección de related work engancha el debate punto por
punto y el paper reporta sus propios sanity checks fallidos (F3: el espectro
real NO se separa de controles freq-matched). No afirmamos lengua; afirmamos
qué es decidible y construimos dentro de ese límite. (Secs. 2–3, Fig. F3.)

**R2. "El corpus paralelo es inventado; el traductor aprende lo que ustedes
pusieron."**
Correcto, y está declarado como necesidad lógica: por T1 no puede existir un
corpus paralelo verificado construido desde evidencia interna. La hipótesis
categoría→serie está etiquetada C2, visualizada como tal (F8, título
"WORKING HYPOTHESIS"), y la ablación noclass mide exactamente cuánto aporta.
El traductor se evalúa contra estadísticas del corpus real, lo único no
inventado disponible. (Secs. 5, App. C.)

**R3. "7.31 < 8.26: ¿su generador es 'más real' que las tablillas? Absurdo."**
Es absurdo, y el paper lo dice antes que el revisor: P6 demuestra que un
decodificador que maximiza un score con término LM produce bits/glifo por
debajo de la media real POR CONSTRUCCIÓN (sesgo de selección modal). El sweep
de λ lo verifica empíricamente (monótono, 8.03→5.37). La lectura correcta es
proximidad a la banda, no minimización. (Prop. P6, Fig. F5, caption de Tab. 1.)

**R4. "¿Dónde están los baselines?"**
Tabla 1: template sin aprendizaje, léxico+bigramas por conteo, LSTM pequeño,
más el sistema previo v5. El baseline de bigramas es además un experimento
sobre las métricas: logra BA=1.00 mientras colapsa (D2=0.03, JS=0.79),
demostrando que ninguna métrica aislada es optimizable con seguridad — por
eso la batería de seis. (Sec. 6, Tab. 1.)

**R5. "El held-out D,F puede estar contaminado por paralelos textuales."**
Auditado: 0 n-gramas compartidos con n≥5 entre {D,F} y {A,B,C,E} (tabla en
el repo). Los paralelos conocidos (H/P/Q, Gv/K) están fuera del corpus usado.
(Sec. 4, `audit_heldout_contamination.py`.)

**R6. "El sesgo posicional del Indus es descriptivo; ¿significancia?"**
Tests de permutación within-inscription (10k permutaciones) con corrección
Benjamini–Hochberg: P324/P086/P217/P000 iniciales y P385/P378 finales a
FDR≤0.05. Réplica del método sobre un benchmark sintético con sesgo diseñado
detecta las clases correctas sin falsos positivos. (Sec. 6.)

**R7. "¿Por qué λ=0.35 y no otro valor?"**
Barrido completo λ∈[0,1] paso 0.05 con CIs bootstrap (F5): 0.35 está en la
meseta que queda dentro de la banda de realismo y lejos de la zona de colapso
modal (λ≥0.85, donde el output converge a las estadísticas degeneradas del
baseline de conteo). La elección precede al sweep (dev) y el sweep la
respalda. (Fig. F5.)

**R8. "Snyder y Luo–Barzilay ya hicieron desciframiento neural; ¿qué hay de
nuevo?"**
Ellos disponen de lengua emparentada conocida — exactamente el anclaje K
cuyo precio cuantifica T2. Sin K, su objetivo sería G-invariante y sus
salidas C2. La contribución aquí no es competir en Ugarítico sino formalizar
qué separa ese régimen del régimen sin anclajes, y construir el mejor sistema
posible en el segundo. (Sec. 2, related work.)

**R9. "El mapa de embeddings (F1) no muestra clusters limpios por serie
Barthel; ¿no debilita el paper?"**
Lo reportamos exactamente así en el caption ("partial, not clean") y lo
contamos como evidencia del régimen T3 de corpus corto — el mismo teorema
que predice la estabilidad bootstrap en el benchmark sintético predice la
no-separación aquí. Un paper que solo reporta las figuras que salen bonitas
es el blanco de Sproat; este reporta ambas. (Figs. F1, F3; Sec. 6.)

**R10. "¿Ética? Están generando pseudo-Rongorongo de patrimonio vivo."**
Sección de ética explícita: sin claim de desciframiento ni lecturas (P7 lo
prohíbe formalmente); procedencia y checksums de la transcripción;
la demo se declara uso creativo personal de un generador estadístico, no
reconstrucción cultural; sin contacto comunitario aún — declarado como
trabajo futuro pendiente, no como logro. Si el venue exige más (p. ej.
retirar la demo), la estructura del paper lo permite sin tocar resultados.
(Sec. 9.)
