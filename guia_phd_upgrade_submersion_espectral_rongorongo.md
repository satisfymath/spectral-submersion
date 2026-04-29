
# Guía PhD para elevar *Submersión Espectral de Lenguajes Perdidos* a investigación matemáticamente seria y auditable

**Proyecto:** Submersión Espectral / Rongorongo Hypothesis Engine  
**Objetivo científico:** pasar de un ensayo matemático-prometedor a un marco formal, falsable, reproducible y revisable para generar hipótesis de análisis estructural y traducción-candidata hacia/desde Rongorongo.  
**Tesis metodológica:** el objetivo científicamente defendible no es afirmar un “traductor Rongorongo” verificado, sino construir un **modelo generativo-auditable de hipótesis Rongorongo** con niveles explícitos de incertidumbre, controles negativos, límites de identificabilidad y pruebas de estabilidad.

---

## 0. Resumen ejecutivo brutal

Tu paper ya tiene una intuición fuerte: un sistema simbólico no descifrado puede inducir una geometría estadística latente, extraíble por co-ocurrencias, PMI/PPMI, SVD, grafos, transporte óptimo y modelos probabilísticos. La parte peligrosa es el salto desde **geometría estructural** hacia **semántica**. Ese salto no es gratis. Matemáticamente, sin anclajes externos, todo sistema simbólico es invariante bajo permutaciones de nombres de glifos. Por tanto, ningún algoritmo puede identificar una traducción absoluta solo desde estructura interna.

El upgrade serio consiste en convertir el paper en una teoría de:

\[
\textbf{identificabilidad parcial} + \textbf{estabilidad espectral} + \textbf{falsabilidad empírica} + \textbf{auditoría de claims}.
\]

La versión PhD debe defender algo así:

> Dado un corpus Rongorongo incierto, un conjunto de corpus candidatos polinesios/rituales/calendáricos y un conjunto explícito de anclajes externos, nuestro método produce una distribución posterior sobre hipótesis glíficas. La salida válida no es una traducción única, sino una familia de hipótesis rankeadas, con evidencia, contraevidencia, estabilidad, controles negativos y nivel de claim permitido.

La meta aplicada “traductor a Rongorongo” debe reformularse científicamente como:

\[
p(x_{1:T}\mid y_{1:M},K),
\]

donde \(y_{1:M}\) es una entrada Rapanui/polinesia/conceptual, \(x_{1:T}\) es una secuencia glífica Rongorongo candidata, y \(K\) es conocimiento externo documentado: paleografía, soporte físico, orientación, calendario, tradición oral, iconografía, lingüística comparada, paralelos textuales y restricciones arqueológicas.

---

# Parte I — Qué le falta matemáticamente al paper actual

## 1. El problema no está todavía cerrado como problema inverso mal planteado

Tu paper habla de pseudoinversa, SVD, Procrustes y submersión. Eso está bien. Pero falta declarar formalmente que el desciframiento es un **problema inverso mal puesto** en el sentido de Hadamard.

Un problema inverso es bien puesto si cumple:

1. Existe solución.
2. La solución es única.
3. La solución depende continuamente de los datos.

En Rongorongo, las tres condiciones fallan o son dudosas:

- **Existencia:** puede no existir una correspondencia léxica simple entre glifos y palabras modernas.
- **Unicidad:** múltiples traducciones pueden explicar las mismas estadísticas.
- **Estabilidad:** pequeñas variaciones de lectura paleográfica pueden alterar bigramas, grafos y embeddings.

Debes agregar una proposición inicial:

### Proposición 1.1 — El desciframiento no anclado es un problema inverso mal puesto

Sea \(X\) un corpus finito de signos no descifrados y sea \(\mathcal{Y}\) una familia de corpus candidatos. Definamos un operador de observación

\[
\mathcal{O}: \Theta \to \mathcal{D}_X,
\]

donde \(\Theta\) parametriza traducciones, segmentaciones, valores fonéticos, clases funcionales y modelos generativos, y \(\mathcal{D}_X\) representa los datos observables: secuencias de glifos, posiciones, frecuencias, co-ocurrencias y metadatos. El problema de desciframiento consiste en recuperar \(\theta\in\Theta\) desde \(\mathcal{O}(\theta)=X\).

En ausencia de anclajes externos, \(\mathcal{O}\) no es inyectivo. Por tanto, el problema no tiene solución única.

**Demostración.**  
Sea \(G=\mathrm{Sym}(V_X)\) el grupo de permutaciones del vocabulario glífico. Para cualquier \(\pi\in G\), se puede construir un parámetro transformado \(\theta_\pi\) que renombra todos los glifos de acuerdo con \(\pi\). Las observaciones puramente estructurales son invariantes por tal renombramiento: patrones de igualdad, longitudes, repeticiones, grados de co-ocurrencia y espectros de matrices permutadas se preservan. Entonces \(\mathcal{O}(\theta)=\mathcal{O}(\theta_\pi)\) para \(\theta\neq\theta_\pi\), salvo que \(\pi\) sea identidad. Así, \(\mathcal{O}\) no es inyectivo. \(\square\)

### Qué mejora esto en tu paper

Esto te protege de críticas. Ya no dices “tenemos un método para traducir”; dices:

> “Formalizamos por qué la traducción fuerte es no identificable sin anclajes, y proponemos un procedimiento regularizado para estimar clases de equivalencia e hipótesis parciales.”

Eso es mucho más serio.

---

## 2. Falta una teoría de niveles de claim

El paper necesita separar con precisión matemática lo que puede inferirse.

Define un espacio de claims:

\[
\mathcal{C}=\{C_0,C_1,C_2,C_3,C_4,C_5\}.
\]

| Nivel | Claim | Lo permitido | Lo prohibido |
|---|---|---|---|
| \(C_0\) | Paleográfico | “Este trazo/glifo existe con probabilidad \(p\)” | significado |
| \(C_1\) | Estructural | frecuencia, posición, repetición, co-ocurrencia | función semántica |
| \(C_2\) | Funcional | marcador inicial/final, clasificador, numeral, determinativo posible | traducción léxica |
| \(C_3\) | Semántico débil | compatible con dominio lunar/genealógico/ritual | lectura literal |
| \(C_4\) | Fonético parcial | compatible con sílaba/sonido bajo anclaje externo | desciframiento completo |
| \(C_5\) | Traducción fuerte | lectura verificable | permitido solo con evidencia independiente fuerte |

### Definición 2.1 — Claim admissible

Un claim \(c\in \mathcal{C}\) es admisible para una hipótesis \(h\) si existe evidencia \(E\), contraevidencia \(B\), conjunto de controles negativos \(N\) y nivel de anclaje \(A\), tales que

\[
\mathrm{Admissible}(h,c)
=
\mathbf{1}
\left[
S(h;E)-\mathbb{E}_{n\sim N}S(h;n)>\lambda_c\sigma_N
\right]
\mathbf{1}
\left[
\mathrm{AnchorLevel}(h)\geq a_c
\right]
\mathbf{1}
\left[
\mathrm{Stability}(h)\geq s_c
\right].
\]

Donde \(\lambda_c,a_c,s_c\) aumentan con el nivel del claim.

### Reglas sugeridas

- \(C_0\): requiere confianza paleográfica.
- \(C_1\): requiere diferencia estadística contra controles.
- \(C_2\): requiere patrón estructural robusto más plausibilidad tipológica.
- \(C_3\): requiere priors culturales explícitos y sensibilidad a priors.
- \(C_4\): requiere anclaje fonético o comparativo fuerte.
- \(C_5\): requiere bilingüe, anclaje externo independiente o validación convergente extraordinaria.

### Teorema 2.2 — Monotonía de admisibilidad

Si los umbrales satisfacen

\[
\lambda_0\leq\lambda_1\leq\cdots\leq\lambda_5,
\quad
a_0\leq a_1\leq\cdots\leq a_5,
\quad
s_0\leq s_1\leq\cdots\leq s_5,
\]

entonces:

\[
\mathrm{Admissible}(h,C_j)=1 \implies \mathrm{Admissible}(h,C_i)=1
\quad \text{para todo } i<j.
\]

**Demostración.**  
Si \(h\) satisface los umbrales de un nivel superior \(j\), satisface automáticamente todos los umbrales menores o iguales de los niveles inferiores. \(\square\)

### Por qué esto importa

Esto convierte tu paper en un sistema de inferencia responsable. Un reviewer puede discutir tus umbrales, pero ya no puede acusarte de mezclar estructura con traducción sin control.

---

# Parte II — Núcleo matemático que debes agregar

## 3. Teorema de no-desciframiento gratuito ampliado

Tu paper ya tiene una versión de no-free-decipherment. El upgrade PhD debe expresarlo como una afirmación de **identificabilidad bajo acción de grupos**.

### Definición 3.1 — Corpus simbólico y acción de permutación

Sea \(V_X=\{x_1,\dots,x_n\}\) un vocabulario glífico. Un corpus es una secuencia

\[
X=(s_1,\dots,s_T), \qquad s_t\in V_X.
\]

El grupo simétrico \(G=\mathrm{Sym}(V_X)\) actúa sobre \(X\) mediante

\[
(\pi X)_t = \pi(s_t).
\]

Un estadístico \(S\) es estructural si

\[
S(\pi X)=S(X)
\quad\forall \pi\in G.
\]

Ejemplos: longitudes, patrones de igualdad, distribuciones de grados, valores singulares de co-ocurrencia bajo conjugación, espectros de Laplacianos, entropía de transiciones si los estados se renombran.

### Teorema 3.2 — No identificabilidad absoluta bajo invariancia estructural

Sea \(\Theta\) un espacio de traductores candidatos \(\tau:V_X\to \mathcal{M}\), donde \(\mathcal{M}\) es un espacio de significados, palabras, morfemas, sílabas o clases funcionales. Supongamos que el likelihood usado por el modelo depende solo de un estadístico estructural \(S(X)\):

\[
p(X\mid \tau)=F(S(X),\tau).
\]

Si no existe término de anclaje que no sea invariante por \(G\), entonces para todo \(\tau\) y toda \(\pi\in G\) existe un traductor \(\tau_\pi=\tau\circ\pi^{-1}\) que induce evidencia observacional equivalente. En particular, la semántica absoluta no es identificable.

**Demostración.**  
Dado \(\pi\in G\), considérese el corpus renombrado \(\pi X\). Para cada posición \(t\),

\[
\tau_\pi((\pi X)_t)
=
(\tau\circ \pi^{-1})(\pi(s_t))
=
\tau(s_t).
\]

Luego la secuencia semántica inducida por \((\pi X,\tau_\pi)\) coincide exactamente con la inducida por \((X,\tau)\). Además, por estructuralidad,

\[
S(\pi X)=S(X).
\]

Si el likelihood solo observa \(S\), no puede distinguir \((X,\tau)\) de \((\pi X,\tau_\pi)\). Por tanto, hay al menos \(|G|\) representaciones equivalentes salvo estabilizadores. La traducción absoluta requiere romper esta simetría con información externa. \(\square\)

### Corolario 3.3 — Solo se identifican órbitas

El objeto identificable no es \(\tau\), sino su órbita:

\[
[\tau]=\{\tau\circ \pi^{-1}:\pi\in G\}.
\]

La inferencia sin anclajes solo puede recuperar propiedades invariantes de \([\tau]\).

### Consecuencia para Rongorongo

Un modelo puramente espectral puede decir:

- este glifo es central;
- este glifo aparece en posición terminal;
- estos signos forman comunidad;
- hay patrones paralelos;
- hay compresión/no compresión espectral.

No puede decir, por sí solo:

- este glifo significa “rey”;
- este glifo se lee “ra”;
- esta línea dice “te amo”;
- esta secuencia corresponde a una oración determinada.

---

## 4. Teoría de anclajes como ruptura de simetría

### Definición 4.1 — Grafo glífico ponderado

Construye un grafo

\[
G_X=(V_X,E,W)
\]

con matriz de pesos \(W\in \mathbb{R}^{n\times n}\), donde \(W_{ij}\) puede integrar:

\[
W_{ij}
=
\alpha_1\operatorname{PPMI}_{ij}
+
\alpha_2\operatorname{Trans}_{ij}
+
\alpha_3\operatorname{Parallel}_{ij}
+
\alpha_4\operatorname{PosSim}_{ij}
+
\alpha_5\operatorname{IconSim}_{ij}.
\]

El grupo de automorfismos estructurales es

\[
\mathrm{Aut}(G_X)
=
\{\pi\in \mathrm{Sym}(V_X): W_{\pi(i)\pi(j)}=W_{ij}\ \forall i,j\}.
\]

### Definición 4.2 — Anclaje

Un anclaje es un par

\[
a=(x,y,\rho,\mathcal{E}),
\]

donde \(x\in V_X\), \(y\) es un significado/categoría/unidad candidata, \(\rho\in[0,1]\) es confianza, y \(\mathcal{E}\) es evidencia: iconográfica, arqueológica, fonética, calendárica, paralela, paleográfica o comparativa.

Un conjunto de anclajes \(A\) induce el subgrupo

\[
\mathrm{Aut}(G_X;A)
=
\{\pi\in \mathrm{Aut}(G_X): \pi(x)=x\ \forall (x,y,\rho,\mathcal{E})\in A, \rho>\rho_0\}.
\]

### Definición 4.3 — Potencia de anclaje

\[
\mathrm{AnchorPower}(A)
=
1-
\frac{\log(|\mathrm{Aut}(G_X;A)|+1)}
{\log(|\mathrm{Aut}(G_X)|+1)}.
\]

- Si \(\mathrm{AnchorPower}(A)\approx 0\), los anclajes no rompen simetría.
- Si \(\mathrm{AnchorPower}(A)\approx 1\), los anclajes eliminan casi toda ambigüedad estructural.

### Teorema 4.4 — Identificabilidad parcial por anclajes

Si \(\mathrm{Aut}(G_X;A)=\{e\}\), entonces toda hipótesis estructural compatible con \(A\) es identificable hasta la identidad dentro del modelo relacional \(G_X\).

**Demostración.**  
Por el Teorema 3.2, las ambigüedades no ancladas corresponden a la órbita generada por \(\mathrm{Aut}(G_X)\). Al imponer \(A\), solo sobreviven automorfismos que preservan los glifos anclados. Si el subgrupo remanente es trivial, no existe permutación no trivial que preserve simultáneamente estructura y anclajes. Luego la clase de equivalencia se reduce a un representante único dentro del modelo. \(\square\)

### Nota crítica

Esto no prueba que el representante único sea históricamente verdadero. Solo prueba identificabilidad **relativa al modelo y a los anclajes**. Por eso necesitas controles negativos y validación externa.

---

## 5. Estabilidad espectral: lo que falta para que el SVD sea serio

En corpus pequeños, la SVD puede ser frágil. Tu paper necesita teoría de perturbación.

### Definición 5.1 — Matriz verdadera y estimador empírico

Sea \(M\) una matriz estructural poblacional:

\[
M=\mathbb{E}[\widehat M],
\]

donde \(\widehat M\) se estima desde un corpus finito. Por ejemplo, \(\widehat M\) puede ser PPMI suavizado, transición regularizada o Laplaciano normalizado.

El error de estimación es

\[
E=\widehat M-M.
\]

### Teorema 5.2 — Estabilidad de subespacios singulares

Sea \(M=U\Sigma V^\top\) y \(\widehat M=\widehat U\widehat\Sigma \widehat V^\top\). Sea \(U_k\) el subespacio singular izquierdo asociado a los primeros \(k\) valores singulares, y \(\widehat U_k\) el estimado. Si existe gap

\[
\delta_k=\sigma_k(M)-\sigma_{k+1}(M)>0
\]

y

\[
\|E\|_2<\delta_k,
\]

entonces

\[
\|\sin\Theta(\widehat U_k,U_k)\|_2
\leq
\frac{\|E\|_2}{\delta_k-\|E\|_2}.
\]

**Demostración.**  
Es una aplicación directa de la teoría de perturbación de subespacios tipo Davis–Kahan/Wedin. La matriz perturbada \(\widehat M=M+E\) desplaza los valores singulares a lo más \(\|E\|_2\) por desigualdad de Weyl. Si el gap no se cierra, el subespacio estimado queda separado del complemento, y la norma del seno de los ángulos principales queda acotada por la razón entre perturbación y gap residual. \(\square\)

### Corolario 5.3 — Regla de rechazo espectral

Si

\[
\widehat \delta_k \leq 2\widehat \varepsilon,
\]

donde \(\widehat \varepsilon\) es error bootstrap estimado, entonces el embedding de dimensión \(k\) debe marcarse como **inestable** y no puede sostener claims \(C_2\) o superiores.

### Métrica que debes reportar

\[
\mathrm{SpectralReliability}_k
=
\max\left\{0,1-\frac{\widehat\varepsilon}{\widehat\delta_k}\right\}.
\]

### Tabla obligatoria

| Corpus | matriz | \(k\) | \(\widehat\delta_k\) | \(\widehat\varepsilon\) bootstrap | reliability |
|---|---|---:|---:|---:|---:|
| Rongorongo-Barthel | PPMI | 8 | ... | ... | ... |
| Rongorongo-Horley | transición | 8 | ... | ... | ... |
| Mamari | positional tensor | ... | ... | ... | ... |
| corpus sintético | PPMI | ... | ... | ... | ... |

---

## 6. Concentración de co-ocurrencias: falta justificar cuándo hay datos suficientes

Tu paper usa co-ocurrencias, pero debe decir cuándo son estimables.

### Modelo

Supón una secuencia estacionaria \((S_t)\) sobre \(V_X\). Para una ventana \(h\), define

\[
\widehat C_{ij}
=
\sum_{t=1}^{T}
\sum_{\substack{u:1\le |u-t|\le h}}
\mathbf{1}\{S_t=i,S_u=j\}.
\]

Sea

\[
p_{ij}^{(h)}
=
\mathbb{P}(S_t=i, S_{t+d}=j \text{ para } 1\le |d|\le h).
\]

Entonces \(\widehat C_{ij}\) estima aproximadamente \(T_h p_{ij}^{(h)}\), con \(T_h\approx 2hT\) salvo bordes.

### Proposición 6.1 — Error efectivo de co-ocurrencia

Bajo independencia aproximada o mezcla fuerte, para todo \(\eta>0\),

\[
\mathbb{P}\left(
\left|
\frac{\widehat C_{ij}}{T_h}-p_{ij}^{(h)}
\right|>\eta
\right)
\lesssim
2\exp(-cT_h\eta^2),
\]

donde \(c\) depende de la mezcla temporal.

Por unión sobre \(n^2\) pares:

\[
\mathbb{P}\left(
\|\widehat P-P\|_\infty>\eta
\right)
\lesssim
2n^2\exp(-cT_h\eta^2).
\]

Entonces, para controlar todos los pares con probabilidad \(1-\alpha\), se requiere aproximadamente

\[
T_h
\gtrsim
\frac{1}{c\eta^2}
\log\left(\frac{2n^2}{\alpha}\right).
\]

### Consecuencia para Rongorongo

Si \(n\) es grande y \(T\) es pequeño, el régimen es adverso. Con pocos miles de signos, muchos pares no se observan. Entonces:

- PPMI se vuelve ruidoso.
- Los ceros pueden ser ausencia de evidencia, no evidencia de ausencia.
- Los valores singulares pequeños son inestables.
- Se necesita regularización, pooling por clases, priors paleográficos e incertidumbre.

### Mejora concreta

Agrega un índice:

\[
\mathrm{CoocCoverage}(h)
=
\frac{
|\{(i,j):\widehat C_{ij}>0\}|
}{n^2}.
\]

Y otro:

\[
\mathrm{ExpectedPairCount}
=
\frac{2hT}{n^2}.
\]

Si \(\mathrm{ExpectedPairCount}\ll 1\), tu matriz de co-ocurrencia es estadísticamente débil.

---

## 7. PPMI regularizado y propagación de error

La PMI es inestable cuando las probabilidades son pequeñas.

\[
\operatorname{PMI}_{ij}
=
\log \frac{p_{ij}}{p_i p_j}.
\]

Si \(p_{ij}\) está cerca de cero, un pequeño error relativo explota.

### Proposición 7.1 — Sensibilidad local de PMI

Sea

\[
f(p_{ij},p_i,p_j)=\log p_{ij}-\log p_i-\log p_j.
\]

Entonces el diferencial es

\[
df
=
\frac{dp_{ij}}{p_{ij}}
-
\frac{dp_i}{p_i}
-
\frac{dp_j}{p_j}.
\]

Por tanto,

\[
|df|
\leq
\frac{|dp_{ij}|}{p_{ij}}
+
\frac{|dp_i|}{p_i}
+
\frac{|dp_j|}{p_j}.
\]

### Corolario 7.2 — No usar PMI cruda en baja frecuencia

Si \(p_{ij},p_i,p_j\) son pequeños, la PMI es altamente sensible. En corpus Rongorongo debes usar:

\[
\widehat p_{ij}^{(\epsilon)}
=
\frac{C_{ij}+\epsilon q_{ij}}{N+\epsilon},
\]

donde \(q_{ij}\) es un prior estructural: uniforme, producto marginal, o prior de clase.

### Recomendación

Reemplaza PPMI simple por:

\[
\operatorname{SPPMI}_{ij}
=
\max\left\{
\log
\frac{\widehat p_{ij}^{(\epsilon)}}{\widehat p_i^{(\epsilon)}\widehat p_j^{(\epsilon)}}
-
\log k_{\text{neg}},
0
\right\},
\]

y reporta sensibilidad a \(\epsilon\) y \(k_{\text{neg}}\).

---

## 8. Procrustes: falta error bajo anclajes ruidosos

Tu paper deriva Procrustes, pero necesita análisis de perturbación.

### Modelo

Supón que hay anclajes \(A=\{(x_a,y_a)\}_{a=1}^m\). Los embeddings verdaderos satisfacen

\[
Y_A=X_A Q_0 + R,
\]

donde \(Q_0\in O(d)\) y \(R\) es error de modelo: no-isometría, ruido semántico, anclajes imperfectos.

Observamos:

\[
\widehat X_A=X_A+\Delta_X,\qquad
\widehat Y_A=Y_A+\Delta_Y.
\]

El estimador Procrustes es

\[
\widehat Q
=
\arg\min_{Q^\top Q=I}
\|\widehat X_A Q-\widehat Y_A\|_F^2.
\]

### Teorema 8.1 — Estabilidad de Procrustes

Sea

\[
C=X_A^\top Y_A,\qquad \widehat C=\widehat X_A^\top \widehat Y_A.
\]

Supón que \(C\) tiene gap singular efectivo

\[
\gamma=\sigma_d(C)-\sigma_{d+1}(C)>0,
\]

con \(\sigma_{d+1}=0\) si \(C\in\mathbb{R}^{d\times d}\). Si

\[
\|\widehat C-C\|_2<\gamma,
\]

entonces

\[
\|\widehat Q-Q_0\|_F
\leq
K\frac{\|\widehat C-C\|_2}{\gamma}
+
\text{término de misspecification por }R.
\]

**Demostración.**  
El estimador Procrustes depende de los factores singulares de \(\widehat C\). Por perturbación de subespacios singulares, la variación de los factores \(U,V\) está controlada por \(\|\widehat C-C\|_2/\gamma\). Como \(Q=UV^\top\), se obtiene la cota por desigualdad triangular y estabilidad de productos ortogonales. El término \(R\) aparece porque el \(Q_0\) perfecto no minimiza exactamente si \(Y_A\neq X_AQ_0\). \(\square\)

### Métrica nueva

\[
\mathrm{AnchorCondition}(A)=\sigma_d(X_A^\top Y_A).
\]

Si esta métrica es baja, tus anclajes son geométricamente degenerados.

### Protocolo

Para cada set de anclajes:

1. calcular \(\mathrm{AnchorPower}\);
2. calcular \(\mathrm{AnchorCondition}\);
3. hacer leave-one-anchor-out;
4. hacer bootstrap de anclajes;
5. medir estabilidad de \(Q\):

\[
\mathrm{QStability}
=
\mathbb{E}_{b,b'}
\left[
\|Q^{(b)}-Q^{(b')}\|_F
\right].
\]

---

## 9. Transporte óptimo: falta separar costo geométrico, costo relacional y priors

Tu paper usa transporte óptimo y Gromov–Wasserstein. El upgrade es definir el problema completo con auditoría.

### Formulación

Sea \(\Pi\in \mathbb{R}_+^{n_X\times n_Y}\) una matriz de acoplamiento:

\[
\Pi \mathbf{1}=a,\qquad
\Pi^\top \mathbf{1}=b.
\]

Define:

\[
\mathcal{L}(\Pi,Q)
=
\lambda_g \sum_{i,j}\Pi_{ij}D_{ij}(Q)
+
\lambda_r
\sum_{i,i',j,j'}
L(\Delta^X_{ii'},\Delta^Y_{jj'})
\Pi_{ij}\Pi_{i'j'}
+
\lambda_p
\sum_{i,j}\Pi_{ij}P_{ij}
+
\varepsilon
\sum_{i,j}
\Pi_{ij}(\log\Pi_{ij}-1).
\]

Donde:

- \(D_{ij}(Q)=\|e_i^XQ-e_j^Y\|^2\).
- \(\Delta^X,\Delta^Y\) son distancias internas.
- \(P_{ij}\) es penalización o prior negativo.
- \(\varepsilon\) controla entropía.

### Definición 9.1 — Separabilidad auditable

Un resultado de transporte es auditable si reporta:

\[
\mathcal{L}_g,\quad
\mathcal{L}_r,\quad
\mathcal{L}_p,\quad
\mathcal{H}(\Pi),\quad
\text{sensibilidad a }(\lambda_g,\lambda_r,\lambda_p,\varepsilon).
\]

No basta entregar \(\Pi\).

### Teorema 9.2 — Entropía impide colapso prematuro

Para \(\varepsilon>0\), el término

\[
\varepsilon\sum_{ij}\Pi_{ij}(\log\Pi_{ij}-1)
\]

hace estrictamente convexo el subproblema de transporte clásico con costo lineal sobre el interior del politopo de transporte. En particular, evita soluciones extremadamente degeneradas cuando los costos son casi empatados.

**Demostración.**  
La función \(x\mapsto x\log x-x\) tiene segunda derivada \(1/x>0\) para \(x>0\). Por suma ponderada, la entropía negativa es estrictamente convexa en el interior. Agregarla a un costo lineal produce objetivo estrictamente convexo en \(\Pi\) para el problema clásico. \(\square\)

### Advertencia

Gromov–Wasserstein no es convexo globalmente. Debes reportar múltiples inicializaciones:

\[
\Pi^{(1)},\dots,\Pi^{(B)}
\]

y estabilidad:

\[
\mathrm{OTStability}
=
\mathbb{E}_{b,b'}
\left[
\|\Pi^{(b)}-\Pi^{(b')}\|_1
\right].
\]

---

## 10. Modelo generativo Rongorongo: falta una probabilidad completa

El paper debe pasar de “pipeline” a modelo probabilístico generativo.

### Variables

- \(Y=(y_1,\dots,y_M)\): entrada candidata en Rapanui/proto-polinesio/conceptos.
- \(X=(x_1,\dots,x_T)\): secuencia de glifos.
- \(Z\): segmentación latente.
- \(A\): alineamiento entre unidades de \(Y\) y glifos de \(X\).
- \(C_t\): clase funcional del glifo \(x_t\).
- \(D_t\): dirección/posición/línea.
- \(U_t\): incertidumbre paleográfica.
- \(R\): registro/género: canto, genealogía, calendario, ritual, lista, etiqueta.
- \(K\): conocimiento externo.

### Modelo

\[
p(X,Z,A,C,D,U,R\mid Y,K)
=
p(R\mid K)
p(Z,A\mid Y,R,K)
\prod_{t=1}^T
p(x_t\mid y_{A_t},C_t,D_t,U_t,R,K)
p(C_t\mid C_{<t},R,K)
p(D_t\mid t,\mathrm{artifact},K)
p(U_t\mid \mathrm{image},K).
\]

Marginal:

\[
p(X\mid Y,K)
=
\sum_{Z,A,C,D,U,R}
p(X,Z,A,C,D,U,R\mid Y,K).
\]

### Traducción-candidata

\[
X^\star
=
\arg\max_X p(X\mid Y,K).
\]

Pero la salida científica debe ser:

\[
\mathcal{H}_N(Y)
=
\left\{
(X^{(m)},p_m,E_m,B_m,\ell_m)
\right\}_{m=1}^N,
\]

donde:

- \(p_m\): score posterior o score calibrado;
- \(E_m\): evidencia;
- \(B_m\): contraevidencia;
- \(\ell_m\): claim level máximo permitido.

### Comentario serio

Este modelo admite que el “traductor” sea generativo, no literal. La dirección Rapanui \(\to\) Rongorongo no prueba desciframiento; prueba compatibilidad con una hipótesis de codificación.

---

## 11. Teorema de imposibilidad de traducción fuerte sin anclajes externos

### Teorema 11.1 — Ningún modelo secuencial interno prueba traducción semántica fuerte

Sea \(X\) un corpus Rongorongo y sea \(Y\) un corpus candidato. Supongamos que un modelo maximiza

\[
\max_\theta p_\theta(X\mid Y)
\]

usando solo restricciones estadísticas internas y priors no anclados. Entonces, para cualquier traducción fuerte \(T:X\to Y\), existe una clase de traducciones alternativas \(T'\) con igual evidencia interna bajo simetrías de renombramiento, colapso o expansión latente. Por tanto, \(p_\theta(X\mid Y)\) no puede certificar \(C_5\).

**Demostración.**  
La verosimilitud secuencial mide compatibilidad distributiva, no verdad semántica. Si dos modelos \(T,T'\) inducen la misma distribución sobre observables \(X\), entonces son observacionalmente equivalentes. Por el teorema de no-identificabilidad, renombramientos de glifos y transformaciones latentes que preservan co-ocurrencias producen la misma evidencia estructural. Además, si el modelo permite clases latentes, pueden existir refinamientos o colapsos que preserven marginales observables. Sin una variable externa observada que conecte un glifo con una entidad semántica particular, la semántica fuerte no queda fijada. \(\square\)

### Consecuencia

El sistema debe bloquear claims \(C_5\) automáticamente salvo evidencia externa.

---

# Parte III — Datos: cómo obtener lo necesario para Rongorongo

## 12. Principio central: no hay “texto limpio” Rongorongo

Antes de ML, necesitas una base paleográfica. Rongorongo no es un corpus Unicode limpio. Es una colección de objetos físicos de madera, con daños, orientaciones, variantes gráficas, ligaduras y lecturas históricas divergentes.

Tu pipeline debe partir de:

\[
\text{artefacto}
\rightarrow
\text{imagen/modelo 3D}
\rightarrow
\text{trazo}
\rightarrow
\text{glifo incierto}
\rightarrow
\text{secuencia probabilística}
\rightarrow
\text{hipótesis}.
\]

## 13. Fuentes de datos Rongorongo

### 13.1 Corpus Barthel digitalizado

Hay una fuente pública que presenta una transliteración numérica del corpus de Thomas Barthel en formato computacional. Debes usarla como capa histórica, no como verdad definitiva.

**Uso recomendado:**

- prototipo de secuencias;
- cálculo de frecuencias;
- identificación de pasajes paralelos;
- baseline reproducible.

**Riesgo:**

- codificación antigua;
- errores de normalización;
- dependencia de la clasificación Barthel;
- no incorpora lecturas 3D recientes.

### 13.2 Imágenes y transcripciones secundarias

Hay colecciones en Internet Archive con imágenes y transcripciones. Úsalas para prototipos visuales, no como corpus crítico final.

### 13.3 Modelado 3D y fotogrametría

La literatura reciente sobre la tablilla Échancrée y Mamari muestra que la reconstrucción 3D/fotogramétrica puede corregir dibujos y transcripciones anteriores. Esto es crucial: un glifo mal leído cambia todo el grafo.

**Acción concreta:**

Contactar autores o instituciones asociadas a trabajos de Lastilla, Ravanelli, Valério, Ferrara y museos custodios para solicitar:

- modelos 3D;
- imágenes ortorrectificadas;
- trazados vectoriales;
- transcripciones corregidas;
- metadatos de incertidumbre.

### 13.4 Rongopy como baseline exploratorio

Rongopy es útil porque ya intentó cargar corpus, experimentar con secuencias y entrenar modelos. Su valor principal no es que “resuelva” Rongorongo, sino que entrega un baseline honesto: los resultados seq2seq con datos escasos no bastan.

Úsalo para:

- replicar baseline;
- comparar tokenizer;
- comparar corpus;
- diseñar “failure cases”.

### 13.5 Fuentes lingüísticas candidatas

Necesitas corpus \(Y\) para Rapanui y lenguas relacionadas:

- Rapanui moderno.
- Tahitian.
- Māori.
- Mangarevan.
- Marquesan.
- Samoan.
- Tongan.
- Proto-polinesio reconstruido.
- Léxicos rituales/calendáricos.
- Genealogías.
- Cantos y recitaciones.

La Austronesian Basic Vocabulary Database sirve para léxico comparativo básico. Glottolog sirve para genealogía, clasificación y referencias bibliográficas. Pero un traductor Rongorongo necesita más que vocabulario: necesita géneros culturales plausibles.

## 14. Estructura de corpus recomendada

```text
data/
  rongorongo/
    raw/
      barthel_numeric/
      images/
      3d_models/
      secondary_transcriptions/
    processed/
      artifacts.jsonl
      lines.jsonl
      glyph_instances.jsonl
      glyph_catalog.jsonl
      uncertain_readings.jsonl
      parallel_passages.jsonl
      direction_metadata.jsonl
      paleographic_sources.jsonl
    derived/
      cooccurrence/
      transition/
      positional/
      repetition/
      graph/
      embeddings/
  candidate_languages/
    rapanui/
    tahitian/
    maori/
    mangarevan/
    marquesan/
    proto_polynesian/
    ritual_calendar/
  experiments/
    synthetic/
    rongorongo_like/
    negative_controls/
```

## 15. Esquema JSONL por glifo

```json
{
  "glyph_instance_id": "RR_A_r_03_017",
  "artifact_id": "A",
  "artifact_name": "Tahua",
  "side": "r",
  "line": 3,
  "position_in_line": 17,
  "global_position": 231,
  "direction": "reverse_boustrophedon",
  "barthel_code": "200",
  "alternative_codes": [
    {"system": "Horley", "code": "H_xxx", "confidence": 0.78}
  ],
  "bbox_2d": [123.4, 511.2, 18.0, 27.5],
  "surface_coordinates_3d": null,
  "damage_score": 0.21,
  "reading_uncertainty": 0.34,
  "is_ligature": false,
  "possible_components": [],
  "source_refs": ["barthel1958", "lastilla2022"],
  "notes": "uncertain lower stroke"
}
```

## 16. Esquema JSONL por hipótesis

```json
{
  "hypothesis_id": "HYP_RR_000123",
  "claim_level": "C2_functional",
  "glyph_or_sequence": ["200", "076", "076"],
  "candidate_interpretation": {
    "type": "functional",
    "label": "terminal classifier or repetitive formula"
  },
  "posterior_score": 0.42,
  "calibrated_probability": null,
  "evidence": [
    {
      "type": "positional",
      "description": "sequence appears disproportionately near line end",
      "score": 3.1,
      "p_value": 0.004
    }
  ],
  "counterevidence": [
    {
      "type": "low_sample",
      "description": "only 7 occurrences across corpus"
    }
  ],
  "negative_control_gap": 2.4,
  "bootstrap_stability": 0.73,
  "anchor_power": 0.11,
  "forbidden_claims": ["phonetic reading", "literal translation"],
  "reproducibility": {
    "run_id": "20260429_001",
    "config_hash": "..."
  }
}
```

---

# Parte IV — Nuevos experimentos matemáticos obligatorios

## 17. Experimento 1: recuperación bajo permutación conocida

### Objetivo

Validar que Procrustes/OT recuperan correspondencias cuando las premisas son verdaderas.

### Generación

1. Toma corpus candidato \(Y\).
2. Construye vocabulario \(V_Y\).
3. Genera escritura artificial \(X\) con permutación \(\pi\):

\[
x_t=\pi(y_t).
\]

4. Oculta \(\pi\).
5. Entrega \(m\) anclajes.

### Métricas

\[
\mathrm{Acc@k}
=
\frac{1}{|V|}
\sum_i
\mathbf{1}\{\pi(i)\in \mathrm{TopK}(i)\}.
\]

\[
\mathrm{MRR}
=
\frac{1}{|V|}
\sum_i
\frac{1}{\mathrm{rank}_i}.
\]

### Claim permitido

Si falla aquí, el método no puede aplicarse a Rongorongo.

---

## 18. Experimento 2: colapso logosilábico

### Objetivo

Simular que varios significados/sílabas colapsan a un mismo glifo.

\[
g:V_Y\to V_X,\qquad |g^{-1}(x)|>1.
\]

### Métrica

No uses Acc@1 simple. Usa:

\[
\mathrm{FiberRecall@k}
=
\frac{1}{|V_X|}
\sum_{x\in V_X}
\frac{
|g^{-1}(x)\cap \mathrm{TopK}(x)|
}{
|g^{-1}(x)|
}.
\]

### Teorema 18.1 — Error irreducible de modelos biyectivos

Si existe \(x\) con \(|g^{-1}(x)|>1\), entonces cualquier modelo que imponga una biyección \(\tau:V_X\to V_Y\) incurre en error no nulo sobre la fibra.

**Demostración.**  
Una biyección asigna a \(x\) un único \(y\). Si la verdad contiene al menos dos elementos \(y_1,y_2\in g^{-1}(x)\), entonces al menos uno queda excluido. Por tanto, la recuperación completa de la fibra es imposible bajo biyección. \(\square\)

### Conclusión

Tu modelo debe devolver fibras/distribuciones, no diccionarios duros.

---

## 19. Experimento 3: segmentación desconocida

Rongorongo puede tener ligaduras, compuestos, biglifos y bloques.

### Modelo

\[
p(X,Z\mid Y)
=
p(Z\mid Y)
\prod_{b\in Z}
p(x_b\mid y_{a(b)}).
\]

Donde \(Z\) particiona la secuencia en bloques.

### Métrica

\[
\mathrm{SegF1}
=
\frac{2PR}{P+R}
\]

para cortes de segmentación en sintéticos con ground truth.

### Baseline

Comparar con:

- glifo individual;
- biglifos;
- BPE;
- unigram LM segmentation;
- HMM segmentation;
- modelo conjunto segmentación-alineamiento.

---

## 20. Experimento 4: orientación boustrophedon

### Problema

Si alterna la dirección por línea, las co-ocurrencias izquierda/derecha se mezclan. Debes modelar dirección.

### Variables

\[
d_\ell\in\{-1,+1\}
\]

para cada línea \(\ell\).

### Likelihood

\[
p(X_\ell\mid d_\ell,\theta)
=
\prod_t
p_\theta(x_{\ell,t+d_\ell}\mid x_{\ell,t}).
\]

### Inferencia

\[
\hat d_\ell
=
\arg\max_{d\in\{-1,+1\}}
p(X_\ell\mid d,\theta).
\]

### Experimento

Genera corpus sintético con direcciones alternadas y evalúa si el modelo recupera orientación.

---

## 21. Experimento 5: pasajes paralelos

Rongorongo tiene secuencias repetidas y pasajes paralelos. Eso puede ser más informativo que co-ocurrencia local.

### Definición

Una pareja de segmentos \((a,b)\) es paralela si

\[
\operatorname{EditSim}(X_a,X_b)>\tau.
\]

Construye un grafo de segmentos:

\[
G_{\mathrm{parallel}}=(\mathcal{S},E_p).
\]

### Claim

Pasajes paralelos pueden inducir unidades formulaicas, pero no traducciones.

### Métrica

\[
\mathrm{ParallelStability}
=
\mathbb{E}_{\mathrm{bootstrap}}
\operatorname{Jaccard}(E_p,E_p^{(b)}).
\]

---

## 22. Experimento 6: calendario lunar

Si una parte del corpus se hipotetiza calendárica, se debe modelar explícitamente.

### Modelo

Sea \(L_t\in\{1,\dots,30\}\) fase lunar. Define

\[
p(x_t\mid L_t,c_t)
\]

y transiciones

\[
p(L_{t+1}=L_t+1 \mod 30)=1-\epsilon.
\]

### Hipótesis calendárica

\[
H_{\mathrm{cal}}: X \text{ fue generado por cadena lunar latente}.
\]

Comparar contra:

\[
H_{\mathrm{ngram}}: X \text{ fue generado por n-gram general}.
\]

Usar razón de evidencia aproximada:

\[
\Delta \mathrm{BIC}
=
\mathrm{BIC}(H_{\mathrm{ngram}})
-
\mathrm{BIC}(H_{\mathrm{cal}}).
\]

Si \(\Delta\mathrm{BIC}>0\), calendario mejora penalizado por complejidad.

---

# Parte V — Métricas nuevas para auditabilidad

## 23. Gap contra controles negativos

Para cualquier score \(S\):

\[
\mathrm{NegCtrlGap}(S)
=
\frac{
S(X)-\mathbb{E}_{X'\sim H_0}S(X')
}{
\operatorname{sd}_{X'\sim H_0}S(X')
}.
\]

Regla:

- \(<1\): no evidencia.
- \(1\) a \(2\): débil.
- \(2\) a \(3\): moderada.
- \(>3\): fuerte.
- \(>5\): muy fuerte, pero revisar leakage.

## 24. Estabilidad bootstrap

\[
\mathrm{Stability}(h)
=
\mathbb{E}_{b,b'}
\left[
\operatorname{sim}(h^{(b)},h^{(b')})
\right].
\]

Para distribuciones:

\[
\operatorname{sim}(\Pi,\Pi')
=
1-
\frac{1}{2}
\|\Pi-\Pi'\|_1.
\]

## 25. Calibración

Si emites probabilidades, debes medir calibración.

Divide predicciones en bins \(B_m\). Define:

\[
\mathrm{ECE}
=
\sum_m
\frac{|B_m|}{N}
\left|
\mathrm{acc}(B_m)-\mathrm{conf}(B_m)
\right|.
\]

En Rongorongo real no hay ground truth, pero puedes calibrar en sintéticos y luego marcar probabilidades reales como “scores calibrados bajo simulación”, no verdad histórica.

## 26. Índice de sobreinterpretación

Define:

\[
\mathrm{OverclaimRisk}(h)
=
\frac{\mathrm{ClaimLevel}(h)}
{1+\mathrm{EvidenceLevel}(h)}.
\]

Donde ClaimLevel se codifica \(0,\dots,5\) y EvidenceLevel combina anclaje, estabilidad y gap.

Si:

\[
\mathrm{OverclaimRisk}(h)>1,
\]

bloquear publicación del claim.

---

# Parte VI — Arquitectura matemática del paper reescrito

## 27. Título recomendado

**Geometría espectral e identificabilidad parcial en sistemas simbólicos no descifrados: un marco auditable para hipótesis Rongorongo**

O en inglés:

**Spectral Geometry and Partial Identifiability in Undeciphered Symbolic Systems: An Auditable Framework for Rongorongo Hypothesis Generation**

## 28. Abstract recomendado

> We propose a mathematically explicit framework for the structural analysis and probabilistic hypothesis generation of undeciphered symbolic systems under extreme data scarcity. Rather than claiming deterministic decipherment, we formalize decipherment as a partially identifiable inverse problem under group symmetries, noisy paleographic observations, weak anchors and relational constraints. We prove non-identifiability results for unanchored translation, characterize how anchors break automorphism groups, derive stability conditions for spectral embeddings under finite-corpus perturbations, and define auditable claim levels that prevent semantic overreach. The framework combines regularized co-occurrence estimation, spectral embeddings, Procrustes alignment, entropic optimal transport, latent segmentation and Bayesian posterior scoring. Rongorongo is treated as a motivating low-resource case: the output is not a verified translation, but a ranked ledger of falsifiable hypotheses with evidence, counterevidence and uncertainty.

## 29. Nueva estructura del paper

```text
1. Introduction
2. Related Work
   2.1 Ancient language ML
   2.2 Automatic decipherment
   2.3 Cross-lingual embeddings
   2.4 Paleographic uncertainty in Rongorongo
3. Problem Formulation
   3.1 Corpus as uncertain symbolic observation
   3.2 Translation as inverse problem
   3.3 Claim levels
4. Non-identifiability Theory
   4.1 Group action on glyph vocabularies
   4.2 No-free-decipherment theorem
   4.3 Orbits and equivalence classes
5. Anchors and Partial Identifiability
   5.1 Anchor taxonomy
   5.2 Automorphism-breaking theorem
   5.3 AnchorPower and AnchorCondition
6. Spectral Estimation under Scarcity
   6.1 Co-occurrence concentration
   6.2 Regularized PPMI
   6.3 Singular gap stability
7. Alignment and Transport
   7.1 Procrustes with noisy anchors
   7.2 Gromov-Wasserstein matching
   7.3 Entropic regularization
8. Generative Rongorongo Hypothesis Model
   8.1 Latent segmentation
   8.2 Direction and artifact metadata
   8.3 Functional classes
   8.4 Posterior inference
9. Experimental Protocol
   9.1 Synthetic permutation
   9.2 Logosyllabic collapse
   9.3 Undersegmentation
   9.4 Boustrophedon
   9.5 Parallel passages
   9.6 Calendar model
10. Real Rongorongo Data Acquisition Plan
11. Auditability and Reproducibility
12. Discussion: What can and cannot be claimed
13. Conclusion
Appendix A. Proofs
Appendix B. Corpus schemas
Appendix C. Hypothesis ledger
Appendix D. Reproducibility checklist
```

---

# Parte VII — Secciones listas para insertar en LaTeX

## 30. Sección: Translation as a partially identifiable inverse problem

```latex
\section{Decipherment as a Partially Identifiable Inverse Problem}

Let \(V_X\) denote the finite vocabulary of an undeciphered symbolic system and let
\(X=(s_1,\ldots,s_T)\), \(s_t\in V_X\), be the observed corpus. We do not assume that
\(X\) is a clean textual object. Instead, each token is the result of a paleographic observation
process with uncertainty. Let \(\Theta\) be a parameter space containing segmentation rules,
glyph classes, alignment variables, phonetic or semantic assignments, and generative parameters.
A decipherment model defines an observation operator
\[
\mathcal{O}:\Theta\to \mathcal{D}_X,
\]
where \(\mathcal{D}_X\) is the space of observable symbolic corpora and their metadata.
Recovering \(\theta\in\Theta\) from \(X\) is an inverse problem.

In the absence of external anchors, this inverse problem is generally ill-posed. In particular,
the observation operator is not injective: distinct semantic assignments may induce identical
observable structural statistics. Therefore the scientifically valid target is not a deterministic
translation map, but a posterior distribution over equivalence classes of hypotheses.
```

## 31. Sección: No-free-decipherment theorem

```latex
\begin{theorem}[No-free-decipherment under glyph renaming]
Let \(G=\mathrm{Sym}(V_X)\) act on corpora by \((\pi X)_t=\pi(s_t)\).
Let \(S\) be any structural statistic satisfying \(S(\pi X)=S(X)\) for all \(\pi\in G\).
If a decipherment likelihood depends on the corpus only through \(S(X)\), then no absolute
semantic assignment \(\tau:V_X\to\mathcal{M}\) is identifiable without external anchors.
Only its orbit under \(G\) is identifiable.
\end{theorem}

\begin{proof}
For any \(\pi\in G\), define \(\tau_\pi=\tau\circ\pi^{-1}\). Then for each position \(t\),
\[
\tau_\pi((\pi X)_t)
=
\tau_\pi(\pi(s_t))
=
\tau(s_t).
\]
Thus the semantic sequence induced by \((X,\tau)\) is identical to that induced by
\((\pi X,\tau_\pi)\). Since \(S(\pi X)=S(X)\), any likelihood depending only on \(S\)
assigns equal observational evidence to both hypotheses. Therefore the representative
\(\tau\) is not identifiable; only the equivalence class
\[
[\tau]=\{\tau\circ\pi^{-1}:\pi\in G\}
\]
is identifiable. \(\square\)
\end{proof}
```

## 32. Sección: AnchorPower

```latex
\begin{definition}[Anchor power]
Let \(G_X=(V_X,E,W)\) be a weighted glyph graph and let
\[
\mathrm{Aut}(G_X)=\{\pi\in\mathrm{Sym}(V_X):W_{\pi(i)\pi(j)}=W_{ij}\ \forall i,j\}.
\]
Given a set of anchors \(A\), define
\[
\mathrm{Aut}(G_X;A)
=
\{\pi\in\mathrm{Aut}(G_X):\pi(x)=x\ \forall (x,y,\rho,\mathcal{E})\in A,\rho>\rho_0\}.
\]
The anchor power of \(A\) is
\[
\mathrm{AnchorPower}(A)
=
1-\frac{\log(|\mathrm{Aut}(G_X;A)|+1)}
{\log(|\mathrm{Aut}(G_X)|+1)}.
\]
\end{definition}
```

## 33. Sección: Spectral stability

```latex
\begin{theorem}[Spectral stability under finite-corpus perturbation]
Let \(M\) be a population structural matrix and \(\widehat M=M+E\) its empirical estimate.
Let \(U_k\) and \(\widehat U_k\) be the left singular subspaces associated with the top \(k\)
singular values of \(M\) and \(\widehat M\). If
\[
\delta_k=\sigma_k(M)-\sigma_{k+1}(M)>0
\quad\text{and}\quad
\|E\|_2<\delta_k,
\]
then
\[
\|\sin\Theta(\widehat U_k,U_k)\|_2
\leq
\frac{\|E\|_2}{\delta_k-\|E\|_2}.
\]
\end{theorem}
```

## 34. Sección: Hypothesis ledger principle

```latex
\begin{principle}[Auditable hypothesis ledger]
No output of the system may be reported as a bare equality \(x=y\). Every hypothesis must be
reported as a tuple
\[
h=(x,\mathcal{Y}_h,p_h,E_h,B_h,\ell_h,R_h),
\]
where \(x\) is a glyph or sequence, \(\mathcal{Y}_h\) is a set of candidate interpretations,
\(p_h\) is a calibrated or simulation-calibrated score, \(E_h\) is evidence, \(B_h\) is
counterevidence, \(\ell_h\in\{C_0,\ldots,C_5\}\) is the maximum admissible claim level, and
\(R_h\) is reproducibility metadata.
\end{principle}
```

---

# Parte VIII — Cómo posicionarlo frente al SOTA

## 35. Qué aprender de Pythia/Ithaca

Pythia e Ithaca son fuertes porque:

- tienen corpus procesable;
- definen tareas cerradas;
- reportan top-k;
- comparan con expertos;
- tratan el sistema como apoyo interpretativo, no oráculo;
- dan interpretabilidad y contexto.

Tu equivalente Rongorongo debe ser:

- no “traducción absoluta”;
- sí “top-k glifos/secuencias compatibles”;
- sí “hipótesis auditadas”;
- sí “comparación contra baselines y controles”.

## 36. Qué aprender de Luo/Barzilay

Los papers de neural decipherment son fuertes porque:

- formalizan desciframiento como optimización;
- incorporan restricciones lingüísticas;
- usan lenguas descifradas para evaluación;
- tratan segmentación desconocida;
- modelan parentesco lingüístico.

Tu mejora debe ser:

- añadir segmentación latente;
- comparar con modelos de cognados/fonética;
- aceptar que Rongorongo puede no ser alfabético ni puramente fonético;
- integrar modelos funcionales y no solo fonéticos.

## 37. Qué aprender de Tamburini

El punto clave es permitir:

- mappings nulos;
- uno-a-muchos;
- muchos-a-uno;
- no correspondencia total.

Esto conecta con tu submersión y fibras:

\[
F^{-1}(x)=\{y\in\mathcal{M}_Y:F(y)=x\}.
\]

Debes usarlo como baseline combinatorio contra OT.

---

# Parte IX — Plan de investigación de 12 semanas

## Semana 1–2: Corpus ledger

- Descargar/cargar Barthel digital.
- Crear parser a JSONL.
- Crear manifest de fuentes.
- Crear normalizador de códigos.
- Crear datasheet inicial.
- Separar lecturas ciertas/inciertas.

## Semana 3–4: Análisis estructural básico

- Frecuencias.
- Longitudes.
- Repeticiones AA/AAA/ABAB.
- Posición relativa.
- Transiciones.
- Boustrophedon correction.
- Pasajes paralelos.

## Semana 5–6: Matrices y estabilidad

- Co-ocurrencia multi-ventana.
- PPMI regularizado.
- Laplaciano.
- SVD multi-k.
- Bootstrap.
- Singular gaps.
- SpectralReliability.

## Semana 7–8: Sintéticos

- Permutación.
- Colapso logosilábico.
- Segmentación desconocida.
- Boustrophedon.
- Calendario.
- Pasajes paralelos.

## Semana 9–10: Modelos candidatos

- Rapanui léxico.
- Lenguas polinesias comparativas.
- Calendario lunar.
- Vocabularios rituales.
- Embeddings candidatos.
- Priors iconográficos.

## Semana 11: Alineamiento y transporte

- Procrustes.
- Gromov-Wasserstein.
- Tamburini-style combinatorial baseline.
- HMM functional baseline.
- Ablations.

## Semana 12: Paper y auditoría

- Hypothesis cards.
- Model card.
- Dataset card.
- Reproducibility package.
- Makefile.
- Apéndice de pruebas.

---

# Parte X — Reproducibilidad obligatoria

## 38. Comando único

```bash
make reproduce_all
```

Debe ejecutar:

```bash
make data
make preprocess
make synthetic
make experiments
make figures
make tables
make paper
```

## 39. Archivos por run

```text
runs/2026-04-29_001/
  config.yaml
  git_commit.txt
  environment.yml
  corpus_manifest.json
  source_versions.json
  random_seeds.json
  metrics.json
  spectral_stability.json
  negative_controls.json
  bootstrap.json
  hypotheses.jsonl
  model_card.md
  dataset_card.md
  figures/
  tables/
  logs.txt
```

## 40. Config mínima

```yaml
corpus:
  source: barthel_numeric
  include_uncertain: false
  direction_correction: true

matrix:
  type: sppmi
  window_sizes: [1, 2, 3, 5]
  smoothing: 0.1
  alpha: [0.0, 0.5, 1.0]

spectral:
  k_values: [4, 8, 16, 32]
  bootstrap_samples: 200

alignment:
  method: gromov_wasserstein
  entropy_epsilon: [0.01, 0.05, 0.1]
  seeds: 20

claims:
  max_claim_level_without_external_anchor: C2
  require_negative_control_gap: 2.0
  require_bootstrap_stability: 0.7
```

---

# Parte XI — Checklist matemático para reviewer

Antes de enviar el paper, cada claim debe responder:

1. ¿Cuál es el objeto matemático?
2. ¿Cuál es el espacio de hipótesis?
3. ¿Qué simetrías lo hacen no identificable?
4. ¿Qué anclaje rompe cuáles simetrías?
5. ¿Cuál es la métrica de estabilidad?
6. ¿Cuál es el control negativo?
7. ¿Cuál es el nivel máximo de claim permitido?
8. ¿Qué pasa si cambio ventana, dimensión, smoothing, prior?
9. ¿Qué evidencia contradice la hipótesis?
10. ¿Dónde está el JSON reproducible?

---

# Parte XII — Bibliografía y fuentes que debes integrar

## ML para lenguas antiguas

- Sommerschield et al., *Machine Learning for Ancient Languages: A Survey*. Computational Linguistics, 2023.  
  URL: https://direct.mit.edu/coli/article/49/3/703/116160/Machine-Learning-for-Ancient-Languages-A-Survey

- Assael, Sommerschield & Prag, *Restoring ancient text using deep learning: a case study on Greek epigraphy*. EMNLP-IJCNLP 2019.  
  URL: https://aclanthology.org/D19-1668/

- Assael et al., *Restoring and attributing ancient texts using deep neural networks*. Nature, 2022.  
  URL: https://www.nature.com/articles/s41586-022-04448-z

## Desciframiento automático

- Luo, Cao & Barzilay, *Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B*. ACL 2019.  
  URL: https://aclanthology.org/P19-1303/

- Luo et al., *Deciphering Undersegmented Ancient Scripts Using Phonetic Prior*. TACL 2021.  
  URL: https://aclanthology.org/2021.tacl-1.5/

- Tamburini, *Automatic decipherment of lost languages via combinatorial optimisation*. CAWL 2023.  
  URL: https://aclanthology.org/2023.cawl-1.10/

## Rongorongo

- Digitized Barthel numerical transliteration.  
  URL: https://kohaumotu.org/rongorongo_org/corpus/digit.html

- Lastilla, Ravanelli & Valério, *Modelling the Rongorongo tablets: A new transcription of the Échancrée tablet and the foundation for decipherment attempts*.  
  URL: https://academic.oup.com/dsh/article/37/2/497/6387816

- Lastilla et al., *3D modelling of the Mamari tablet from Easter Island*. ISPRS Archives, 2019.  
  URL: https://isprs-archives.copernicus.org/articles/XLII-2-W18/85/2019/

- Ferrara et al., *The invention of writing on Rapa Nui (Easter Island). New radiocarbon dates on the Rongorongo script*. Scientific Reports, 2024.  
  URL: https://www.nature.com/articles/s41598-024-53063-7

- Rongopy project.  
  URL: https://jgregoriods.github.io/rongopy/

## Lenguas austronesias/polinesias

- Austronesian Basic Vocabulary Database.  
  URL: https://abvd.eva.mpg.de/austronesian/

- Glottolog database.  
  URL: https://glottolog.org/meta/downloads  
  Dataset: https://zenodo.org/records/15525265

## Transparencia y auditoría

- Gebru et al., *Datasheets for Datasets*.  
  URL: https://arxiv.org/abs/1803.09010

- Mitchell et al., *Model Cards for Model Reporting*.  
  URL: https://arxiv.org/abs/1810.03993

---

# Parte XIII — Veredicto final

La versión seria de tu investigación no debe decir:

\[
\text{“traducimos Rongorongo”}.
\]

Debe decir:

\[
\text{“construimos una teoría auditable de hipótesis Rongorongo parcialmente identificables”.}
\]

Tu contribución matemática fuerte sería:

1. Formular desciframiento como problema inverso mal puesto.
2. Probar no-identificabilidad bajo simetrías.
3. Formalizar anclajes como ruptura de automorfismos.
4. Derivar estabilidad espectral bajo corpus finito.
5. Derivar sensibilidad de PMI y Procrustes.
6. Reemplazar diccionarios duros por fibras/distribuciones.
7. Separar claims estructurales, funcionales, semánticos y fonéticos.
8. Forzar que todo resultado pase por controles negativos.
9. Diseñar un ledger reproducible de hipótesis.
10. Aplicar todo eso a Rongorongo con humildad científica y ambición técnica.

La frase que debe gobernar el paper:

> **La traducción perdida no es una clave única; es una clase de equivalencia parcialmente observable, que solo puede refinarse mediante anclajes externos, estabilidad estadística y falsación experimental.**

Ese es el salto de “idea brillante” a “investigación PhD revisable”.
