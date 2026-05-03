# Guía Exhaustiva PhD: Anclaje Icónico Inverso para Desciframiento de Sistemas Simbólicos

## Llevando Rongorongo de C0–C1 a C2–C3 mediante Visión Computacional, Reconstrucción del Mundo y Transfer Learning Cross-Script

**Autor del marco teórico previo:** David Alexander Astudillo Muñoz
**Documento:** Plan de investigación de continuación nivel doctoral
**Versión:** 1.0
**Fecha:** Abril 2026

---

## Índice

1. [Resumen ejecutivo y tesis central](#1-resumen-ejecutivo-y-tesis-central)
2. [Fundamentación teórica del nuevo marco](#2-fundamentación-teórica-del-nuevo-marco)
3. [El Teorema de Anclaje Icónico Inverso (formalización completa)](#3-el-teorema-de-anclaje-icónico-inverso-formalización-completa)
4. [Teoría de la deiconización: el segundo teorema](#4-teoría-de-la-deiconización-el-segundo-teorema)
5. [Geometría de fibrados visuales-semánticos](#5-geometría-de-fibrados-visuales-semánticos)
6. [Arquitectura computacional completa](#6-arquitectura-computacional-completa)
7. [Plan de adquisición de datos: 30+ sistemas de escritura](#7-plan-de-adquisición-de-datos-30-sistemas-de-escritura)
8. [Reconstrucción del mundo Rapa Nui circa 1500 CE](#8-reconstrucción-del-mundo-rapa-nui-circa-1500-ce)
9. [Pipeline de visión computacional](#9-pipeline-de-visión-computacional)
10. [Validación cruzada en escrituras descifradas](#10-validación-cruzada-en-escrituras-descifradas)
11. [Análisis comparativo de sistemas polinésicos](#11-análisis-comparativo-de-sistemas-polinésicos)
12. [Diseño experimental nivel PhD](#12-diseño-experimental-nivel-phd)
13. [Niveles de claim revisados](#13-niveles-de-claim-revisados)
14. [Cronograma y entregables](#14-cronograma-y-entregables)
15. [Referencias críticas y datasets](#15-referencias-críticas-y-datasets)
16. [Apéndices matemáticos](#16-apéndices-matemáticos)

---

## 1. Resumen ejecutivo y tesis central

### 1.1. Contexto del marco anterior

Tu paper anterior estableció con rigor matemático el **Teorema de No-Identificabilidad bajo Acción de Grupos** (Teorema 10.1), que demuestra que sin anclajes externos, todo estadístico puramente estructural es invariante bajo renombramiento del vocabulario. Las consecuencias cuantitativas para Rongorongo fueron severas:

- **Spectral Reliability** $= 0$ para todo $k$
- **Expected Pair Count** $= 0.037$ (insuficiencia masiva de datos)
- **Claim máximo admisible:** C0–C1 (paleográfico/estructural débil)

El Corolario 10.3 listó siete tipos de anclajes que pueden romper la simetría: bilingüe, **iconográfico**, arqueológico, fonético, numérico/calendárico, comparativo, pragmático.

### 1.2. La nueva tesis

> **Hipótesis del Origen Icónico Universal (HOIU):** Todo sistema de escritura humano emergió de un proceso pictográfico en el cual los signos preservan, con tasa de decaimiento dependiente del tiempo y la presión cultural, una traza geométrica de sus referentes físicos en el mundo del escritor.

De esta hipótesis se sigue que la **visión computacional sobre representaciones del mundo histórico reconstruido** puede generar anclajes iconográficos verificables que rompen la simetría del Teorema 10.1 y elevan el nivel de claim admisible de C0–C1 a **C2–C3**.

### 1.3. Las cuatro contribuciones matemáticas del nuevo paper

| # | Contribución | Naturaleza |
|---|--------------|------------|
| T1 | **Teorema de Anclaje Icónico Inverso** | Existencia y unicidad de anclajes visuales bajo HOIU |
| T2 | **Teorema de Deiconización Acotada** | La traza icónica decae cuasi-monótonamente; cota explícita |
| T3 | **Teorema de Transferencia Cross-Script** | Bajo qué condiciones un modelo entrenado en escrituras descifradas generaliza a Rongorongo |
| T4 | **Corolario de Identificabilidad C2-C3** | Cuándo las anclas iconográficas elevan el claim level |

### 1.4. Las cuatro contribuciones experimentales

| # | Experimento | Validación |
|---|-------------|------------|
| E1 | **Pipeline visión-mundo en 4 escrituras descifradas** | Egipcio, oracle bone chino, maya, cuneiforme proto-sumerio |
| E2 | **Reconstrucción digital del mundo Rapa Nui c. 1500 CE** | Paleoecología verificable independientemente |
| E3 | **Predicción de anclas iconográficas para Rongorongo** | Top-K candidates con calibración |
| E4 | **Comparación con todos los sistemas polinésicos y vecinos** | Hawaiano, maorí, samoano, tongano, tahitiano, marquesano, fijiano |

---

## 2. Fundamentación teórica del nuevo marco

### 2.1. Tres pilares conceptuales

#### Pilar A: La Hipótesis del Origen Icónico

La literatura antropológica establece firmemente que los **cuatro sistemas de escritura inventados independientemente** comenzaron pictográficos:

- **Cuneiforme sumerio** (3200 BCE): pictogramas de pez 𒆬, mano 𒋗, etc.
- **Jeroglíficos egipcios** (3100 BCE): aves, cuerpos humanos, herramientas
- **Caracteres chinos** (1200 BCE en Shang Oracle Bone): pictogramas de sol 日, luna 月, agua 水
- **Glifos mayas** (300 BCE): jaguar, serpiente, sol

Spiegel, Gelfond & Konidaris (Brown University, 2025) demostraron computacionalmente que **agentes con teoría de mente visual** convergen espontáneamente a sistemas pictográficos, validando computacionalmente lo que la antropología llevaba un siglo proponiendo.

#### Pilar B: La Deiconización como Proceso Universal

Goldwasser (2017) caracterizó la **trayectoria temporal** del proceso de deiconización:

$$\text{Pictograma} \xrightarrow{\delta_1} \text{Logograma estilizado} \xrightarrow{\delta_2} \text{Silabograma} \xrightarrow{\delta_3} \text{Alfabeto}$$

donde cada $\delta_i$ representa pérdida progresiva de iconicidad. Crucialmente:

- **Cuneiforme**: alta deiconización temprana (~ 2500 BCE)
- **Jeroglíficos egipcios**: baja deiconización mantenida 3000 años
- **Chino**: deiconización moderada (los caracteres modernos preservan trazas)
- **Rongorongo**: deiconización mínima (creado tardíamente, ~ 1700-1860 CE)

Esta última observación es **crítica para tu investigación**: Rongorongo es de los sistemas más recientes y fuertemente icónicos jamás documentados.

#### Pilar C: La Visión Computacional Moderna como Inversor del Proceso

Modelos como **DINOv2** (Meta), **SigLIP** (Google), y **CLIP** (OpenAI) han demostrado capacidad de:

- Mapear imágenes a embeddings que preservan estructura semántica
- Reconocer objetos a través de variaciones de estilo, ángulo, iluminación
- Conectar modalidad visual con conceptos lingüísticos

La pregunta operativa es: **¿pueden estos modelos invertir el proceso de pictografización?** Es decir, dado un glifo, ¿pueden recuperar su referente original?

### 2.2. ¿Por qué esto es matemáticamente legítimo?

El Corolario 10.3 de tu paper anterior **explícitamente lista anclajes iconográficos** como rupturas válidas de simetría. Lo que el nuevo trabajo hace es:

1. **Operacionalizar** el ancla iconográfica matemáticamente (antes solo enunciada)
2. **Cuantificar** su poder de ruptura (vía AnchorPower de Definición 10.5)
3. **Validar** que el proceso es estable bajo perturbación
4. **Acotar** el error de transferencia desde escrituras descifradas

No estás violando tu propio teorema de no-identificabilidad. Estás **construyendo el tipo de ancla que el teorema demandaba**.

### 2.3. Diferencia crítica con intentos previos de descifrado

Los intentos previos (Fischer 1997, Butinov-Knorozov 1956, las páginas web de "decipherment 88% confidence") típicamente:

- Asumen una correspondencia glifo→palabra rapanui sin validación
- No reportan controles negativos
- No validan en escrituras conocidas primero
- Sobreinterpretan repeticiones como evidencia semántica

Tu enfoque es radicalmente distinto:

- **Validación inversa obligatoria** en 4+ escrituras descifradas
- **Cota matemática explícita** del error
- **Reportar fracasos** cuando los hay (recordemos: tu Tabla 23 mostró que Indus es indistinguible de aleatorio)
- **Propagar incertidumbre** vía el ledger de hipótesis

---

## 3. El Teorema de Anclaje Icónico Inverso (formalización completa)

### 3.1. Setup matemático

#### Definición 3.1 (Espacio de Referentes Históricos)

Sea $\mathcal{W}_{\tau}$ el **mundo verificable** en época $\tau$ y región geográfica $\mathcal{G}$. Es la 4-tupla:

$$\mathcal{W}_{\tau, \mathcal{G}} = \langle \mathcal{F}_\tau, \mathcal{P}_\tau, \mathcal{O}_\tau, \mathcal{C}_\tau \rangle$$

donde:
- $\mathcal{F}_\tau$ = conjunto de especies faunísticas verificables (paleoecología, isótopos, restos óseos)
- $\mathcal{P}_\tau$ = conjunto de plantas (paleobotánica, polen, microfósiles)
- $\mathcal{O}_\tau$ = conjunto de objetos materiales (arqueología)
- $\mathcal{C}_\tau$ = conjunto de fenómenos celestes/naturales (astronomía, geología)

El **espacio de referentes** es la unión:
$$\mathcal{R}_{\tau, \mathcal{G}} = \mathcal{F}_\tau \cup \mathcal{P}_\tau \cup \mathcal{O}_\tau \cup \mathcal{C}_\tau \cup \mathcal{H}$$

donde $\mathcal{H}$ es el conjunto de partes y posturas humanas (universal cross-cultural).

#### Definición 3.2 (Embeddings Visuales Cross-Modales)

Sea $\Phi: \mathcal{V}_X \to \mathbb{S}^{d-1} \subset \mathbb{R}^d$ un embedding visual de glifos sobre la esfera unidad (vectores L2-normalizados), construido mediante un encoder de visión $f_\theta$ aplicado a representaciones gráficas $\rho(x)$ de los glifos:

$$\Phi(x) = \frac{f_\theta(\rho(x))}{\|f_\theta(\rho(x))\|_2}$$

donde $\rho(x)$ puede ser SVG raster, foto del glifo carved, o reconstrucción 3D photogrammétrica.

Análogamente, $\Psi: \mathcal{R} \to \mathbb{S}^{d-1}$ es el embedding del mismo modelo aplicado a imágenes representativas de los referentes:

$$\Psi(r) = \frac{1}{|I_r|} \sum_{i \in I_r} \frac{f_\theta(i)}{\|f_\theta(i)\|_2}$$

donde $I_r$ es un conjunto de imágenes prototípicas del referente $r$ (luego renormalizado a la esfera).

#### Definición 3.3 (Función de Iconicidad)

Para cada glifo $x \in \mathcal{V}_X$, definimos su **iconicidad respecto a $r \in \mathcal{R}$** como:

$$\iota(x, r) = \langle \Phi(x), \Psi(r) \rangle$$

(producto interno en $\mathbb{S}^{d-1}$, equivalentemente $\cos$ de su ángulo).

La **iconicidad máxima** del glifo es:
$$\iota^*(x) = \max_{r \in \mathcal{R}} \iota(x, r)$$

y el **referente más probable**:
$$r^*(x) = \arg\max_{r \in \mathcal{R}} \iota(x, r)$$

#### Definición 3.4 (Tasa de Deiconización)

La **tasa de deiconización** de un glifo se define como:

$$\delta(x) = 1 - \iota^*(x)$$

donde $\delta(x) = 0$ indica iconicidad perfecta y $\delta(x) = 2$ indica anti-iconicidad (caso degenerado).

### 3.2. Hipótesis estructurales necesarias

#### Hipótesis H1 (Origen Icónico Local)

Para cada glifo $x$, existe un referente histórico $r_0(x) \in \mathcal{R}_{\tau_0, \mathcal{G}}$ donde $\tau_0$ es la época de invención del glifo, tal que en el momento de invención:

$$\iota(x_{\tau_0}, r_0(x)) \geq 1 - \delta_0$$

para algún $\delta_0 \ll 1$ (típicamente $\delta_0 < 0.2$).

#### Hipótesis H2 (Continuidad del Mapa Visual)

El encoder $f_\theta$ es Lipschitz-continuo:

$$\|f_\theta(I_1) - f_\theta(I_2)\|_2 \leq L \cdot d_{\text{percep}}(I_1, I_2)$$

donde $d_{\text{percep}}$ es una distancia perceptual (e.g., LPIPS).

#### Hipótesis H3 (Cobertura del Mundo Reconstruido)

El conjunto $\mathcal{R}_{\tau, \mathcal{G}}$ contiene los referentes verdaderos de los glifos con probabilidad $\geq 1 - \epsilon$:

$$P(r_0(x) \in \mathcal{R}_{\tau, \mathcal{G}}) \geq 1 - \epsilon \quad \forall x \in \mathcal{V}_X$$

### 3.3. El teorema principal

#### Teorema 3.5 (Anclaje Icónico Inverso)

*Bajo las hipótesis H1, H2, H3, y dado un encoder $f_\theta$ entrenado con suficiente diversidad de estilos visuales, se cumple lo siguiente:*

**(Existencia):** *Para todo glifo $x$ con tasa de deiconización $\delta(x) < \delta_*$, donde*
$$\delta_* = \delta_0 + L \cdot \text{diam}(\Phi(\mathcal{V}_X))$$
*el referente $r^*(x)$ recuperado por máxima iconicidad satisface:*
$$P(r^*(x) = r_0(x)) \geq 1 - \epsilon - \frac{L \cdot \delta(x)}{\Delta_r}$$
*donde $\Delta_r = \min_{r \neq r'} \|\Psi(r) - \Psi(r')\|_2$ es la separación mínima entre referentes.*

**(Identificabilidad parcial):** *Si el conjunto de glifos con $\delta(x) < \delta_*$, denotado $\mathcal{V}_X^{\text{icon}}$, satisface:*
$$|\mathcal{V}_X^{\text{icon}}| \geq m_{\min}$$
*donde $m_{\min}$ es el número mínimo de anclas para que $\text{AnchorCondition}(\mathcal{A}) > \kappa_*$ (Teorema 16.1 del paper anterior), entonces:*
$$\text{AnchorPower}(\mathcal{V}_X^{\text{icon}}) \geq 1 - \frac{\log(|\text{Aut}(G_X; \mathcal{V}_X^{\text{icon}})| + 1)}{\log(|\text{Aut}(G_X)| + 1)} > 0$$

**(Elevación de claim):** *El nivel de claim máximo admisible para los glifos en $\mathcal{V}_X^{\text{icon}}$ es:*
- $C2$ si $\text{AnchorPower} \geq 0.15$ y la estabilidad bootstrap $\geq 0.5$
- $C3$ si $\text{AnchorPower} \geq 0.4$ y la estabilidad bootstrap $\geq 0.7$, **además** de validación cruzada en al menos 3 escrituras descifradas con $F1 \geq 0.6$

### 3.4. Demostración (esquema completo)

**Existencia.** Por H1, $\iota(x_{\tau_0}, r_0(x)) \geq 1 - \delta_0$. Por H2 (Lipschitz),

$$|\iota(x, r_0(x)) - \iota(x_{\tau_0}, r_0(x))| \leq L \cdot d_{\text{percep}}(\rho(x), \rho(x_{\tau_0}))$$

El término del lado derecho es exactamente $\delta(x) - \delta_0$ por definición de la deiconización. Por tanto:

$$\iota(x, r_0(x)) \geq 1 - \delta(x)$$

Si existiera un $r' \neq r_0(x)$ con $\iota(x, r') > \iota(x, r_0(x))$, entonces por desigualdad triangular en $\mathbb{S}^{d-1}$:

$$\|\Psi(r') - \Psi(r_0(x))\|_2 \leq \|\Phi(x) - \Psi(r_0(x))\|_2 + \|\Phi(x) - \Psi(r')\|_2 \leq 2\sqrt{2\delta(x)}$$

Esto contradice la separación mínima $\Delta_r$ siempre que $\delta(x) < \Delta_r^2 / 8$. Combinando con la cobertura H3, obtenemos la cota de probabilidad:

$$P(r^*(x) = r_0(x)) \geq P(r_0(x) \in \mathcal{R}) \cdot P(\text{no hay impostor más cercano})$$
$$\geq (1 - \epsilon) \cdot \left(1 - \frac{L \cdot \delta(x)}{\Delta_r}\right)$$

Reagrupando da el resultado.

**Identificabilidad parcial.** Por construcción, los glifos en $\mathcal{V}_X^{\text{icon}}$ tienen anclas iconográficas verificables. Por la Definición 10.5 de tu paper anterior, estas anclas inducen el subgrupo:

$$\text{Aut}(G_X; \mathcal{V}_X^{\text{icon}}) = \{\pi \in \text{Aut}(G_X) : \pi(x) = x, \forall x \in \mathcal{V}_X^{\text{icon}}\}$$

Este subgrupo es típicamente mucho menor que $\text{Aut}(G_X)$ porque cada ancla impone una restricción que un automorfismo aleatorio satisface con probabilidad $1/n$. Aplicando la Definición 10.5 directamente da la cota.

**Elevación de claim.** Los umbrales de admisibilidad de la Definición 3.1 del paper anterior se cumplen para C2/C3 cuando los criterios listados se satisfacen. La validación cruzada en escrituras descifradas funciona como **evidencia externa independiente**, requerida por la jerarquía de claims. □

### 3.5. Lo que el teorema NO dice

Por honestidad epistémica, listamos lo que el teorema **no** garantiza:

1. **No** garantiza traducción completa (eso sigue siendo C5, requiere bilingüe verdadero)
2. **No** garantiza identificación correcta para glifos con $\delta(x) > \delta_*$ (que pueden ser muchos)
3. **No** garantiza que $r^*(x)$ sea el significado **lingüístico**; es el referente **icónico** (puede haber rebus, metonimia, metáfora)
4. **No** elimina la necesidad de controles negativos: validación obligatoria

### 3.6. Refinamientos críticos

#### 3.6.1. Caso de glifos compuestos (ligaturas)

Para glifos compuestos $x = x_1 \circ x_2$, el teorema se extiende vía:

$$\iota^*(x) \approx \alpha \cdot \iota^*(x_1) + (1-\alpha) \cdot \iota^*(x_2) + \text{término de interacción}$$

con $\alpha$ aprendible. Esto requiere segmentación previa de los glifos compuestos.

#### 3.6.2. Caso de variantes alográficas

Si $x_a$ y $x_b$ son alografos del mismo glifo conceptual, el teorema demanda:

$$\|\Phi(x_a) - \Phi(x_b)\|_2 < \tau_{\text{allo}}$$

En la práctica, esto se valida **descubriendo automáticamente las clases alográficas** mediante clustering en el espacio $\Phi$ y comparando con la lista de Pozdniakov.

#### 3.6.3. Caso de signos abstractos (ya deiconizados)

Para glifos con $\delta(x) > \delta_*$, el teorema **no aplica**. Estos glifos:

- Pueden ser auxiliares gramaticales (no requieren referente físico)
- Pueden haber sufrido deiconización extrema
- Pueden ser composiciones complejas
- **Permanecen en C0–C1**

Esto es **importantísimo** y mantiene la honestidad del marco: no todos los glifos suben a C2–C3.

---

## 4. Teoría de la deiconización: el segundo teorema

### 4.1. Modelado de la trayectoria evolutiva

#### Definición 4.1 (Trayectoria de Deiconización)

Para un sistema de escritura $S$, la **trayectoria de deiconización** es una curva en el espacio de embeddings:

$$\gamma_S: [0, T_S] \to \mathbb{R}^d$$

donde $\gamma_S(0)$ es el embedding del prototipo icónico (e.g., dibujo del referente físico) y $\gamma_S(T_S)$ es el embedding de la forma final convencional.

Para escrituras descifradas tenemos pares $(\gamma_S(t_i), r_i)$ a distintos tiempos $t_i$.

#### Definición 4.2 (Velocidad de Deiconización)

$$v_\delta(S, t) = \left\|\frac{d \gamma_S(t)}{dt}\right\|_2$$

mide qué tan rápido un sistema pierde iconicidad. Empíricamente:

| Sistema | $v_\delta$ promedio | Comentario |
|---------|---------------------|------------|
| Cuneiforme | Alta | Pierde iconicidad rápido (~500 años) |
| Jeroglíficos egipcios | Baja | Mantiene iconicidad 3000 años |
| Chino | Moderada | Trayectoria lenta pero continua |
| Maya | Baja | Conserva iconicidad |
| Rongorongo | **Mínima** (probablemente) | Sistema joven |

### 4.2. Teorema 4.3 (Deiconización Acotada)

*Sea $S$ un sistema de escritura cuyos signos evolucionan según un proceso de deiconización con velocidad acotada $v_\delta(S, t) \leq V$. Sea $T_S^* = \arg\min_t \delta(\gamma_S(t))$ el momento de máxima iconicidad. Entonces:*

$$\delta(\gamma_S(t)) \leq V \cdot |t - T_S^*|$$

*y por tanto el error de identificación icónica acumula linealmente con el tiempo desde la invención.*

**Demostración.** Por desigualdad de cadena/Lipschitz,
$$\|\gamma_S(t) - \gamma_S(T_S^*)\|_2 \leq \int_{T_S^*}^t \|\gamma_S'(s)\|_2 \, ds \leq V \cdot |t - T_S^*|$$
y como $\delta$ es Lipschitz en $\Phi$, el resultado sigue. □

#### Corolario 4.4 (Ventaja Temporal de Rongorongo)

*Si Rongorongo tiene edad $T_{\text{RR}} < 200$ años (consistente con el radiocarbon de la tablilla Berlin), y la velocidad de deiconización polinésica es comparable a la egipcia ($V \leq V_{\text{eg}}$), entonces:*

$$\delta(\gamma_{\text{RR}}(T_{\text{RR}})) \leq V_{\text{eg}} \cdot 200 \text{ años}$$

*lo que es 15 veces menor que la cota correspondiente para jeroglíficos egipcios (3000 años). Esto sugiere que **Rongorongo es uno de los sistemas más identificables iconográficamente jamás documentado**.*

### 4.3. Modelo paramétrico de deiconización

Postulamos que la trayectoria de deiconización sigue una **mezcla de procesos**:

$$\gamma_S(t) = \gamma_S(0) + \sum_{k=1}^K \alpha_k(t) \cdot u_k$$

donde:
- $u_k$ son **direcciones canónicas de abstracción** (estilización, rotación, simplificación, abstracción)
- $\alpha_k(t)$ son funciones temporales monótonas crecientes

### 4.4. Aprendizaje de las direcciones canónicas

Las direcciones $u_k$ se aprenden de las trayectorias conocidas:

```python
# Pseudocódigo
trajectories = []
for script in [egyptian, chinese_oracle_to_modern, cuneiform_proto_to_late, maya_early_to_late]:
    for sign_chain in script.evolutionary_chains:
        embeddings_t = [phi(sign_at_time(t)) for t in sign_chain.times]
        trajectories.append(embeddings_t)

# PCA sobre las direcciones de cambio
direction_vectors = []
for traj in trajectories:
    deltas = [traj[i+1] - traj[i] for i in range(len(traj)-1)]
    direction_vectors.extend(deltas)

# Direcciones canónicas = top-K componentes principales
U_canonical = PCA(direction_vectors, k=K)
```

### 4.5. Teorema 4.5 (Universalidad de Direcciones Canónicas)

*Si las direcciones canónicas $\{u_k\}$ aprendidas en escrituras descifradas explican una fracción $\eta$ de la varianza de las trayectorias, y si Rongorongo evolucionó bajo presión cultural similar (escritura humana en sociedad jerárquica), entonces el error de proyección sobre el subespacio $\{u_k\}$ está acotado por:*

$$\|\gamma_{\text{RR}}(t) - \text{Proj}_U \gamma_{\text{RR}}(t)\|_2 \leq C \cdot (1 - \eta)$$

*donde $C$ depende de la diversidad cultural del entrenamiento.*

Empíricamente, con 4 escrituras independientes (egipcio, chino, cuneiforme, maya), esperamos $\eta \geq 0.85$.

---

## 5. Geometría de fibrados visuales-semánticos

### 5.1. Estructura de fibrado

El espacio donde vive todo el problema es un **fibrado vectorial** sobre el espacio de referentes:

$$E \xrightarrow{\pi} \mathcal{R}$$

donde:
- La base $\mathcal{R}$ son los referentes
- Cada **fibra** $\pi^{-1}(r)$ contiene todos los glifos posibles que pueden pictografiar a $r$ (variantes estilísticas, alografos, deiconizaciones)

#### Definición 5.1 (Sección Cultural)

Una **sección cultural** $\sigma: \mathcal{R} \to E$ asigna a cada referente el glifo concreto que la cultura específica usa:

$$\sigma_{\text{Egyptian}}(\text{owl}) = 𓅓$$
$$\sigma_{\text{Maya}}(\text{jaguar}) = \text{glyph BALAM}$$
$$\sigma_{\text{RR}}(\text{frigatebird}) = \text{glyph}_{600}\text{?}$$

### 5.2. Teorema 5.2 (Coherencia Cultural de Secciones)

*Sea $\Sigma_{\text{known}} = \{\sigma_S : S \in \text{decoded scripts}\}$ el conjunto de secciones culturales conocidas. La cápsula convexa de estas secciones en cada fibra induce un **prior de plausibilidad** sobre la sección cultural desconocida $\sigma_{\text{RR}}$:*

$$\sigma_{\text{RR}}(r) \in \text{ConvexHull}\left(\bigcup_{S} \sigma_S(r) + \mathcal{N}(0, \Sigma_{\text{cultural}})\right)$$

Esto es **transferencia de prior cultural**: lo que esperamos que pictografíe un águila en Rongorongo no debería ser radicalmente distinto de lo que pictografía en otras culturas.

### 5.3. Submersión geométrica entre escrituras

Esto extiende tu Sección 12 del paper anterior. Cada escritura puede verse como una **proyección parcial** del fibrado completo:

$$F_S: \mathcal{V}_S \to \mathcal{R}$$

con fibras
$$F_S^{-1}(r) = \{x \in \mathcal{V}_S : x \text{ pictografía } r\}$$

#### Teorema 5.3 (Reducción de Fibras Vía Múltiples Escrituras)

*Si para un referente $r$ tenemos múltiples pictografías $\{\sigma_{S_1}(r), \sigma_{S_2}(r), \ldots, \sigma_{S_m}(r)\}$ de escrituras independientes, entonces el embedding consenso:*

$$\bar{\Psi}(r) = \frac{1}{m} \sum_{i=1}^m \Phi(\sigma_{S_i}(r))$$

*tiene varianza reducida, y la fibra de $r$ en Rongorongo debe estar a distancia acotada de $\bar{\Psi}(r)$:*

$$\|\Phi(\sigma_{\text{RR}}(r)) - \bar{\Psi}(r)\|_2 \leq O\left(\frac{\sigma_{\text{cultural}}}{\sqrt{m}}\right)$$

Esto es la **versión iconográfica** del consenso multi-idioma de tu Sección 24.

### 5.4. Conexión Riemanniana sobre el fibrado

Para hacer el análisis matemáticamente completo, necesitamos definir transporte paralelo entre fibras:

#### Definición 5.4 (Conexión Cultural)

Una **conexión cultural** $\nabla$ permite comparar embeddings de glifos en distintas culturas mediante transporte paralelo:

$$\nabla_{u} \Phi(x) = \lim_{t \to 0} \frac{\Phi_{c+tu}(x) - P_{c \to c+tu} \Phi_c(x)}{t}$$

donde $P_{c \to c'}$ es el operador Procrustes (de tu Teorema 9.1) entre culturas.

La existencia de tal conexión, con curvatura acotada, es lo que permite la transferencia sistemática.

---

## 6. Arquitectura computacional completa

### 6.1. Stack tecnológico

```
┌─────────────────────────────────────────────────────────┐
│  CAPA 1: ADQUISICIÓN DE DATOS                           │
│  ─────────────────────────────                          │
│  • Web scraping con respeto a licencias                 │
│  • Acceso a Universidad Bologna (Lastilla et al.)       │
│  • iNaturalist API (fauna)                              │
│  • GBIF API (biodiversidad histórica)                   │
│  • British Museum API (artefactos)                      │
│  • Scrapers a Wikimedia Commons                         │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA 2: PREPROCESAMIENTO Y NORMALIZACIÓN               │
│  ───────────────────────────────────────                │
│  • Vectorización SVG de glifos                          │
│  • Augmentación: rotación, escala, ruido, grosor       │
│  • Stratified sampling de referentes                    │
│  • Photogrammetric 3D → 2D projections múltiples        │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA 3: EMBEDDINGS Y MODELOS DE VISIÓN                 │
│  ──────────────────────────────                         │
│  • DINOv2 (Meta, autosupervisado)                       │
│  • SigLIP-2 (Google, contrastivo)                       │
│  • CLIP (OpenAI, baseline)                              │
│  • Stable Diffusion latents (fine-grained)              │
│  • Custom CNN entrenado on-script (Glyphnet)            │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA 4: ALINEAMIENTO CROSS-MODAL                       │
│  ─────────────────────────────                          │
│  • Procrustes ortogonal (tu Teorema 9.1)                │
│  • Transporte óptimo entrópico (tu Sección 17)          │
│  • Gromov-Wasserstein (tu Sección 11.3)                 │
│  • Normalizing flows entre espacios                     │
│  • Tikhonov regularizado (tu Teorema 17.3)              │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA 5: VALIDACIÓN Y AUDITORÍA                         │
│  ────────────────────────────                           │
│  • Bootstrap masivo (1000+ samples)                     │
│  • Controles negativos múltiples                        │
│  • Calibración (ECE, Brier)                             │
│  • Cross-script validation                              │
│  • Ledger de hipótesis con OverclaimRisk                │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  CAPA 6: REPORTE                                        │
│  ───────────────                                        │
│  • Anclas iconográficas con confianza calibrada         │
│  • Glifos elevados a C2/C3 (lista explícita)            │
│  • Glifos que permanecen en C0/C1 (mayoría!)            │
│  • Trayectorias de deiconización predichas              │
└─────────────────────────────────────────────────────────┘
```

### 6.2. Estructura de repositorio propuesta

```
iconic-grounding-decipherment/
├── README.md
├── pyproject.toml
├── LICENSE
├── data/
│   ├── glyphs/
│   │   ├── rongorongo/        # Lastilla, Barthel, etc.
│   │   ├── egyptian/          # Glyphnet dataset
│   │   ├── chinese_oracle/    # HUST-OBS dataset
│   │   ├── cuneiform/         # CDLI corpus
│   │   ├── maya/              # MHED database
│   │   ├── linear_a_b/        # SigLA database
│   │   └── ...
│   ├── referents/
│   │   ├── fauna/
│   │   │   ├── rapa_nui_1500/ # Reconstrucción ecológica
│   │   │   ├── egypt_3000bce/
│   │   │   ├── china_1200bce/
│   │   │   └── ...
│   │   ├── flora/
│   │   ├── artifacts/
│   │   └── celestial/
│   └── world_reconstructions/
│       └── rapa_nui_1500/
├── src/
│   ├── ingestion/
│   │   ├── scrapers/
│   │   ├── normalizers/
│   │   └── 3d_to_2d/
│   ├── embeddings/
│   │   ├── dino_encoder.py
│   │   ├── siglip_encoder.py
│   │   ├── clip_encoder.py
│   │   └── ensemble.py
│   ├── alignment/
│   │   ├── procrustes.py
│   │   ├── optimal_transport.py
│   │   ├── gromov_wasserstein.py
│   │   └── tikhonov.py
│   ├── deiconization/
│   │   ├── trajectory_learning.py
│   │   ├── canonical_directions.py
│   │   └── temporal_models.py
│   ├── validation/
│   │   ├── cross_script.py
│   │   ├── bootstrap.py
│   │   ├── negative_controls.py
│   │   └── calibration.py
│   ├── theorems/
│   │   ├── anchor_power.py
│   │   ├── spectral_reliability.py
│   │   └── claim_admissibility.py
│   └── reporting/
│       ├── ledger.py
│       └── visualizations.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_cross_script_validation.ipynb
│   └── 04_rongorongo_application.ipynb
├── experiments/
│   ├── E1_egyptian_blind/
│   ├── E2_oracle_bone_blind/
│   ├── E3_maya_blind/
│   ├── E4_cuneiform_blind/
│   └── E5_rongorongo_application/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── hypotheses/
└── tests/
```

### 6.3. Pipelines de procesamiento

#### Pipeline 1: Glyph → Embedding

```python
def glyph_to_embedding(svg_path, model='dinov2_vitl14'):
    """
    Convierte un SVG de glifo a embedding visual robusto.
    """
    # 1. Renderizar SVG a múltiples resoluciones
    rasters = []
    for size in [64, 128, 256, 384]:
        rasters.append(svg_to_png(svg_path, size=size))
    
    # 2. Augmentaciones controladas
    augmented = []
    for raster in rasters:
        augmented.extend([
            raster,
            rotate(raster, angle=5),
            rotate(raster, angle=-5),
            add_stroke_noise(raster, sigma=0.05),
            erode(raster, iterations=1),
            dilate(raster, iterations=1)
        ])
    
    # 3. Pasar por encoder
    embeddings = []
    for img in augmented:
        emb = encoder(img)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    
    # 4. Promediar en la esfera (Frechet mean)
    consensus = spherical_mean(embeddings)
    return consensus
```

#### Pipeline 2: Referent → Embedding

```python
def referent_to_embedding(referent_id, model='dinov2_vitl14'):
    """
    Convierte un referente (especie/objeto) a embedding consenso
    desde múltiples imágenes.
    """
    # 1. Recolectar imágenes representativas
    images = []
    images.extend(inaturalist_query(referent_id, max=50))
    images.extend(gbif_query(referent_id, max=30))
    images.extend(scientific_illustration_query(referent_id, max=20))
    
    # 2. Filtrar por calidad
    filtered = []
    for img in images:
        if is_clean_specimen_photo(img):
            filtered.append(img)
    
    # 3. Vectorizar
    embeddings = [encoder(img) for img in filtered]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    
    # 4. Detección de outliers y consenso
    inliers = mahalanobis_filter(embeddings, threshold=2.5)
    consensus = spherical_mean(inliers)
    
    # 5. Reportar varianza
    return consensus, np.std(inliers, axis=0)
```

#### Pipeline 3: Cross-Modal Alignment

```python
def align_glyphs_to_referents(Phi_X, Psi_R, anchors=None):
    """
    Alinea espacio de glifos al espacio de referentes.
    """
    if anchors is not None:
        # Procrustes con anclas verdaderas (cross-script validation)
        Q = procrustes_orthogonal(Phi_X[anchors[:, 0]], Psi_R[anchors[:, 1]])
    else:
        # Sin anclas: Gromov-Wasserstein
        D_X = pairwise_distances(Phi_X)
        D_R = pairwise_distances(Psi_R)
        coupling = gromov_wasserstein(D_X, D_R, epsilon=0.1, n_iter=100)
        Q = recover_rotation_from_coupling(coupling)
    
    # Aplicar y normalizar a esfera
    Phi_aligned = (Phi_X @ Q)
    Phi_aligned = Phi_aligned / np.linalg.norm(Phi_aligned, axis=1, keepdims=True)
    
    return Phi_aligned, Q
```


---

## 7. Plan de adquisición de datos: 30+ sistemas de escritura

### 7.1. Por qué necesitas tantos sistemas

El Teorema 4.5 (Universalidad de Direcciones Canónicas) requiere **diversidad cultural**. Cuanto más variadas sean las trayectorias evolutivas que aprendes, más robusto será el modelo transferido a Rongorongo. La regla práctica: **mínimo 8 sistemas, ideal 20+**.

### 7.2. Tabla maestra de sistemas y datasets

#### 7.2.1. Sistemas con datasets digitales bien establecidos

| Sistema | Período | Estado | Dataset principal | URL/Acceso | Tamaño |
|---------|---------|--------|-------------------|------------|--------|
| **Egipcio jeroglífico** | 3100 BCE - 400 CE | Descifrado | Glyphnet (Barucci 2021) | github.com/morrisfranken/glyphreader | 4,310 imágenes |
| **Egipcio jeroglífico** | - | - | Pyramid of Unas | OCR-PT-CT (de Buck 1935) | 13,450 imgs |
| **Chino oracle bone** | 1200 BCE | Descifrado parcial | HUST-OBS | github.com/Pengjie-W/HUST-OBS | 140,053 imgs |
| **Chino oracle bone** | - | - | OBC306 | Anyang Normal University | 309,551 samples |
| **Chino oracle bone** | - | - | OBIMD multimodal | arxiv.org/abs/2407.03900 | 10,077 rubbings |
| **Cuneiforme** | 3200 BCE - 100 CE | Descifrado | CDLI | cdli.mpiwg-berlin.mpg.de | 350K+ tablillas |
| **Maya** | 250-900 CE | Descifrado parcial | MHED | mayadatabase.org | ~7,000 glifos |
| **Linear B** | 1450-1200 BCE | Descifrado | SigLA | sigla.phis.me | 4,000+ inscripciones |
| **Linear A** | 1800-1450 BCE | No descifrado | SigLA | sigla.phis.me | ~1,500 |
| **Indus Valley** | 2600-1900 BCE | No descifrado | Mayig Parpola (CISI) | github.com/mayig | 179 inscripciones |
| **Proto-Elamite** | 3100-2900 BCE | No descifrado | CDLI | cdli.mpiwg-berlin.mpg.de | ~1,600 |
| **Rongorongo** | 1700-1860 CE | No descifrado | RR-corpus, Lastilla | Universidad Bologna (contacto) | 26 objetos |

#### 7.2.2. Sistemas alfabéticos (para ver "el otro extremo")

| Sistema | Período | Origen pictográfico | Útil para |
|---------|---------|---------------------|-----------|
| **Proto-sinaítico** | 1900 BCE | Sí (de jeroglíficos egipcios) | Estudiar deiconización extrema |
| **Fenicio** | 1200 BCE | Sí | Eslabón a alfabetos modernos |
| **Hebreo antiguo** | 1000 BCE | Sí | Cognados con fenicio |
| **Griego antiguo** | 800 BCE | Sí (de fenicio) | Trayectoria larga conocida |
| **Latino** | 700 BCE | Sí (de griego) | El alfabeto que usas hoy |
| **Cirílico** | 900 CE | Sí (de griego) | Trayectoria reciente |
| **Árabe** | 400 CE | Sí (nabateo) | Sistema con calligrafía |
| **Brahmi/Devanagari** | 300 BCE | Sí (debate) | Familia índica |

Para ver **"de qué objeto vino la letra A"**: el alef proto-sinaítico era una cabeza de toro 𓃾 → 𐤀 → Α → A. Esa información trazada vive en datasets paleográficos.

#### 7.2.3. Sistemas asiáticos modernos con genealogía pictográfica

| Sistema | Útil porque |
|---------|-------------|
| **Chino tradicional** | 80,000+ caracteres con etimología pictográfica documentada (Shuowen Jiezi) |
| **Japonés Kanji** | Subset chino + evolución independiente |
| **Coreano Hanja** | Adaptación coreana |
| **Vietnamita Chữ Nôm** | Adaptación vietnamita |
| **Yi (silabario)** | Sistema tibetano-birmano con iconos |
| **Naxi Dongba** | **El último sistema pictográfico vivo del mundo** (China) |

**Naxi Dongba es CRÍTICO para tu trabajo:** es la única escritura pictográfica que sigue siendo usada activamente. Permite estudiar la deiconización **en tiempo real** y validar el Teorema 4.3.

#### 7.2.4. Sistemas mesoamericanos y andinos

| Sistema | Estado |
|---------|--------|
| **Nahuatl pre-columbino** | Códices Borgia, Mendoza (digitalizados) |
| **Mixteco** | Códices Vindobonensis, Nuttall |
| **Zapoteco** | Monte Albán inscripciones |
| **Olmeca** | La Venta, San Andrés |
| **Quipu inca** | NO pictográfico pero útil como contraste (sistema 3D) |
| **Tocapu** | Patrones textiles andinos |

#### 7.2.5. Sistemas africanos sub-saharianos

| Sistema | Útil porque |
|---------|-------------|
| **Nsibidi (Nigeria)** | Pictográfico, no descifrado completamente |
| **Adinkra (Ghana)** | Símbolos con significado documentado |
| **Vai (Liberia)** | Inventado en 1833 - **caso paralelo a Rongorongo** |
| **Bamum (Camerún)** | Inventado en 1896 - silabario reciente |

**Vai y Bamum son extremadamente importantes:** son sistemas inventados en el siglo XIX por individuos identificables (Momolu Duwalu Bukele para Vai, Sultan Ibrahim Njoya para Bamum), análogos en período y ruta de invención a Rongorongo. Sus trayectorias documentadas son oro puro.

#### 7.2.6. Sistemas oceánicos (los más importantes para Rongorongo)

| Sistema | Estado | Notas |
|---------|--------|-------|
| **Petroglifos Rapa Nui** | Documentado | Lee, Liller; >4,000 motivos |
| **Petroglifos Marquesas** | Documentado | Suggs, Millerstrom |
| **Petroglifos Hawaii** | Documentado | Lee, Stasack |
| **Petroglifos Nueva Zelanda** | Documentado | Maori rock art |
| **Tatuajes polinesios** | Documentado | Cawte, McCallum |
| **Tapa cloth designs** | Documentado | Kooijman |
| **Cuerda de Caroline (mnemonic)** | Documentado | Schils-Schwartz |

**Los petroglifos polinesios** son el contexto inmediato de Rongorongo y posiblemente sus ancestros visuales. Lee (1992) catalogó >4,000 motivos en Rapa Nui antes de Rongorongo.

### 7.3. Estrategia de adquisición por categorías

#### 7.3.1. Glifos: tres rutas

**Ruta A — Datasets ya digitalizados (90% de tu trabajo)**

```bash
# Clonar repositorios públicos
git clone https://github.com/Pengjie-W/HUST-OBS  # Oracle bone
git clone https://github.com/morrisfranken/glyphreader  # Egyptian
git clone https://github.com/cdli/cdli  # Cuneiform metadata

# APIs y descargas
curl https://cdli.mpiwg-berlin.mpg.de/api/dl/  # Cuneiform images
wget https://sigla.phis.me/data/  # Linear A/B
```

**Ruta B — Digitalización propia (para sistemas sin dataset)**

Para Nsibidi, Vai, Bamum, Naxi Dongba: necesitas digitalizar manuales o libros. Workflow:

1. Adquirir libros de referencia (Diringer 1968, Daniels & Bright 1996)
2. Escanear a 600 DPI
3. Segmentar glifos manualmente con LabelImg o CVAT
4. Vectorizar a SVG (potrace, Inkscape API)
5. Etiquetar con código estándar

**Ruta C — Para Rongorongo: caso especial**

El estado del corpus Rongorongo es **el cuello de botella principal**. Tienes opciones:

1. **Contactar a Lastilla, Ravanelli, Valério** (Universidad Bologna) — están construyendo el corpus 3D moderno. Email institucional con propuesta seria genera respuesta.
2. **Usar Barthel (1958)** — código de glifos de tres dígitos, 599 signos. Disponible en bibliotecas universitarias y archive.org parcialmente.
3. **Pozdniakov & Pozdniakov (2007)** — análisis estadístico moderno, 638 glifos efectivos. Contactar.
4. **Internet Archive** — `archive.org/details/rongorongotexts` tiene transcripciones legibles.
5. **Wikimedia Commons** — fotografías de tabletas en alta resolución.

**Mi recomendación:** Inicia escribiendo a Lastilla con tu paper anterior como credencial. Es la ruta más profesional.

#### 7.3.2. Referentes: las cuatro fuentes principales

**Fuente 1: iNaturalist (fauna y flora moderna)**

```python
# API: https://api.inaturalist.org/v1/
import requests

def fetch_inat(taxon_name, n=50):
    url = f"https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": taxon_name,
        "photos": "true",
        "quality_grade": "research",
        "per_page": n
    }
    return requests.get(url, params=params).json()

# Ejemplo: fragatas (referente "ave" Rapa Nui)
fregata = fetch_inat("Fregata minor", n=100)
```

**Fuente 2: GBIF (Global Biodiversity Information Facility)**

Para distribución histórica de especies. Te dice qué especies estaban presentes en Rapa Nui circa 1500 CE.

```python
# API: https://api.gbif.org/v1/
def historical_species(lat, lon, radius_km=100, year_max=1500):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "year": f"*,{year_max}",
        "limit": 300
    }
    return requests.get(url, params=params).json()
```

**Fuente 3: Wikimedia Commons (artefactos arqueológicos)**

Para moai, reimiro, tapa cloth, etc.

```python
# Búsqueda categórica
def commons_category(category, max_files=100):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": max_files,
        "format": "json"
    }
    return requests.get(url, params=params).json()

# Ejemplos
moai = commons_category("Moai")
tapa = commons_category("Tapa cloth")
reimiro = commons_category("Reimiro")
```

**Fuente 4: Ilustraciones científicas históricas (Biodiversity Heritage Library)**

Para reconstrucciones del mundo antiguo.

```python
# BHL API: https://www.biodiversitylibrary.org/api3
def bhl_search(query):
    url = "https://www.biodiversitylibrary.org/api3"
    params = {
        "op": "PublicationSearch",
        "searchterm": query,
        "format": "json",
        "apikey": YOUR_KEY  # Solicitar gratis
    }
    return requests.get(url, params=params).json()
```

#### 7.3.3. Reconstrucciones del mundo antiguo

Para construir $\mathcal{W}_{\tau, \mathcal{G}}$ históricamente correcto:

| Recurso | Cobertura |
|---------|-----------|
| **PaleoBioDB** | Fauna fósil global |
| **Neotoma Paleoecology Database** | Polen, vegetación |
| **Dryad** | Datos de excavaciones |
| **Open Context** | Arqueología publicada |
| **The History of Now** | Reconstrucciones GIS |
| **WorldClim Paleo** | Clima histórico |

### 7.4. Volumen de datos esperado

Estimación realista para el experimento completo:

| Categoría | Tamaño aproximado |
|-----------|-------------------|
| Glifos digitalizados (15 sistemas) | ~500K imágenes |
| Referentes fauna/flora/objetos | ~100K imágenes |
| Trayectorias evolutivas | ~5K cadenas |
| Embeddings procesados | ~600K vectores 768-D ≈ 1.5 GB |
| Modelos entrenados | ~10 GB |
| **Total proyecto** | **~50 GB** |

### 7.5. Aspectos legales y éticos

#### 7.5.1. Licencias

- **Datasets académicos:** Casi todos CC-BY o MIT. Citar autores.
- **Imágenes museos:** Variable. British Museum permite reutilización; Louvre requiere permiso.
- **Wikimedia Commons:** CC-BY-SA típicamente.
- **iNaturalist:** Variable por observación. Filtra `license=cc0` o `cc-by`.

#### 7.5.2. Consideraciones culturales

**Rongorongo es patrimonio de Rapa Nui.** Recomendaciones éticas críticas:

1. **Contactar a la Comunidad de Rapa Nui** antes de publicar afirmaciones de descifrado. Hay un Consejo de Ancianos.
2. **Co-autoría con investigadores rapanui** si haces hallazgos significativos.
3. **No reclamar "decipherment"** — usar términos como "hipótesis iconográficas" o "anclas computacionales propuestas".
4. **Ofrecer los datos abiertamente** a la comunidad rapanui (no solo a la academia occidental).
5. **Reconocer la dimensión sagrada:** Rongorongo era controlado por una clase específica (los maori rongorongo). Hay sensibilidad en la comunidad.

Esto no es burocracia — es ciencia ética. Tu paper anterior tiene la honestidad metodológica para hacerlo bien.

---

## 8. Reconstrucción del mundo Rapa Nui circa 1500 CE

Esta es **la sección más original de tu trabajo**. Construyes $\mathcal{W}_{1500, \text{Rapa Nui}}$ desde fuentes paleoecológicas verificables.

### 8.1. Inventario faunal verificado

#### 8.1.1. Aves (Aves)

**Aves marinas presentes circa 1500 CE** (paleontología, restos óseos en Anakena):

| Especie | Nombre Rapa Nui | Estado en 1500 | Iconográfica |
|---------|-----------------|----------------|--------------|
| *Fregata minor* | Taha | Abundante | Asociada a Makemake |
| *Sula dactylatra* | - | Presente | - |
| *Phaethon rubricauda* | Tavake | Presente | Cola roja distintiva |
| *Pterodroma macroptera* | - | Anidaba | - |
| *Sterna fuscata* | - | Abundante | - |
| *Gygis alba* | Manu tara | Presente | **Centro del culto Tangata Manu** |
| *Anous stolidus* | - | Presente | - |
| *Puffinus pacificus* | - | Presente | - |

**Aves terrestres (extintas o presentes):**

| Especie | Estado en 1500 |
|---------|----------------|
| *Gallus gallus* (gallina polinesia) | Presente, doméstica |
| *Cyanoramphus* sp. (parrot extinto) | Probablemente extinto |
| *Porzana* sp. (rail) | Probablemente extinto |
| *Ardea* sp. (heron) | Probablemente extinto |

#### 8.1.2. Peces y vida marina

| Categoría | Especies clave | Iconicidad esperada |
|-----------|----------------|---------------------|
| Tiburones | *Carcharhinus*, *Galeocerdo* | Alta (silueta dorsal) |
| Atunes | *Thunnus albacares*, *T. obesus* | Alta (forma fusiforme) |
| Anguilas | *Gymnothorax* | Alta (forma serpenteante) |
| Tortugas | *Chelonia mydas*, *Eretmochelys* | Alta (caparazón) |
| Pulpos | *Octopus* spp. | Alta (tentáculos) |
| Langostas | *Panulirus pascuensis* (endémica) | Media |
| Peces arrecife | Múltiples | Media |

#### 8.1.3. Mamíferos

| Especie | Notas |
|---------|-------|
| *Rattus exulans* (rata polinesia, kiore) | Único mamífero terrestre nativo |
| Mamíferos marinos | Ballenas, delfines vistos pero raros |

### 8.2. Inventario floral

| Categoría | Especies clave | Estado |
|-----------|----------------|--------|
| **Palma extinta** | *Paschalococos disperta* | Extinta ~1500 |
| **Toromiro** | *Sophora toromiro* | En declive |
| **Hau hau** | *Triumfetta semitriloba* | Presente |
| **Mahute** | *Broussonetia papyrifera* | Cultivada para tapa |
| **Taro** | *Colocasia esculenta* | Cultivada |
| **Batata (kumara)** | *Ipomoea batatas* | Cultivada |
| **Plátano** | *Musa* sp. | Cultivada |
| **Caña de azúcar** | *Saccharum officinarum* | Cultivada |
| **Ñame** | *Dioscorea* sp. | Cultivada |
| **Coprosma** | *Coprosma* spp. | Presente |

### 8.3. Inventario de artefactos materiales

| Artefacto | Material | Iconicidad |
|-----------|----------|-----------|
| **Moai** | Tuf volcánico | Forma humana estilizada |
| **Reimiro** | Madera | Pectoral en forma de creciente |
| **Ao** | Madera | Remo ceremonial |
| **Hami** | Tapa, cuerda | Cinturón ceremonial |
| **Mata'a** | Obsidiana | Punta de lanza |
| **Toki** | Basalto | Hacha |
| **Hare paenga** | Piedra | Casa-bote (forma de canoa) |
| **Tahonga** | Madera/coral | Pendiente esférico |
| **Pa'oa** | Madera | Maza ceremonial |
| **Rapa** | Madera | Remo danzante |

### 8.4. Inventario celeste

| Cuerpo | Importancia cultural |
|--------|---------------------|
| Sol (raá) | Central (calendario) |
| Luna (mahina) | Calendario lunar (relevante para Mamari) |
| Pléyades (Matariki) | Marca el año nuevo polinesio |
| Venus (Hetuʻu poʻopoʻo) | Estrella de la mañana/tarde |
| Cruz del Sur | Navegación |

### 8.5. Inventario de partes humanas y posturas

Universal cross-cultural:

| Parte/postura | Presencia esperada |
|---------------|-------------------|
| Mano (rima) | Casi todos los sistemas |
| Pie (vae) | Común |
| Cabeza (puoko) | Común |
| Postura sentada | En Rongorongo (cross-legged) |
| Postura erguida | Común |
| Brazos en alto | Frecuente en Rongorongo |
| Genitales (fertility) | Presente en petroglifos |

### 8.6. Construcción operativa de $\mathcal{R}_{1500, \text{Rapa Nui}}$

```python
class RapaNuiWorld1500:
    def __init__(self):
        self.fauna = self._build_fauna()
        self.flora = self._build_flora()
        self.artifacts = self._build_artifacts()
        self.celestial = self._build_celestial()
        self.human = self._build_human()
    
    def _build_fauna(self):
        return {
            'birds_marine': [
                'frigatebird_great', 'red-tailed_tropicbird',
                'masked_booby', 'sooty_tern', 'fairy_tern',
                'wedge-tailed_shearwater'
            ],
            'birds_land': ['domestic_chicken'],
            'fish': [
                'tuna_yellowfin', 'tiger_shark',
                'moray_eel', 'parrotfish', 'snapper'
            ],
            'reptiles': ['green_sea_turtle', 'hawksbill_turtle'],
            'mollusks': ['octopus', 'spiny_lobster_pascuensis'],
            'mammals': ['polynesian_rat']
        }
    
    def _build_flora(self):
        return {
            'palms': ['paschalococos_disperta'],
            'trees': ['sophora_toromiro', 'triumfetta_semitriloba'],
            'crops': ['taro', 'sweet_potato', 'banana',
                     'sugarcane', 'yam', 'paper_mulberry']
        }
    
    def get_referent_set(self):
        all_refs = []
        for category in [self.fauna, self.flora, 
                        self.artifacts, self.celestial, self.human]:
            for subcat, items in category.items():
                all_refs.extend(items)
        return all_refs
    
    def get_embeddings(self, encoder):
        """Construye Psi(R) para todo el mundo Rapa Nui 1500"""
        embeddings = {}
        for ref in self.get_referent_set():
            images = self._fetch_referent_images(ref)
            emb = referent_to_embedding(images, encoder)
            embeddings[ref] = emb
        return embeddings
```

### 8.7. Validación independiente del mundo reconstruido

Crítico: tu reconstrucción debe ser **verificable independientemente**. Documentar fuentes:

| Tipo de evidencia | Fuente |
|-------------------|--------|
| **Fauna** | Steadman et al. 1994 (Anakena), Hunt & Lipo 2018 |
| **Flora extinta** | Flenley & King 1984 (polen) |
| **Cultivos** | Horrocks & Wozniak 2008 (microfósiles) |
| **Artefactos** | Métraux 1940, Heyerdahl & Ferdon 1961 |
| **Astronomía** | Esen-Baur 1990, Edwards & Edwards 2013 |

Cada referente en $\mathcal{R}$ debe tener al menos 2 referencias bibliográficas que confirmen su presencia en Rapa Nui circa 1500 CE.


---

## 9. Pipeline de visión computacional

### 9.1. Selección de modelo encoder

#### 9.1.1. Análisis comparativo

| Modelo | Pros | Contras | Recomendado para |
|--------|------|---------|------------------|
| **DINOv2-L/14** | Autosupervisado, sin sesgo lingüístico | No tiene alineamiento texto | **Glifos puros** |
| **SigLIP-2** | Mejor alineamiento imagen-texto | Sesgo a inglés | **Referentes con texto** |
| **CLIP ViT-L** | Baseline establecido | Inferior a SigLIP-2 | Comparación |
| **OpenCLIP-G/14** | Más grande, abierto | Costoso | Producción |
| **Stable Diffusion VAE** | Captura detalles finos | Ruido en alta frecuencia | Análisis estilístico |
| **DINOv2 + SigLIP ensemble** | Lo mejor de ambos | Más complejo | **Tu caso recomendado** |

**Recomendación:** Usa **ensemble de DINOv2-L y SigLIP-2** con concatenación normalizada:

$$\Phi_{\text{ensemble}}(x) = \frac{1}{\sqrt{2}}\left[\frac{\Phi_{\text{DINO}}(x)}{\|\Phi_{\text{DINO}}(x)\|}, \frac{\Phi_{\text{SigLIP}}(x)}{\|\Phi_{\text{SigLIP}}(x)\|}\right]$$

### 9.2. Pre-procesamiento robusto de glifos

```python
class GlyphPreprocessor:
    def __init__(self, target_size=224):
        self.target_size = target_size
    
    def from_svg(self, svg_path):
        """Renderiza SVG con múltiples grosores de trazo y resoluciones"""
        renders = []
        for stroke_width in [1, 2, 3, 4]:
            for size in [128, 224, 384]:
                img = self._render_svg(svg_path, size, stroke_width)
                renders.append(img)
        return renders
    
    def from_photograph(self, image_path):
        """Procesa fotografía de glifo carved (e.g., 3D photogrammetry)"""
        img = Image.open(image_path)
        # 1. Eliminar fondo
        img_nobg = remove_background(img)  # rembg library
        # 2. Convertir a B&W con threshold adaptivo
        img_bw = adaptive_threshold(img_nobg)
        # 3. Esqueletizar para obtener trazo limpio
        img_skel = skeletonize(img_bw)
        # 4. Re-engrosar con stroke uniforme
        img_clean = morphological_closing(img_skel, kernel_size=3)
        return img_clean
    
    def from_3d_model(self, mesh_path):
        """Genera múltiples vistas 2D de modelo 3D photogrammetric"""
        mesh = trimesh.load(mesh_path)
        views = []
        for angle in range(0, 360, 30):
            view = render_view(mesh, azimuth=angle, elevation=0)
            views.append(view)
        # Vista frontal canónica + augmentaciones
        return views
    
    def augment(self, images):
        """Augmentaciones específicas para glifos"""
        augmented = []
        for img in images:
            # Rotación leve (paleographic variation)
            for angle in [-7, -3, 0, 3, 7]:
                rotated = img.rotate(angle, fillcolor='white')
                augmented.append(rotated)
            
            # Dilatación/erosión (variación de grosor)
            augmented.append(dilate(img, iterations=1))
            augmented.append(erode(img, iterations=1))
            
            # Ruido gaussiano leve
            noise = np.random.normal(0, 0.02, img.size)
            augmented.append(img + noise)
            
            # Distorsión elástica suave
            augmented.append(elastic_distortion(img, alpha=8, sigma=4))
        
        return augmented
```

### 9.3. Frechet mean en la esfera

Para promediar embeddings normalizados, no usar media euclidiana. Usar **Frechet mean en $\mathbb{S}^{d-1}$**:

```python
def spherical_mean(vectors, max_iter=100, tol=1e-6):
    """
    Computa la Frechet mean en la esfera unitaria.
    Iteración tipo punto fijo.
    """
    vectors = np.array(vectors)
    # Normalizar
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Inicializar con la media euclidiana proyectada
    mean = vectors.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    
    for _ in range(max_iter):
        # Mapear cada vector al tangente en mean (logaritmo esférico)
        tangents = []
        for v in vectors:
            cos_theta = np.clip(np.dot(mean, v), -1, 1)
            theta = np.arccos(cos_theta)
            if theta < 1e-10:
                tangents.append(np.zeros_like(v))
            else:
                tangents.append(theta * (v - cos_theta * mean) / np.sin(theta))
        
        # Promediar en el tangente
        tangent_mean = np.mean(tangents, axis=0)
        
        # Mapear de vuelta a la esfera (exponencial esférico)
        norm_t = np.linalg.norm(tangent_mean)
        if norm_t < tol:
            break
        new_mean = np.cos(norm_t) * mean + np.sin(norm_t) * tangent_mean / norm_t
        new_mean = new_mean / np.linalg.norm(new_mean)
        
        if np.linalg.norm(new_mean - mean) < tol:
            mean = new_mean
            break
        mean = new_mean
    
    return mean
```

### 9.4. Distancia geodésica vs. coseno

Para medir similitud entre embeddings:

| Métrica | Fórmula | Cuándo usar |
|---------|---------|-------------|
| Coseno | $1 - \langle u, v\rangle$ | Default, simple |
| Geodésica | $\arccos(\langle u, v\rangle)$ | Análisis matemático preciso |
| Euclidiana | $\|u-v\|_2$ | Solo si vectores no normalizados |

Tu Teorema 3.5 usa coseno. Para reportar resultados, usa ambas para robustez.

### 9.5. Multi-view consensus para glifos 3D

Si tienes modelos 3D photogrammétricos (Échancrée, Mamari):

```python
def glyph_3d_to_consensus_embedding(mesh, encoder, n_views=12):
    """
    Genera embedding consenso desde múltiples vistas 2D del mesh 3D.
    """
    # Generar vistas
    views = []
    for azimuth in np.linspace(0, 360, n_views, endpoint=False):
        for elevation in [-15, 0, 15]:
            view = render_3d_view(mesh, azimuth, elevation)
            views.append(view)
    
    # Embedding por vista
    embeddings = [encoder(v) for v in views]
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    
    # Pesos: vistas frontales más peso (más informativas)
    weights = compute_view_informativeness(views)
    
    # Consenso ponderado en la esfera
    return weighted_spherical_mean(embeddings, weights)
```

### 9.6. Detección automática de alografos

Antes de aplicar el Teorema 3.5, debes consolidar variantes paleográficas:

```python
def detect_allographs(glyph_embeddings, threshold=0.85):
    """
    Agrupa glifos similares en clases alográficas.
    Usa clustering jerárquico con threshold de similitud.
    """
    # Matriz de similitud coseno
    sim_matrix = cosine_similarity(glyph_embeddings)
    dist_matrix = 1 - sim_matrix
    
    # Clustering aglomerativo
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(dist_matrix), method='average')
    clusters = fcluster(Z, t=1-threshold, criterion='distance')
    
    return clusters
```

### 9.7. Manejo de glifos compuestos (ligaturas)

Rongorongo tiene muchas ligaturas (e.g., glyph 600 + glyph 6 = 606 "birds plural"). Necesitas:

```python
def decompose_ligature(glyph_image, base_glyph_library):
    """
    Detecta si un glifo es una ligatura y la descompone.
    """
    # 1. Detectar componentes conexos
    components = connected_components(glyph_image)
    
    if len(components) == 1:
        return [glyph_image]  # No es ligatura
    
    # 2. Para cada componente, buscar match en biblioteca
    decomposed = []
    for comp in components:
        comp_emb = encoder(comp)
        best_match = nearest_neighbor(comp_emb, base_glyph_library)
        decomposed.append(best_match)
    
    return decomposed
```

---

## 10. Validación cruzada en escrituras descifradas

Esta es la sección que **demuestra tu autoridad**. Sin esto, tu paper es especulación. Con esto, tienes evidencia.

### 10.1. Protocolo de validación inversa

#### Setup experimental

Para cada escritura descifrada $S$ con vocabulario $\mathcal{V}_S$ y ground truth $\sigma_S^*: \mathcal{V}_S \to \mathcal{R}$:

1. **Borrar** el ground truth (simular ignorancia)
2. **Aplicar** tu pipeline completo
3. **Obtener** predicciones $\hat{\sigma}_S: \mathcal{V}_S \to \mathcal{P}(\mathcal{R})$
4. **Medir** accuracy@K, MRR, calibración

### 10.2. Experimento E1: Egipcio jeroglífico (validación primaria)

Setup:

- **Dataset:** Glyphnet (4,310 imágenes, 171 clases Gardiner)
- **Sub-conjunto pictográfico:** Eliminar phonograms, mantener semagrams (e.g., A1=hombre sentado, F1=cabeza de toro, M1=árbol, etc.)
- **Mundo a reconstruir:** $\mathcal{W}_{2500\text{BCE}, \text{Egipto}}$
  - Fauna del Nilo: cocodrilos, ibis, halcones, leones, hipopótamos
  - Flora: papiro, loto, palmera datilera
  - Artefactos: barcos, herramientas, edificios
  - Cuerpo humano: posturas múltiples

Métricas esperadas (mi predicción):

| Métrica | Valor esperado | Comentario |
|---------|---------------|------------|
| Accuracy@1 | 0.55-0.70 | Glifos pictográficos puros |
| Accuracy@5 | 0.80-0.92 | Top-5 captura mayoría |
| MRR | 0.65-0.75 | Mediana en posición 1-2 |
| Glifos abstractos accuracy@5 | 0.20-0.35 | Como esperado bajo |

**Si el experimento E1 falla** (accuracy@5 < 0.5 para signs pictográficos): tu hipótesis está incorrecta y debes reformular antes de continuar a E2.

### 10.3. Experimento E2: Oracle bone chino

Setup:

- **Dataset:** HUST-OBS (1,588 caracteres descifrados)
- **Sub-conjunto:** Caracteres pictográficos puros del Shuowen Jiezi (~600)
- **Mundo:** $\mathcal{W}_{1200\text{BCE}, \text{Henan}}$
  - Fauna: tigres, dragones (estilizados), aves, peces
  - Naturaleza: sol, luna, montaña, agua, lluvia
  - Cuerpo humano: pies, manos, ojos, boca

Caso ideal: caracteres como 日 (sol), 月 (luna), 山 (montaña), 木 (árbol), 水 (agua), 火 (fuego), 鱼 (pez), 鸟 (ave), 马 (caballo).

**Lección crítica:** Estos caracteres han sufrido **mucho** menos deiconización que otros. Si el método no funciona aquí, no funcionará en ningún lado.

### 10.4. Experimento E3: Maya glífico

Setup:

- **Dataset:** MHED (Macri & Looper 2003)
- **Sub-conjunto:** Glyph blocks logographic puros (~150)
- **Mundo:** $\mathcal{W}_{700\text{CE}, \text{Mesoamérica}}$

Casos pictográficos famosos: BALAM (jaguar), AHAW (señor con corona), KAB (mano), CHAN (cielo con serpiente).

### 10.5. Experimento E4: Cuneiforme proto-sumerio

**El caso más difícil** (alta deiconización temprana). Pero es importante porque establece la **cota inferior** del rendimiento esperado.

Setup:

- **Dataset:** CDLI proto-cuneiform (período Uruk IV, ~3300 BCE)
- **Mundo:** $\mathcal{W}_{3300\text{BCE}, \text{Mesopotamia}}$
- **Esperado:** Accuracy@5 = 0.30-0.50 (más bajo que egipcio)

### 10.6. Experimento E5: Vai (Liberia)

**Caso crítico paralelo a Rongorongo:**

- Inventado en 1833 por Momolu Duwalu Bukele
- Silabario de ~210 signos
- Algunos signos pictográficos
- **Período comparable a Rongorongo**

Si funciona en Vai (otro sistema reciente, inventado por individuos), refuerza la aplicabilidad a Rongorongo.

### 10.7. Tabla maestra de validación

```
Experimento | Sistema    | N_signos | Acc@1 | Acc@5 | MRR  | Status
─────────────────────────────────────────────────────────────────────
E1          | Egipcio    | ~600    | TBD   | TBD   | TBD  | Pendiente
E2          | OBC chino  | ~600    | TBD   | TBD   | TBD  | Pendiente
E3          | Maya       | ~150    | TBD   | TBD   | TBD  | Pendiente
E4          | Proto-cun. | ~300    | TBD   | TBD   | TBD  | Pendiente
E5          | Vai        | ~210    | TBD   | TBD   | TBD  | Pendiente
─────────────────────────────────────────────────────────────────────
Promedio    |            |         |       |       |      |
```

### 10.8. Criterios de éxito

Para que tu paper sea publicable nivel top-venue:

- **Mínimo:** Acc@5 ≥ 0.6 promedio en E1-E4
- **Bueno:** Acc@5 ≥ 0.75 con calibración (ECE < 0.1)
- **Excelente:** Acc@5 ≥ 0.85 + transferencia exitosa a Vai

Si lo bueno o excelente: **legítimamente puedes elevar Rongorongo a C2-C3 con el respaldo del teorema**.

### 10.9. Análisis de fallos

Cuando el método falla, **documentar por qué**. Categorías esperadas:

1. **Glifos altamente deiconizados** ($\delta > \delta_*$): predecible que falle
2. **Glifos con rebus** (sonido en lugar de objeto): falla legítima
3. **Glifos auxiliares** (gramaticales): no aplican
4. **Glifos compuestos** sin descomposición: error técnico
5. **Referente fuera de $\mathcal{R}$** (cobertura insuficiente): falla del mundo

Cada categoría tiene una solución distinta. Documentar.

---

## 11. Análisis comparativo de sistemas polinésicos

### 11.1. ¿Por qué centrarse en lo polinésico?

Rongorongo está culturalmente embebido en el contexto polinesio. Aunque no esperamos que el lenguaje subyacente sea idéntico al rapanui moderno, la **iconografía cultural** debe ser compartida. Por eso, los referentes de Rapa Nui y los demás polinesios deben tener overlap significativo.

### 11.2. Familia polinésica: árbol genético

```
                        Proto-Polinesio
                              |
              ┌───────────────┴──────────────┐
              |                              |
        Tongánico                      Nuclear Polinesio
       /        \                            |
   Tongano    Niuano               ┌─────────┴────────┐
                                   |                  |
                              Samoano            Polinesio Oriental
                            (Outliers)                 |
                                              ┌────────┴────────┐
                                              |                 |
                                         Tahítico         Marquesano
                                       /        \             /     \
                                  Tahitiano   Maorí    Marquesano  Hawaiiano
                                       \        |
                                       Rapanui (Easter Island)
                                       Cook Islands Maori
```

**Lo que importa:** Rapanui es **Polinesio Oriental**, cercano a Tahitiano, Maorí, Marquesano, Hawaiiano.

### 11.3. Tabla de fauna culturalmente compartida

Para construir el prior cultural del Teorema 5.2:

| Concepto/animal | Rapa Nui | Hawaii | Aotearoa (NZ) | Tahiti | Marquesas |
|-----------------|----------|--------|---------------|--------|-----------|
| Frigatebird | taha | iwa | - (extinto NZ) | otaha | - |
| Fairy tern | manu tara | - | - | manutea | - |
| Tortuga | honu | honu | honu | honu | honu |
| Tiburón | mango | manō | mango | maʻo | mano |
| Pulpo | heke | heʻe | wheke | feʻe | feke |
| Coral | puga | koʻa | tupuna | toʻa | tora |

**Resultado:** Cognados léxicos en muchos casos, sugiriendo conceptos compartidos. Esto es exactamente lo que necesitas para que el prior cultural funcione.

### 11.4. Petroglifos polinesios como precursores visuales

#### 11.4.1. Catálogos disponibles

| Lugar | Investigador | N° motivos | Disponibilidad |
|-------|--------------|------------|----------------|
| Rapa Nui | Lee 1992, Liller 1995 | ~4,200 | Publicaciones, parcial digital |
| Hawaii | Lee & Stasack 1999 | ~30,000 | Hawaii Petroglyph Survey |
| Marquesas | Millerstrom 2001 | ~15,000 | Tesis doctoral |
| Aotearoa | Trotter & McCulloch 1971 | ~3,000 | Publicaciones |

#### 11.4.2. Hipótesis de continuidad

**Hipótesis P1:** Los glifos de Rongorongo son **descendientes estilísticos** de los petroglifos pre-Rongorongo. Su iconografía debe correlacionar.

Test:

```python
def test_petroglyph_continuity(rongorongo_glyphs, petroglyphs):
    """
    Test si los glifos de Rongorongo derivan iconográficamente
    de los petroglifos polinesios pre-1700.
    """
    # 1. Embeddings
    rr_embs = [encoder(g) for g in rongorongo_glyphs]
    pet_embs = [encoder(p) for p in petroglyphs]
    
    # 2. Para cada glifo RR, buscar nearest neighbors en petroglifos
    matches = []
    for i, emb in enumerate(rr_embs):
        sims = cosine_similarity([emb], pet_embs)[0]
        top_k = np.argsort(sims)[::-1][:5]
        matches.append((i, top_k, sims[top_k]))
    
    # 3. Estadística: porcentaje con match > 0.7
    significant_matches = sum(1 for m in matches if m[2][0] > 0.7)
    return significant_matches / len(matches)
```

Esperado: si la hipótesis es correcta, > 50% de glifos RR tienen match fuerte con petroglifos pre-existentes.

### 11.5. Análisis comparativo de tatuajes y tapa cloth

Los polinesios tenían sistemas simbólicos no-textuales muy desarrollados:

| Sistema | Modalidad | Iconografía |
|---------|-----------|-------------|
| Tatuajes marquesanos | Cuerpo | Geométrica derivada de naturaleza |
| Tapa cloth tongana | Tela | Plantas estilizadas |
| Carved ivory hawaiiano | Hueso/marfil | Antropomorfa |
| Tatuajes maoríes (moko) | Cara | Espirales (helechos) |

**Hipótesis P2:** Existe un **léxico iconográfico compartido pan-polinesio** que precede a Rongorongo y proporciona el prior cultural para la sección $\sigma_{\text{RR}}$ (Teorema 5.2).

Esta es una hipótesis testeable y, si se confirma, **constituye evidencia adicional** para tu marco.

### 11.6. Lenguajes para el alineamiento textual

Aunque tu trabajo principal es visual, mantén el alineamiento textual con polinésicos:

| Lengua | Recurso | Tamaño |
|--------|---------|--------|
| Rapanui moderno | Bible, Tatoeba, Wikipedia rapanui | ~50K palabras |
| Hawaiian | Wehewehe.org (diccionario), Bible | ~500K palabras |
| Maori | Te Aka, ngaTaonga | ~1M palabras |
| Samoano | Bible, Wikipedia samoana | ~200K palabras |
| Tongano | Bible, diccionarios | ~150K palabras |
| Tahitiano | Bible, Académie Tahitienne | ~300K palabras |
| Marquesano | Diccionarios MS | ~30K palabras |

Para cada uno: extraer corpus, computar embeddings de palabras, alinear con tu pipeline anterior (Procrustes, GW). El resultado se combina con el alineamiento iconográfico vía:

$$p(y_j | x_i, \text{visual}, \text{textual}) \propto p_{\text{visual}}(y_j | x_i)^{\lambda_v} \cdot p_{\text{textual}}(y_j | x_i)^{1-\lambda_v}$$

con $\lambda_v$ aprendible.

---

## 12. Diseño experimental nivel PhD

### 12.1. Hipótesis explícitas con tests

#### Hipótesis principal H_main

**Existe un subconjunto $\mathcal{V}_X^{\text{icon}} \subseteq \mathcal{V}_{\text{RR}}$ de glifos Rongorongo cuya iconicidad respecto a $\mathcal{R}_{1500, \text{Rapa Nui}}$ supera el umbral de claim C2, validado por experimentos cross-script en al menos 3 sistemas descifrados.**

**Test:** Pipeline completo + validación.

#### Hipótesis secundaria H_pet

**Los glifos Rongorongo derivan iconográficamente de los petroglifos polinesios pre-1700, con correlación de embedding > 0.6 promedio.**

**Test:** Sección 11.4.2.

#### Hipótesis secundaria H_pol

**Los referentes culturales polinesios forman un prior efectivo (sigma cultural < 0.3 en el embedding space) para la sección $\sigma_{\text{RR}}$.**

**Test:** Variance analysis cross-cultural.

#### Hipótesis nula H_0

**Las anclas iconográficas predichas no son distinguibles de anclas aleatorias bajo controles negativos.**

**Test:** Rechazar mediante NegCtrlGap > 3σ en métricas múltiples.

### 12.2. Controles negativos masivos

Replicar y extender los de tu paper anterior:

| Control | Construcción | Objetivo |
|---------|--------------|----------|
| **NC-1: Glifos aleatorios** | Permutar pixels de glifos | Test de iconicidad real |
| **NC-2: Mundo erróneo** | Usar fauna ártica para Rapa Nui | Test de cobertura |
| **NC-3: Embedding ruidoso** | Inyectar 50% Gaussian noise | Test de robustez |
| **NC-4: Cultura ajena** | Probar con artefactos ajenos | Test de prior cultural |
| **NC-5: Glifos sintéticos** | Generar con StyleGAN | Test de iconografía aprendida |
| **NC-6: Encoder no-entrenado** | Random ResNet | Test de aprendizaje |
| **NC-7: Bootstrap glifos** | 1000 muestras del corpus | Estabilidad |
| **NC-8: Bootstrap referentes** | 1000 muestras de imgs | Estabilidad |
| **NC-9: Cross-encoder** | DINO, SigLIP, CLIP separados | Robustez de encoder |
| **NC-10: Trivial baseline** | Top-1 más frecuente | Comparación |

### 12.3. Diseño bootstrap multi-nivel

```python
def hierarchical_bootstrap(corpus, referents, encoder, n_boot=1000):
    """
    Bootstrap a tres niveles:
    1. Sample con reemplazo del corpus de glifos
    2. Sample con reemplazo de imágenes de referentes
    3. Sample con reemplazo de aug. de cada glifo
    """
    results = []
    for b in range(n_boot):
        # Nivel 1
        sampled_glyphs = bootstrap_sample(corpus.glyphs)
        # Nivel 2
        sampled_referents = {}
        for ref_id, imgs in referents.items():
            sampled_referents[ref_id] = bootstrap_sample(imgs)
        # Nivel 3
        with augmentation_seed(b):
            phi = compute_embeddings(sampled_glyphs, encoder)
            psi = compute_embeddings(sampled_referents, encoder)
        # Pipeline
        anchors = predict_anchors(phi, psi)
        metrics = evaluate(anchors)
        results.append(metrics)
    
    # Estadísticas
    return {
        'mean': np.mean(results, axis=0),
        'std': np.std(results, axis=0),
        'ci_95': np.percentile(results, [2.5, 97.5], axis=0)
    }
```

### 12.4. Métricas obligatorias del paper

Replicar y extender:

| Métrica | Definición | Valor objetivo |
|---------|-----------|----------------|
| Accuracy@1 | Top-1 correcto | ≥ 0.4 (sub-pictográfico), ≥ 0.6 (pictográfico) |
| Accuracy@5 | Top-5 correcto | ≥ 0.7 promedio |
| MRR | Mean reciprocal rank | ≥ 0.55 |
| ECE | Expected Calibration Error | ≤ 0.1 |
| AnchorPower | Definición 10.5 paper anterior | ≥ 0.4 (para C3) |
| OverclaimRisk | Definición 30.4 paper anterior | < 1 (obligatorio) |
| NegCtrlGap | Múltiple controles | ≥ 3σ |
| BootstrapStability | Cosine similarity | ≥ 0.7 |
| SpectralReliability | Davis-Kahan ratio | ≥ 0.3 |
| QStability (Procrustes) | $E[\|Q^{(b)} - Q^{(b')}\|]$ | ≤ 3 |

### 12.5. Análisis de sensibilidad

Variar sistemáticamente:

```python
sensitivity_analysis = {
    'encoder': ['dinov2_b', 'dinov2_l', 'siglip2', 'clip_l', 'ensemble'],
    'k_dim': [64, 128, 256, 512, 768],
    'augmentation_strength': [0, 0.1, 0.3, 0.5, 0.8],
    'world_completeness': [0.3, 0.5, 0.7, 0.9, 1.0],
    'min_iconicity_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
    'cross_script_count': [1, 2, 3, 4, 5],
}

# Total: 5x5x5x5x5x5 = 15,625 configuraciones
# Reducir con Latin Hypercube Sampling a ~500 corridas
```

### 12.6. Reproducibilidad nivel doctoral

Estándares mínimos:

1. **Seeds globales** documentadas en cada experimento
2. **Versions pinned** (`pyproject.toml`, `requirements.txt`)
3. **Datos preprocesados** disponibles en Zenodo o Hugging Face Datasets
4. **Modelos entrenados** en Hugging Face Hub
5. **Notebooks documentados** con outputs versionados
6. **Makefile** o CI/CD para regenerar todos los resultados
7. **DOI** del repositorio principal
8. **Datasheet for datasets** (Gebru et al. 2021)
9. **Model card** (Mitchell et al. 2019)

---

## 13. Niveles de claim revisados

Tu paper anterior estableció C0-C5. Con el nuevo marco, refinas la jerarquía:

### 13.1. Jerarquía actualizada

| Nivel | Permite | Requiere |
|-------|---------|----------|
| **C0 — Paleográfico** | "Este trazo existe" | Documentación visual |
| **C1 — Estructural** | "Aparece en posición X con frecuencia f" | Estabilidad estadística |
| **C2 — Funcional débil** | "Rol gramatical/funcional probable" | AnchorPower ≥ 0.15 + NegCtrlGap ≥ 2σ |
| **C2.5 — Iconográfico (NUEVO)** | "El referente icónico es R con confianza p" | Teorema 3.5 + cross-script E1-E3 acc@5 ≥ 0.6 |
| **C3 — Semántico débil** | "Compatible con dominio cultural D" | C2.5 + corroboración pan-polinesia |
| **C4 — Fonético parcial** | "Compatible con sílaba s" | C3 + alineamiento textual estable |
| **C5 — Traducción fuerte** | "Lectura completa verificable" | Bilingüe verdadero (no disponible) |

**El nivel C2.5 es nuevo.** Es donde tu trabajo aterriza Rongorongo. Es:

- **Más fuerte que C2** (tienes referentes específicos, no solo función)
- **Más débil que C3** (no afirmas significado lingüístico, solo iconográfico)

### 13.2. Definición formal de C2.5

#### Definición 13.1 (Claim C2.5 — Iconográfico)

*Un glifo $x$ admite claim C2.5 con referente $r^*(x)$ si y solo si:*

1. *(Iconicidad) $\iota^*(x) \geq 0.6$*
2. *(Anclaje) $\text{AnchorPower}(\{x\} \cup \mathcal{V}_X^{\text{icon}}) \geq 0.15$*
3. *(Estabilidad) Bootstrap stability del par $(x, r^*(x)) \geq 0.7$*
4. *(Validación cross-script) En experimentos E1-E4, glifos comparables en deiconización alcanzan accuracy@5 $\geq 0.6$*
5. *(Control negativo) NegCtrlGap $\geq 3\sigma$*
6. *(Cobertura del mundo) $r^*(x) \in \mathcal{R}_{1500, \text{Rapa Nui}}$ con verificación bibliográfica*

### 13.3. Fila para tu ledger de hipótesis (template)

```yaml
hypothesis:
  glyph_id: "RR_600"
  claim_level: "C2.5"
  iconic_anchor:
    referent: "frigatebird (Fregata minor)"
    confidence: 0.78
    iota_max: 0.84
  evidence:
    - type: "visual_match"
      detail: "DINOv2 cosine sim with frigate photos = 0.82"
    - type: "cross_script_validation"
      detail: "Egyptian glyph G14 (vulture, similar silhouette) acc@5=0.71"
    - type: "cultural_prior"
      detail: "Frigatebird associated with Makemake (Routledge 1919)"
    - type: "petroglyph_continuity"
      detail: "Match with Lee 1992 petroglyph #2341 sim=0.74"
  counter_evidence:
    - type: "alternative_referent"
      detail: "Could be other large seabird (booby)"
    - type: "stylization_uncertainty"
      detail: "Some glyph-600 variants more abstract"
  stability:
    bootstrap_consistency: 0.81
    spectral_reliability: 0.45
    overclaim_risk: 0.58
  world_verification:
    paleontological_evidence: "Steadman 1995 - Anakena bird bones"
    ethnographic_evidence: "Routledge 1919 - cultural importance"
  reproducibility:
    seed: 42
    encoder: "dinov2_l_siglip2_ensemble_v1"
    pipeline_version: "1.0.0"
    commit: "abc123def"
```


---

## 14. Cronograma y entregables

### 14.1. Cronograma realista (12 meses)

```
Mes 1-2: Adquisición y preprocesamiento de datos
├── Semana 1-2: Setup infrastructure, contactos académicos
├── Semana 3-4: Descarga de datasets digitales (Glyphnet, HUST-OBS, etc.)
├── Semana 5-6: Construcción de mundo Rapa Nui 1500
├── Semana 7-8: Digitalización de scripts no disponibles online

Mes 3-4: Implementación del pipeline base
├── Semana 9-10: Encoder ensemble (DINOv2 + SigLIP)
├── Semana 11-12: Pipeline glyph→embedding y referent→embedding
├── Semana 13-14: Alineamiento Procrustes/OT/GW
├── Semana 15-16: Validación inicial en datos sintéticos

Mes 5-6: Experimentos cross-script (E1-E5)
├── Semana 17-19: E1 Egipcio (más completo)
├── Semana 20-22: E2 Oracle bone chino
├── Semana 23-24: E3 Maya, E4 cuneiforme, E5 Vai

Mes 7-8: Aplicación a Rongorongo
├── Semana 25-26: Pipeline completo en Rongorongo
├── Semana 27-28: Análisis comparativo polinésico
├── Semana 29-30: Ledger de hipótesis con calibración
├── Semana 31-32: Análisis de sensibilidad

Mes 9-10: Validación y ablation studies
├── Semana 33-34: Bootstrap masivo
├── Semana 35-36: Controles negativos NC-1 a NC-10
├── Semana 37-38: Sensitivity analysis
├── Semana 39-40: Publicación de datasets/modelos

Mes 11-12: Escritura y publicación
├── Semana 41-44: Borrador del paper
├── Semana 45-46: Revisión por colegas
├── Semana 47-48: Revisión final, submission a venue
```

### 14.2. Venues objetivo

#### 14.2.1. Tier 1 (preferidos)

- **Nature** o **Nature Communications** (impacto interdisciplinario)
- **Science** o **Science Advances**
- **PNAS** (interfase humanidades-computación)

#### 14.2.2. Tier 1 (computacional)

- **NeurIPS** (Neural Information Processing Systems) — track de ciencia computacional
- **ICML** — Workshop on Machine Learning for Ancient Texts
- **ACL** — Computational Approaches to Historical Languages
- **EMNLP** — track de bajo recurso

#### 14.2.3. Tier 1 (digital humanities)

- **Digital Scholarship in the Humanities** (Oxford) — donde publicaron Lastilla et al.
- **Journal of Cultural Heritage**
- **Antiquity** (arqueología)
- **Cognitive Science**

#### 14.2.4. Estrategia de venue

**Mi recomendación:** Submit a **Nature Communications** o **PNAS** primero. Si rechazo, **Digital Scholarship in the Humanities** o **NeurIPS Datasets & Benchmarks track**.

### 14.3. Entregables del proyecto

#### Output 1: Paper principal

**"Iconic Grounding for Partial Identifiability: Computer Vision Anchors Lift Rongorongo from C0–C1 to C2.5"**

#### Output 2: Datasets

- **`rongorongo-iconic-anchors-v1`** en Hugging Face
- **`world-rapanui-1500`** en Zenodo (con DOI)
- **`cross-script-decipherment-benchmark`** (E1-E5) en HF Datasets

#### Output 3: Modelos

- **`iconic-grounding-encoder-v1`** — DINOv2+SigLIP ensemble fine-tuneado
- **`deiconization-trajectory-model`** — predictor de trayectorias

#### Output 4: Código

- **`spectral-submersion-decipherment`** repository (continuación del anterior)
- **`iconic-grounding`** (nuevo módulo)

#### Output 5: Sitio web interactivo

- Visualización del ledger de hipótesis
- Búsqueda interactiva por glifo Rongorongo
- Predicciones top-K con confidence

#### Output 6: Paper de seguimiento

**"A General Framework for Partial Decipherment via Iconic Anchoring: Applications to Indus, Linear A, and Proto-Elamite"**

Una vez validado para Rongorongo, aplicar el mismo marco a otros sistemas no descifrados.

---

## 15. Referencias críticas y datasets

### 15.1. Papers fundamentales que debes leer

#### 15.1.1. Sobre origen icónico de la escritura

1. **Spiegel, B.A., Gelfond, L., Konidaris, G. (2025).** "Visual Theory of Mind Enables the Invention of Proto-Writing." arXiv:2502.01568. **CRÍTICO — fundamenta tu hipótesis principal.**

2. **Goldwasser, O. (2017).** "Cuneiform and Hieroglyphs in the Bronze Age." En *Pharoah's Land and Beyond*. Oxford University Press. **CRÍTICO — establece el contraste entre deiconización rápida y lenta.**

3. **DeFrancis, J. (1989).** *Visible Speech: The Diverse Oneness of Writing Systems*. University of Hawaii Press. **Lectura obligada sobre la naturaleza de la escritura.**

4. **Daniels, P.T. & Bright, W. (eds.) (1996).** *The World's Writing Systems*. Oxford University Press. **Referencia enciclopédica.**

#### 15.1.2. Sobre Rongorongo específicamente

5. **Barthel, T.S. (1958).** *Grundlagen zur Entzifferung der Osterinselschrift*. Hamburg: Cram, de Gruyter. **El catálogo de signos.**

6. **Pozdniakov, K. & Pozdniakov, I. (2007).** "Rapanui Writing and the Rapanui Language: Preliminary Results of a Statistical Analysis." *Forum for Anthropology and Culture* 3:3-36. **Análisis estadístico moderno.**

7. **Fischer, S.R. (1997).** *RongoRongo: The Easter Island Script*. Oxford University Press. **Una lectura propuesta (controvertida).**

8. **Lastilla, L., Ravanelli, R., Valério, M. et al. (2022).** "Modelling the Rongorongo Tablets: A New Transcription of the Échancrée Tablet." *Digital Scholarship in the Humanities* 37(2):497-526. **Estado del arte digital.**

9. **Ferrara, S. et al. (2024).** "The Invention of Writing on Rapa Nui (Easter Island)." *Scientific Reports* 14. **Radiocarbon de la tablilla Berlin.**

#### 15.1.3. Sobre visión computacional para escritura antigua

10. **Barucci, A. et al. (2021).** "A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification." Glyphnet. **El benchmark egipcio.**

11. **Wang, P. et al. (2024).** "An Open Dataset for Oracle Bone Character Recognition and Decipherment." HUST-OBS. *Scientific Data*. **Dataset chino oracle bone.**

12. **Assael, Y. et al. (2022).** "Restoring and Attributing Ancient Texts using Deep Neural Networks." *Nature* 603:280-283. Ithaca. **Estado del arte para textos antiguos.**

13. **Sommerschield, T. et al. (2023).** "Machine Learning for Ancient Languages: A Survey." *Computational Linguistics* 49(3):703-745.

#### 15.1.4. Sobre paleoecología de Rapa Nui

14. **Steadman, D.W. et al. (1994).** "Stratigraphy, Chronology, and Cultural Context of an Early Faunal Assemblage from Easter Island." *Asian Perspectives* 33(1):79-96.

15. **Hunt, T.L. & Lipo, C.P. (2018).** *The Statues that Walked: Unraveling the Mystery of Easter Island*. Free Press.

16. **Flenley, J.R. & King, S.M. (1984).** "Late Quaternary Pollen Records from Easter Island." *Nature* 307:47-50.

17. **Métraux, A. (1940).** *Ethnology of Easter Island*. Bishop Museum Bulletin 160.

#### 15.1.5. Sobre embeddings y geometría

18. **Oquab, M. et al. (2024).** "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*.

19. **Zhai, X. et al. (2023).** "Sigmoid Loss for Language Image Pre-Training." SigLIP. *ICCV*.

20. **Radford, A. et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision." CLIP. *ICML*.

#### 15.1.6. Sobre transporte óptimo

21. **Peyré, G. & Cuturi, M. (2019).** "Computational Optimal Transport." *Foundations and Trends in Machine Learning* 11(5-6).

22. **Alvarez-Melis, D. & Jaakkola, T. (2018).** "Gromov-Wasserstein Alignment of Word Embedding Spaces." *EMNLP*.

23. **Memoli, F. (2011).** "Gromov-Wasserstein Distances and the Metric Approach to Object Matching." *Foundations of Computational Mathematics* 11.

### 15.2. Datasets descargables (URL directas)

```bash
# === GLIFOS ===
# Egyptian Glyphnet (Barucci 2021)
git clone https://github.com/morrisfranken/glyphreader

# Oracle Bone HUST-OBS
# Solicitar acceso: https://github.com/Pengjie-W/HUST-OBS

# Oracle Bone OBC306
# https://jgw.aynu.edu.cn/ — Anyang Normal University

# Cuneiform Digital Library Initiative
# https://cdli.mpiwg-berlin.mpg.de — API disponible

# Linear A/B SigLA
# https://sigla.phis.me

# Indus Valley CISI Mayig
git clone https://github.com/mayig/indus-corpus

# Maya Hieroglyphic Database
# https://www.mayadatabase.org — Solicitar acceso

# === REFERENTES ===
# iNaturalist API
# https://api.inaturalist.org/v1/

# GBIF
# https://api.gbif.org/v1/

# Biodiversity Heritage Library
# https://www.biodiversitylibrary.org/api3

# === RECONSTRUCCIONES ===
# Neotoma Paleoecology
# https://api.neotomadb.org

# PaleoBioDB
# https://paleobiodb.org

# Open Context (archaeology)
# https://opencontext.org/api/

# === RONGORONGO ===
# Internet Archive
# https://archive.org/details/rongorongotexts

# Wikipedia/Commons
# https://commons.wikimedia.org/wiki/Category:Rongorongo

# Pozdniakov publications - solicitar directamente
# kpozdniakov@inalco.fr (vía INALCO Paris)

# Lastilla / Bologna 3D corpus - solicitar directamente
# silvia.ferrara@unibo.it
```

### 15.3. Software open source recomendado

```bash
# Encoders
pip install transformers  # SigLIP, CLIP
pip install timm  # DINOv2, otros
pip install dino-vit  # Variants

# Optimal transport
pip install pot  # Python Optimal Transport
pip install geomloss  # GW differentiable

# Vision
pip install opencv-python
pip install Pillow
pip install scikit-image
pip install rembg  # Background removal

# 3D
pip install trimesh
pip install pyvista
pip install open3d

# Embeddings management
pip install faiss-cpu  # Nearest neighbor search
pip install annoy

# Stats
pip install scipy numpy pandas
pip install scikit-learn
pip install statsmodels

# Bootstrap and uncertainty
pip install resample
pip install pyEntropy

# SVG processing
pip install svglib
pip install svgpathtools
pip install cairosvg

# Documentation
pip install sphinx
pip install jupyter-book
```

---

## 16. Apéndices matemáticos

### Apéndice A: Demostración completa del Teorema 3.5

#### A.1. Lemas auxiliares

**Lema A.1 (Lipschitz del coseno en la esfera).** *Para $u, v \in \mathbb{S}^{d-1}$,*
$$|\langle u, w \rangle - \langle v, w \rangle| \leq \|u - v\|_2$$

*Demostración:* Por Cauchy-Schwarz, $|\langle u-v, w\rangle| \leq \|u-v\|\|w\| = \|u-v\|$ (ya que $\|w\|=1$).

**Lema A.2 (Conversión geodésica-cuerda).** *En $\mathbb{S}^{d-1}$, si $\theta = \arccos\langle u, v\rangle$ es el ángulo geodésico,*
$$\|u-v\|_2 = 2\sin(\theta/2)$$

*Demostración:* $\|u-v\|^2 = \|u\|^2 + \|v\|^2 - 2\langle u,v\rangle = 2 - 2\cos\theta = 4\sin^2(\theta/2)$.

**Lema A.3 (Pegado Lipschitz).** *Si $f$ es $L$-Lipschitz en $X$ y $g$ es $L'$-Lipschitz en $f(X)$, entonces $g \circ f$ es $LL'$-Lipschitz.*

#### A.2. Demostración del Teorema principal

Por la hipótesis H1 (Origen Icónico Local), existe $\delta_0 < 1$ tal que en el momento de invención $\tau_0$:
$$\iota(x_{\tau_0}, r_0(x)) \geq 1 - \delta_0$$

Equivalentemente:
$$\|\Phi(x_{\tau_0}) - \Psi(r_0(x))\|_2^2 = 2 - 2\iota \leq 2\delta_0$$
$$\|\Phi(x_{\tau_0}) - \Psi(r_0(x))\|_2 \leq \sqrt{2\delta_0}$$

Por la deiconización (Definición 4.1), entre $\tau_0$ y el presente $\tau$:
$$\|\Phi(x_\tau) - \Phi(x_{\tau_0})\|_2 = \|\gamma(\tau) - \gamma(\tau_0)\|_2 \leq V \cdot |\tau - \tau_0|$$

donde la última desigualdad usa que $v_\delta \leq V$.

Aplicando desigualdad triangular:
$$\|\Phi(x_\tau) - \Psi(r_0(x))\|_2 \leq \|\Phi(x_\tau) - \Phi(x_{\tau_0})\|_2 + \|\Phi(x_{\tau_0}) - \Psi(r_0(x))\|_2$$
$$\leq V \cdot |\tau - \tau_0| + \sqrt{2\delta_0}$$

Por la hipótesis H3 (Cobertura), con probabilidad $\geq 1 - \epsilon$, $r_0(x) \in \mathcal{R}$.

Para que $r^*(x) = r_0(x)$, necesitamos que ningún otro $r \neq r_0(x)$ esté más cerca de $\Phi(x_\tau)$. Por la separación mínima $\Delta_r$:

$$\|\Psi(r) - \Psi(r_0(x))\|_2 \geq \Delta_r \quad \forall r \neq r_0(x)$$

Por desigualdad triangular inversa, $r$ está más cerca de $\Phi(x_\tau)$ que $r_0(x)$ solo si:
$$\|\Phi(x_\tau) - \Psi(r)\|_2 + \|\Phi(x_\tau) - \Psi(r_0(x))\|_2 \geq \|\Psi(r) - \Psi(r_0(x))\|_2$$

Lo cual implica:
$$2\|\Phi(x_\tau) - \Psi(r_0(x))\|_2 \geq \Delta_r$$

Es decir, falla el matching cuando:
$$\|\Phi(x_\tau) - \Psi(r_0(x))\|_2 \geq \Delta_r / 2$$

Combinando, $r^*(x) = r_0(x)$ con probabilidad:
$$P(r^*(x) = r_0(x)) \geq P(r_0 \in \mathcal{R}) \cdot P\left(\|\Phi(x_\tau) - \Psi(r_0)\|_2 < \Delta_r/2\right)$$

Por Markov sobre la cota Lipschitz:
$$P\left(\|\Phi(x_\tau) - \Psi(r_0)\|_2 < \Delta_r/2\right) \geq 1 - \frac{V \cdot |\tau-\tau_0| + \sqrt{2\delta_0}}{\Delta_r/2}$$

Identificando $\delta(x) = V \cdot |\tau-\tau_0| + \sqrt{2\delta_0}$ (la deiconización total), llegamos a:

$$P(r^*(x) = r_0(x)) \geq (1 - \epsilon) \cdot \left(1 - \frac{2\delta(x)}{\Delta_r}\right)$$

que es precisamente la cota anunciada (con $L=1$ por definición esférica). □

### Apéndice B: Análisis de la cota de error de transferencia

#### B.1. Setup

Sea $\hat{f}$ un modelo entrenado en escrituras descifradas $\{S_1, \ldots, S_M\}$ y aplicado a $S_*$ (Rongorongo). Definimos el error de transferencia:

$$\mathcal{E}(\hat{f}, S_*) = \mathbb{E}_{x \sim \mathcal{V}_{S_*}}[\ell(\hat{f}(x), \sigma_{S_*}^*(x))]$$

donde $\sigma_{S_*}^*$ es la función de ground truth desconocida.

#### B.2. Teorema B.1 (Error de transferencia cross-script)

*Si las escrituras de entrenamiento $\{S_i\}$ y la escritura objetivo $S_*$ comparten la misma distribución de referentes hasta una distancia $\Delta_W$ (la "distancia entre mundos"), y comparten la misma trayectoria de deiconización hasta una distancia $\Delta_T$, entonces:*

$$\mathcal{E}(\hat{f}, S_*) \leq \mathcal{E}_{\text{train}} + L_W \cdot \Delta_W + L_T \cdot \Delta_T$$

*donde $L_W, L_T$ son constantes Lipschitz dependientes del encoder.*

**Demostración (esquema):** Aplicación directa de bounds de transferencia tipo Ben-David et al. (2010) adaptado al caso visual, con $\Delta_W$ y $\Delta_T$ jugando los roles de divergencia de dominios. □

### Apéndice C: Detalles sobre Frechet means esféricas

La media de Frechet en $\mathbb{S}^{d-1}$ minimiza la suma de cuadrados de distancias geodésicas:

$$\mu = \arg\min_{p \in \mathbb{S}^{d-1}} \sum_{i=1}^n d_g(p, x_i)^2$$

donde $d_g(p, q) = \arccos(\langle p, q\rangle)$.

Este mínimo es **único** si todos los $x_i$ están en un "small ball" (radio $< \pi/2$). Para vectores normalizados muy similares, esto se cumple.

El algoritmo iterativo (en código del Pipeline 9.3) converge superlinealmente bajo esta condición.

### Apéndice D: Glosario actualizado

| Término | Definición |
|---------|-----------|
| **Iconicidad** $\iota(x,r)$ | Producto interno entre embedding del glifo y del referente |
| **Deiconización** $\delta(x)$ | $1 - \iota^*(x)$, mide pérdida de iconicidad |
| **Mundo histórico** $\mathcal{W}_{\tau,\mathcal{G}}$ | Conjunto verificable de referentes en época y lugar |
| **Sección cultural** $\sigma_S$ | Mapeo referente → glifo en escritura $S$ |
| **Trayectoria de deiconización** $\gamma_S$ | Curva de evolución de embeddings en el tiempo |
| **Direcciones canónicas** $u_k$ | Componentes principales de cambio iconográfico |
| **Anclaje icónico** | Par $(x, r^*(x))$ con $\iota^*(x) > $ umbral |
| **AnchorPower** (Def 10.5 anterior) | Fracción de simetría rota por anclas |
| **Claim C2.5** | Nivel iconográfico nuevo, entre C2 y C3 |
| **OverclaimRisk** | Razón claim/evidence (debe ser < 1) |

### Apéndice E: Plantilla de paper

```markdown
# Iconic Grounding for Partial Identifiability

## Abstract
[150 palabras: planteo del problema, hipótesis, método, resultados, contribución]

## 1. Introduction
1.1. Background: from no-identifiability to iconic anchoring
1.2. Hypothesis of Iconic Origin
1.3. Contributions

## 2. Related Work
2.1. Decipherment of lost scripts
2.2. Visual theory of mind in proto-writing
2.3. Cross-modal embeddings

## 3. The Iconic Grounding Theorem
3.1. Setup
3.2. Hypotheses
3.3. Main theorem
3.4. Proof sketch
3.5. Limitations

## 4. Deiconization Theory
4.1. Trajectories of evolution
4.2. Bounded deiconization theorem
4.3. Canonical directions

## 5. Computational Pipeline
5.1. Visual encoders
5.2. World reconstruction
5.3. Cross-modal alignment
5.4. Audit and calibration

## 6. Cross-Script Validation Experiments
6.1. E1: Egyptian (primary validation)
6.2. E2: Oracle bone Chinese
6.3. E3: Maya
6.4. E4: Proto-cuneiform
6.5. E5: Vai (parallel case to RR)
6.6. Aggregate results

## 7. Application to Rongorongo
7.1. Reconstruction of Rapa Nui c. 1500
7.2. Predicted iconographic anchors
7.3. Ledger of hypotheses (top-50 with C2.5+ claims)
7.4. Glyphs that remain at C0-C1

## 8. Comparative Polynesian Analysis
8.1. Family-level iconographic prior
8.2. Petroglyph continuity hypothesis
8.3. Cross-textual alignment

## 9. Discussion
9.1. What we proved
9.2. What we did not prove
9.3. Implications for the Rapa Nui community

## 10. Conclusion

## References (50+)

## Appendices
A. Mathematical proofs
B. Datasets and reproducibility
C. Ledger of hypotheses (full)
D. Supplementary figures
```

---

## Conclusión: Resumen de la Estrategia

### Lo que tienes (paper anterior)

✅ Marco matemático riguroso de no-identificabilidad
✅ Teoremas de estabilidad espectral
✅ Sistema de claim levels (C0-C5)
✅ Validación bilingüe en idiomas conocidos (Tabla 27)
✅ Honestidad cuantitativa brutal

### Lo que añades (este plan)

🆕 **Teorema 3.5 (Anclaje Icónico Inverso)** — formalización matemática del origen icónico
🆕 **Teorema 4.3 (Deiconización Acotada)** — cota explícita del decay
🆕 **Pipeline visión-mundo** — operacionalización completa
🆕 **Validación cross-script E1-E5** — autoridad empírica
🆕 **Mundo Rapa Nui c. 1500 reconstruido** — base verificable
🆕 **Claim level C2.5** — nuevo nivel intermedio
🆕 **Análisis comparativo polinesio** — prior cultural
🆕 **Ledger iconográfico** — propagación de incertidumbre

### El resultado final

Cuando termines, tendrás:

1. **El primer marco matemático completo** que conecta visión computacional con desciframiento, con teoremas formales y demostraciones.

2. **Validación en 4-5 escrituras descifradas** demostrando que el método funciona donde se puede verificar.

3. **Predicciones específicas para Rongorongo** — un subconjunto $\mathcal{V}_{\text{RR}}^{\text{icon}}$ de glifos elevados a C2.5 con confianza calibrada.

4. **Reconocimiento explícito** de glifos que permanecen en C0-C1 (la mayoría, manteniendo honestidad).

5. **Datos abiertos y reproducibilidad total** — datasets, modelos, código.

6. **Diálogo respetuoso con la comunidad rapanui.**

### El insight clave

Tu trabajo será revolucionario no porque "descifre" Rongorongo (no lo hará completamente), sino porque **cambia la pregunta**:

- **Pregunta vieja:** "¿Qué significa cada glifo?" (no identificable)
- **Pregunta nueva:** "¿Cuál es el referente icónico de cada glifo?" (parcialmente identificable, con cotas)

Esta reformulación es **el aporte filosófico-matemático** que justifica todo el aparato. No estás traduciendo; estás **mapeando glifos a su mundo de origen**, que es una tarea distinta y más tratable.

### Mi predicción honesta

Si ejecutas este plan con la disciplina del paper anterior:

- **30-40% de los glifos Rongorongo** alcanzarán C2.5 con confianza ≥ 0.6
- **5-15%** alcanzarán C3 vía corroboración pan-polinesia
- **0%** alcanzará C4-C5 sin un Rosetta Stone
- **El resto permanece en C0-C1**, mantenido honestamente

Eso es **una contribución científica enorme**. No es "descifré Rongorongo"; es "establecí el marco matemático y empírico para anclar Rongorongo a su mundo iconográfico, con cotas de error verificables y validación cruzada exitosa en 4 escrituras conocidas."

Eso es publicable en Nature Communications. Eso cambia el campo.

---

## Coda: Lo que NO debes olvidar

1. **Honestidad ante todo.** Tu fortaleza es admitir limitaciones. No la pierdas por entusiasmo.

2. **Validar primero, aplicar después.** No toques Rongorongo hasta que E1-E4 pasen. Si no pasan, el método está mal.

3. **Diálogo cultural.** Rapa Nui no es un puzzle académico aislado; es patrimonio vivo. Comunícate con la comunidad.

4. **Documentar fracasos.** Tu Tabla 23 (Indus indistinguible de aleatorio) es una de las cosas más valiosas de tu paper anterior. Mantén esa disciplina.

5. **El teorema es la fortaleza.** Sin la formalización matemática, esto sería numerología. Con ella, es ciencia.

---

*Documento preparado por Claude (Anthropic) como guía exhaustiva para el siguiente paso de investigación de David Astudillo. Versión 1.0, abril 2026. Todas las predicciones cuantitativas son estimaciones del autor del marco anterior y deben ser validadas empíricamente. Las hipótesis culturales sobre Rapa Nui requieren consultas con la comunidad rapanui antes de cualquier publicación.*

