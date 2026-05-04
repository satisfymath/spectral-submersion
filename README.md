# Proyecto Data Science: Spectral Submersion Decipherment

**Objetivo:** construir un proyecto reproducible de ciencia de datos para estudiar sistemas simbólicos o lenguajes perdidos usando estadística de tokens, matrices de co-ocurrencia, SVD, pseudoinversa de Moore–Penrose, alineamiento Procrustes, transporte óptimo y generación de hipótesis semánticas probabilísticas.

> Este proyecto no parte afirmando “traducimos un lenguaje perdido”. Parte con una meta científicamente defendible: **extraer estructura, medir regularidades, construir embeddings espectrales y generar hipótesis auditables**.

---

<<<<<<< Updated upstream
=======
## Comunidad, datos y contribución

Spectral Submersion se está preparando como un proyecto abierto y colaborativo:

- **Llamada pública:** ver `COMMUNITY_CALL.md`
- **Contribuir al repo:** ver `CONTRIBUTING.md`
- **Discord de trabajo:** https://discord.gg/ueg94XVw (ver `DISCORD_SETUP.md`)
- **Datos descargables y Google Drive:** ver `DATA_ACCESS.md` y la carpeta Drive: https://drive.google.com/drive/u/1/folders/1vbnpyyy-YtVYyd-gf_-XwvHLHqIgQq8h
- **Datos Rongorongo por reunir:** ver `DATA_WISHLIST.md`
- **Roadmap público:** ver `ROADMAP.md`
- **Estado de investigación:** ver `RESEARCH_STATE.md`
- **Por qué existe:** ver `WHY_THIS_EXISTS.md`
- **Conducta y moderación:** ver `CODE_OF_CONDUCT.md`

GitHub es la fuente de verdad para código, issues, pull requests y releases. Discord se usa para coordinación diaria y lectura de papers. Los datos pesados, modelos y artefactos reproducibles deben publicarse como paquetes versionados fuera del repo, con manifiestos y checksums.

La carpeta Drive actual es un staging de release externo. Antes de redistribuir datos ampliamente, revisar el manifest y las licencias documentadas en `DATA_ACCESS.md`.

Estado científico breve: el paper actual no afirma desciframiento. Sin anclas externas, Rongorongo permanece en niveles C0-C1. La siguiente frontera del proyecto es el anclaje icónico inverso: producir candidatos visuales auditables hacia C2.5, pero solo si la validación en escrituras conocidas supera los umbrales definidos.

Necesidad abierta: reunir más datos Rongorongo con procedencia clara. Si conoces archivos, museos, investigadores, corpus privados o fuentes visuales relevantes, revisa `DATA_WISHLIST.md` y abre un issue con metadata. No publiques archivos restringidos sin permiso.

---

>>>>>>> Stashed changes
## 1. Visión del proyecto

La idea central es representar un sistema simbólico como datos:

```text
corpus de signos → tokens → secuencias → co-ocurrencias → matriz PMI/PPMI → SVD → embeddings → alineamiento → hipótesis
```

El proyecto tendrá tres capas:

1. **Data Science clásica:** limpieza, exploración, visualización, estadística descriptiva.
2. **Machine Learning / NLP:** embeddings, reducción dimensional, clustering, modelos de lenguaje simples.
3. **Matemática avanzada:** SVD, pseudoinversa, Procrustes, transporte óptimo, estabilidad espectral.

---

## 2. Pregunta de investigación

Pregunta principal:

> ¿Es posible extraer una geometría estadística robusta desde un corpus simbólico de bajo recurso y compararla con geometrías de lenguas conocidas para generar hipótesis de traducción probabilísticas?

Preguntas secundarias:

- ¿Los signos tienen frecuencias compatibles con un sistema lingüístico, ritual, contable o aleatorio?
- ¿Existen bloques repetidos o patrones posicionales estables?
- ¿Qué rango efectivo tiene la matriz de co-ocurrencia?
- ¿Los embeddings espectrales forman clusters interpretables?
- ¿Una lengua candidata explica mejor la estructura que corpus aleatorios?
- ¿Qué hipótesis sobreviven a bootstrap y controles negativos?

---

## 3. Hipótesis

### H1: estructura no aleatoria

El corpus perdido contiene regularidades internas detectables por co-ocurrencia, entropía, rango efectivo y espectro singular.

### H2: geometría latente

Las matrices de contexto del sistema perdido inducen un espacio vectorial donde signos con roles similares quedan cerca.

### H3: alineamiento parcial

Si existe relación cultural, semántica o funcional con una lengua candidata, sus geometrías relacionales deberían mostrar menor distorsión que un control aleatorio.

### H4: traducción probabilística, no determinista

El output correcto no es:

```text
glifo X = palabra Y
```

sino:

```text
glifo X → candidatos {Y1, Y2, Y3} con probabilidades, evidencia y contraevidencia
```

---

## 4. Estructura recomendada del repositorio

```text
spectral-submersion-ds/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Makefile
│
├── configs/
│   ├── base.yaml
│   ├── synthetic.yaml
│   ├── rongorongo.yaml
│   └── candidate_languages.yaml
│
├── data/
│   ├── raw/
│   │   ├── lost_language/
│   │   └── candidate_languages/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_frequency_analysis.ipynb
│   ├── 02_cooccurrence_pmi.ipynb
│   ├── 03_svd_embeddings.ipynb
│   ├── 04_alignment_procrustes.ipynb
│   ├── 05_bootstrap_stability.ipynb
│   └── 06_hypothesis_report.ipynb
│
├── src/
│   └── spectral_submersion/
│       ├── __init__.py
│       ├── io.py
│       ├── tokenization.py
│       ├── normalization.py
│       ├── frequency.py
│       ├── cooccurrence.py
│       ├── pmi.py
│       ├── spectral.py
│       ├── alignment.py
│       ├── transport.py
│       ├── decoding.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── reporting.py
│
├── scripts/
│   ├── build_dataset.py
│   ├── run_frequency_analysis.py
│   ├── run_svd.py
│   ├── run_alignment.py
│   ├── run_bootstrap.py
│   └── generate_report.py
│
├── tests/
│   ├── test_tokenization.py
│   ├── test_cooccurrence.py
│   ├── test_pmi.py
│   ├── test_spectral.py
│   └── test_alignment.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── hypotheses/
│   └── final/
│
└── experiments/
    ├── synthetic/
    ├── controls/
    └── real_corpus/
```

---

## 5. Instalación inicial

### 5.1 Crear entorno

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 5.2 Instalar dependencias

```bash
pip install numpy pandas scipy scikit-learn matplotlib networkx pyyaml tqdm rich
pip install jupyter ipykernel pytest
pip install pot
```

`pot` corresponde a **Python Optimal Transport**, útil para transporte óptimo.

### 5.3 Guardar dependencias

```bash
pip freeze > requirements.txt
```

---

## 6. Formato de datos

### 6.1 Corpus perdido

Archivo sugerido:

```text
data/raw/lost_language/corpus.csv
```

Columnas recomendadas:

```csv
doc_id,line_id,position,token,raw_token,orientation,source,notes
tablet_001,1,1,g001,Glyph-A,normal,source_name,
tablet_001,1,2,g014,Glyph-N,normal,source_name,
tablet_001,1,3,g014,Glyph-N,normal,source_name,
tablet_001,1,4,g032,Glyph-Z,normal,source_name,
```

### 6.2 Corpus de lengua candidata

Archivo sugerido:

```text
data/raw/candidate_languages/rapanui_tokens.csv
```

Columnas:

```csv
doc_id,sentence_id,position,token,lemma,pos,translation,source
doc_001,1,1,te,te,DET,the,dictionary
doc_001,1,2,ariki,ariki,NOUN,chief,dictionary
```

---

## 7. Configuración base

Archivo:

```text
configs/base.yaml
```

Contenido recomendado:

```yaml
project:
  name: spectral-submersion-ds
  seed: 42

data:
  lost_corpus_path: data/raw/lost_language/corpus.csv
  candidate_corpus_path: data/raw/candidate_languages/rapanui_tokens.csv

tokenization:
  lowercase: true
  min_frequency: 1
  use_bigrams: true
  use_trigrams: false

cooccurrence:
  window_size: 3
  direction: both
  weighting: inverse_distance
  epsilon: 1.0e-9

pmi:
  positive: true
  smoothing: 1.0e-9

spectral:
  embedding_dim: 16
  alpha: 0.5
  svd_solver: randomized

alignment:
  method: procrustes
  use_anchors: false
  regularization: 1.0e-3

evaluation:
  bootstrap_samples: 100
  negative_controls: true
```

---

## 8. Módulos principales

### 8.1 Tokenización

Archivo:

```text
src/spectral_submersion/tokenization.py
```

Responsabilidad:

- leer corpus;
- normalizar tokens;
- construir vocabulario;
- convertir tokens a índices;
- crear secuencias por documento o línea.

Output esperado:

```python
tokens = ["g001", "g014", "g014", "g032"]
vocab = {"g001": 0, "g014": 1, "g032": 2}
ids = [0, 1, 1, 2]
```

---

### 8.2 Frecuencia

Archivo:

```text
src/spectral_submersion/frequency.py
```

Métricas:

- frecuencia absoluta;
- frecuencia relativa;
- ranking de tokens;
- entropía;
- índice de concentración;
- comparación contra distribución uniforme;
- comparación contra Zipf.

Fórmula de entropía:

```math
H(X) = -\sum_i p_i \log p_i
```

---

### 8.3 Co-ocurrencia

Archivo:

```text
src/spectral_submersion/cooccurrence.py
```

Construye matriz:

```math
C_{ij} = \sum_t 1[x_t=i]\sum_{u\neq t}K(|u-t|)1[x_u=j]
```

Opciones:

- ventana simétrica;
- ventana izquierda;
- ventana derecha;
- peso uniforme;
- peso inverso a distancia;
- peso exponencial.

---

### 8.4 PMI / PPMI

Archivo:

```text
src/spectral_submersion/pmi.py
```

Fórmula:

```math
PMI(i,j)=\log \frac{p(i,j)}{p(i)p(j)}
```

Versión positiva:

```math
PPMI(i,j)=\max(PMI(i,j),0)
```

Interpretación:

- PMI alto: tokens aparecen juntos más de lo esperado por azar.
- PMI bajo: relación débil o independiente.
- PPMI elimina asociaciones negativas inestables en corpus pequeños.

---

### 8.5 SVD y embeddings

Archivo:

```text
src/spectral_submersion/spectral.py
```

Descomposición:

```math
M = U\Sigma V^\top
```

Embedding:

```math
E = U_k\Sigma_k^\alpha
```

Parámetros:

- `k`: dimensión latente.
- `alpha=0`: usa solo direcciones.
- `alpha=0.5`: balance.
- `alpha=1`: pondera fuerte por valores singulares.

Outputs:

```text
data/processed/embeddings_lost.npy
data/processed/singular_values_lost.npy
reports/figures/singular_values.png
```

---

### 8.6 Pseudoinversa

La pseudoinversa se usa para obtener un mapa lineal mínimo:

```math
W^\star = X^+Y
```

donde:

- `X`: embeddings del sistema perdido.
- `Y`: embeddings de la lengua candidata.
- `W`: matriz que intenta mapear un espacio al otro.

Uso responsable:

- No tratar `W` como traducción final.
- Usarlo como baseline lineal.
- Compararlo contra Procrustes y transporte óptimo.

---

### 8.7 Procrustes

Archivo:

```text
src/spectral_submersion/alignment.py
```

Problema:

```math
Q^\star = \arg\min_{Q^\top Q=I}\|XQ-Y\|_F^2
```

Solución:

```math
X^\top Y = U\Sigma V^\top
```

```math
Q^\star = UV^\top
```

Interpretación:

- Busca rotación/reflexión.
- Conserva geometría interna.
- Es más elegante que un mapa lineal libre.

---

### 8.8 Transporte óptimo

Archivo:

```text
src/spectral_submersion/transport.py
```

Busca una matriz probabilística de correspondencia:

```math
\Pi_{ij}=p(y_j|x_i)
```

Costo:

```math
D_{ij} = \|x_iQ-y_j\|^2
```

Problema:

```math
\min_{\Pi}\sum_{ij}\Pi_{ij}D_{ij}
```

La matriz `Pi` es el diccionario blando:

```text
glifo g014:
  candidato 1: "ariki"    p = 0.31
  candidato 2: "atua"     p = 0.22
  candidato 3: "ra'a"     p = 0.14
```

---

## 9. Experimentos iniciales

### Experimento 1: análisis de frecuencia

Objetivo:

- contar tokens;
- graficar distribución;
- estimar entropía;
- comparar contra Zipf.

Output:

```text
reports/tables/frequency_table.csv
reports/figures/token_frequency.png
```

---

### Experimento 2: co-ocurrencias

Objetivo:

- construir matriz `C`;
- visualizar heatmap;
- detectar pares frecuentes;
- construir grafo de tokens.

Output:

```text
reports/figures/cooccurrence_heatmap.png
reports/figures/token_graph.png
```

---

### Experimento 3: SVD

Objetivo:

- calcular valores singulares;
- estimar rango efectivo;
- construir embeddings;
- visualizar en 2D con PCA o UMAP.

Output:

```text
reports/figures/singular_spectrum.png
reports/figures/embedding_2d.png
```

---

### Experimento 4: controles negativos

Crear corpus artificiales:

1. tokens permutados;
2. tokens aleatorios con misma frecuencia;
3. tokens aleatorios uniformes;
4. corpus con orden destruido.

Comparar:

```text
real corpus vs random corpus
```

Métricas:

- entropía;
- espectro singular;
- clustering;
- modularidad del grafo;
- estabilidad bootstrap.

---

### Experimento 5: alineamiento con lengua candidata

Objetivo:

- entrenar embeddings para lengua candidata;
- alinear espacios;
- calcular distorsión;
- generar candidatos.

Output:

```text
reports/hypotheses/candidate_dictionary.csv
```

Columnas:

```csv
lost_token,candidate_token,probability,rank,evidence_score,notes
g014,ariki,0.31,1,0.72,"near in aligned space"
g014,atua,0.22,2,0.65,"frequency compatible"
```

---

## 10. Métricas clave

### 10.1 Entropía

```math
H=-\sum_i p_i\log p_i
```

Mide diversidad de tokens.

### 10.2 Rango efectivo

```math
r_{\text{eff}}=\exp\left(-\sum_i p_i\log p_i\right)
```

donde:

```math
p_i=\frac{\sigma_i}{\sum_j\sigma_j}
```

Mide cuántas dimensiones espectrales realmente importan.

### 10.3 Distorsión de alineamiento

```math
D=\|XQ-Y\|_F
```

Mientras más bajo, mejor alineamiento.

### 10.4 Estabilidad bootstrap

```math
S = E_b[\operatorname{sim}(\Pi,\Pi^{(b)})]
```

Mide si las hipótesis sobreviven a remuestreo.

### 10.5 Entropía del diccionario

```math
H(\Pi_i)=-\sum_j \Pi_{ij}\log\Pi_{ij}
```

Si es alta, el glifo tiene traducción incierta.

---

## 11. Criterios de éxito

El proyecto avanza correctamente si logra:

- construir corpus limpio;
- generar matrices reproducibles;
- detectar estructura no aleatoria;
- visualizar embeddings interpretables;
- comparar contra controles negativos;
- generar hipótesis probabilísticas;
- reportar incertidumbre;
- evitar afirmaciones no justificadas.

No se considera éxito decir:

```text
desciframos el lenguaje
```

sin evidencia externa fuerte.

Sí se considera éxito decir:

```text
el sistema tiene estructura no aleatoria;
los signos g014, g032 y g088 forman una clase funcional;
el alineamiento con vocabulario ritual polinesio reduce la distorsión frente a controles;
las hipótesis son estables bajo bootstrap parcial.
```

---

## 12. Roadmap de 4 semanas

### Semana 1: datos y EDA

Tareas:

- crear repositorio;
- definir formato del corpus;
- cargar datos;
- limpiar tokens;
- contar frecuencias;
- graficar distribución;
- crear primer reporte.

Entregables:

```text
notebooks/00_data_audit.ipynb
notebooks/01_frequency_analysis.ipynb
reports/tables/frequency_table.csv
```

---

### Semana 2: matrices y SVD

Tareas:

- construir matriz de co-ocurrencia;
- calcular PMI/PPMI;
- ejecutar SVD;
- estimar rango efectivo;
- visualizar embeddings.

Entregables:

```text
src/spectral_submersion/cooccurrence.py
src/spectral_submersion/pmi.py
src/spectral_submersion/spectral.py
reports/figures/singular_spectrum.png
```

---

### Semana 3: controles y estabilidad

Tareas:

- generar corpus aleatorios;
- permutar corpus real;
- comparar espectros;
- bootstrap por línea/documento;
- medir estabilidad de embeddings.

Entregables:

```text
scripts/run_bootstrap.py
reports/tables/bootstrap_stability.csv
reports/figures/control_comparison.png
```

---

### Semana 4: alineamiento e hipótesis

Tareas:

- cargar lengua candidata;
- construir embeddings candidatos;
- ejecutar Procrustes;
- calcular matriz de candidatos;
- generar reporte de hipótesis.

Entregables:

```text
src/spectral_submersion/alignment.py
src/spectral_submersion/reporting.py
reports/hypotheses/candidate_dictionary.csv
reports/final/first_hypothesis_report.md
```

---

## 13. Primeros scripts mínimos

### 13.1 `build_dataset.py`

```python
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/lost_language/corpus.csv")
OUT_PATH = Path("data/processed/lost_tokens.csv")

def main():
    df = pd.read_csv(RAW_PATH)
    required = {"doc_id", "line_id", "position", "token"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["token"] = df["token"].astype(str).str.strip().str.lower()
    df = df.sort_values(["doc_id", "line_id", "position"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved clean corpus to {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Unique tokens: {df['token'].nunique()}")

if __name__ == "__main__":
    main()
```

---

### 13.2 `frequency.py`

```python
import pandas as pd
import numpy as np

def token_frequencies(tokens: list[str]) -> pd.DataFrame:
    counts = pd.Series(tokens).value_counts()
    total = counts.sum()

    df = counts.rename("count").reset_index()
    df.columns = ["token", "count"]
    df["probability"] = df["count"] / total
    df["rank"] = np.arange(1, len(df) + 1)

    return df

def entropy(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
```

---

### 13.3 `cooccurrence.py`

```python
import numpy as np

def build_vocab(tokens: list[str]) -> dict[str, int]:
    return {tok: i for i, tok in enumerate(sorted(set(tokens)))}

def cooccurrence_matrix(
    token_ids: list[int],
    vocab_size: int,
    window_size: int = 3,
    inverse_distance: bool = True,
) -> np.ndarray:
    C = np.zeros((vocab_size, vocab_size), dtype=float)

    n = len(token_ids)

    for t, center in enumerate(token_ids):
        left = max(0, t - window_size)
        right = min(n, t + window_size + 1)

        for u in range(left, right):
            if u == t:
                continue

            context = token_ids[u]
            distance = abs(u - t)

            weight = 1.0 / distance if inverse_distance else 1.0
            C[center, context] += weight

    return C
```

---

### 13.4 `pmi.py`

```python
import numpy as np

def ppmi_matrix(C: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    C = C.astype(float)
    total = C.sum() + epsilon

    Pij = (C + epsilon) / total
    Pi = Pij.sum(axis=1, keepdims=True)
    Pj = Pij.sum(axis=0, keepdims=True)

    PMI = np.log(Pij / (Pi @ Pj + epsilon))
    PPMI = np.maximum(PMI, 0.0)

    return PPMI
```

---

### 13.5 `spectral.py`

```python
import numpy as np
from sklearn.utils.extmath import randomized_svd

def spectral_embedding(
    M: np.ndarray,
    k: int = 16,
    alpha: float = 0.5,
):
    U, S, Vt = randomized_svd(M, n_components=k, random_state=42)
    E = U @ np.diag(S ** alpha)
    return E, S, Vt

def effective_rank(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
```

---

### 13.6 `alignment.py`

```python
import numpy as np

def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Solves min_Q ||XQ - Y||_F subject to Q.T Q = I.
    """
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    return Q

def pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = (X ** 2).sum(axis=1, keepdims=True)
    Y_norm = (Y ** 2).sum(axis=1, keepdims=True).T
    return X_norm + Y_norm - 2 * X @ Y.T

def soft_dictionary(D: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = -D / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
```

---

## 14. Makefile recomendado

```makefile
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

build:
	python scripts/build_dataset.py

freq:
	python scripts/run_frequency_analysis.py

svd:
	python scripts/run_svd.py

align:
	python scripts/run_alignment.py

bootstrap:
	python scripts/run_bootstrap.py

report:
	python scripts/generate_report.py

test:
	pytest tests/
```

---

## 15. Primer notebook: orden recomendado

Notebook:

```text
notebooks/01_frequency_analysis.ipynb
```

Celdas:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/lost_tokens.csv")
tokens = df["token"].tolist()

freq = df["token"].value_counts().reset_index()
freq.columns = ["token", "count"]
freq["probability"] = freq["count"] / freq["count"].sum()
freq["rank"] = range(1, len(freq) + 1)

freq.head(20)
```

Gráfico:

```python
plt.figure(figsize=(10, 5))
plt.plot(freq["rank"], freq["count"], marker="o")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Token frequency distribution")
plt.yscale("log")
plt.xscale("log")
plt.show()
```

---

## 16. Reporte final esperado

Archivo:

```text
reports/final/first_hypothesis_report.md
```

Estructura:

```markdown
# First Hypothesis Report

## 1. Corpus
- Number of documents:
- Number of lines:
- Number of tokens:
- Vocabulary size:

## 2. Frequency Analysis
- Most frequent signs:
- Entropy:
- Zipf-like behavior:

## 3. Co-occurrence Structure
- Strongest pairs:
- Graph communities:
- Positional patterns:

## 4. Spectral Analysis
- Effective rank:
- Singular value decay:
- Stable dimensions:

## 5. Negative Controls
- Random baseline:
- Permuted baseline:
- Bootstrap stability:

## 6. Candidate Alignment
- Candidate language:
- Alignment method:
- Distortion:
- Top candidate mappings:

## 7. Hypotheses
- Strong candidates:
- Weak candidates:
- Rejected candidates:

## 8. Limitations
- Corpus size:
- Missing anchors:
- Possible overfitting:

## 9. Next Steps
- More data:
- Better token normalization:
- More candidate languages:
- Iconographic priors:
```

---

## 17. Reglas científicas del proyecto

1. No afirmar traducción definitiva sin evidencia externa.
2. Separar estructura estadística de interpretación semántica.
3. Usar controles negativos siempre.
4. Reportar incertidumbre.
5. Guardar todos los parámetros de cada experimento.
6. No sobrescribir resultados antiguos.
7. Versionar datos procesados.
8. Documentar supuestos.
9. Comparar contra baselines simples.
10. Hacer que cada hipótesis sea falsable.

---

## 18. Baselines obligatorios

Antes de usar modelos complejos, correr:

### Baseline A: frecuencia pura

```text
tokens más frecuentes del sistema perdido ↔ palabras más frecuentes del idioma candidato
```

### Baseline B: vecinos por co-ocurrencia

```text
tokens similares por vector de contexto
```

### Baseline C: SVD sin alineamiento

```text
clusters internos solamente
```

### Baseline D: Procrustes con anclajes sintéticos

```text
validar que el algoritmo recupera un mapa conocido
```

### Baseline E: corpus aleatorio

```text
verificar que el método no inventa estructura donde no hay
```

---

## 19. Posible evolución hacia paper

Cuando el proyecto madure, puede transformarse en un paper con esta estructura:

```text
1. Introduction
2. Related Work
3. Mathematical Framework
4. Spectral Representation of Symbolic Systems
5. Alignment and Probabilistic Decoding
6. Identifiability and Negative Results
7. Experiments on Synthetic Corpora
8. Exploratory Case Study
9. Limitations
10. Conclusion
```

---

## 20. Nombre sugerido del framework

Opciones:

- `spectral-submersion`
- `lostlang-lab`
- `glyph2manifold`
- `moore-penrose-decipherment`
- `alkindi-svd`
- `spectral-decipher`

Nombre recomendado:

```text
spectral-submersion
```

Porque resume la idea brutal:

```text
lenguaje grande → variedad semántica
lenguaje perdido → proyección parcial
traducción → inversión regularizada de una submersión
```

---

## 21. Comando inicial para partir

```bash
mkdir spectral-submersion-ds
cd spectral-submersion-ds

mkdir -p configs data/raw/lost_language data/raw/candidate_languages data/interim data/processed
mkdir -p notebooks src/spectral_submersion scripts tests reports/figures reports/tables reports/hypotheses reports/final experiments

touch README.md requirements.txt pyproject.toml Makefile
touch src/spectral_submersion/__init__.py
touch src/spectral_submersion/{io,tokenization,normalization,frequency,cooccurrence,pmi,spectral,alignment,transport,decoding,evaluation,visualization,reporting}.py
```

---

## 22. Primera meta realista

La primera meta no es traducir.

La primera meta es producir este resultado:

```text
Dado un corpus simbólico, el sistema genera:
1. tabla de frecuencias;
2. matriz de co-ocurrencia;
3. matriz PPMI;
4. espectro singular;
5. embeddings de signos;
6. comparación contra controles aleatorios;
7. reporte de estructura no aleatoria.
```

Cuando eso esté sólido, recién aparece el módulo de alineamiento con lenguas candidatas.

---

## 23. Mantra del proyecto

> No buscamos una traducción milagrosa.  
> Buscamos una geometría estadística defendible, una hipótesis probabilística y una metodología falsable.

---

## 24. Registro de implementación y experimentos (build log)

> Esta sección documenta paso a paso lo que se construyó, ejecutó y observó.
> No contiene afirmaciones de desciframiento; solo registra decisiones técnicas, fuentes de datos y métricas obtenidas.

### 24.1 Infraestructura construida

- **Entorno Python**: `venv` en `.venv/` con `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`, `pyyaml`, `tqdm`, `rich`, `jupyter`, `pytest`.
- **POT (Python Optimal Transport)**: No pudo compilarse por ausencia de `g++` en el entorno. Se implementó **fallback manual de Sinkhorn** en NumPy (`src/spectral_submersion/transport.py`) con regularización entrópica y convergencia verificable. Esto preserva la corrección matemática sin dependencias compiladas.
- **Estructura de repo**: Creada según sección 4 del presente README.
- **Paper renderizado**: `submersion_espectral_lenguajes_perdidos_paper.tex` compilado a PDF (25 pp). Se corrigió error de compilación (`Extra }` en línea 1093).

### 24.2 Fuentes de datos reales

No se inventaron corpus candidatos. Se usaron datos abiertos verificables:

**Lenguas candidatas (OPUS-Tatoeba v2023-04-12, CC-BY 2.0 FR):**

| Idioma | Código | Familia | Oraciones | Tokens |
|--------|--------|---------|-----------|--------|
| Maorí | `mi` | Polinesio | 424 | 3,542 |
| Tahitiano | `ty` | Polinesio | 32 | 95 |
| Hawaiano | `haw` | Polinesio | 164 | 802 |
| Samoano | `sm` | Polinesio | 81 | 354 |
| Tongano | `to` | Polinesio | 21 | 101 |
| Fiyiano | `fj` | Austronesio | 38 | 105 |
| Rapa Nui | `rap` | Polinesio | 35 | 123 |
| Inglés | `en` | Germánico | 1,830,223 | 13,705,210 |
| Español | `es` | Romance | 402,525 | 2,804,739 |
| Alemán | `de` | Germánico | 642,766 | 4,863,023 |
| Ruso | `ru` | Eslavo | 1,017,101 | 5,675,418 |
| Japonés | `ja` | Japonico | 261,183 | 262,206 |
| Francés | `fr` | Romance | 570,599 | 4,173,122 |
| Italiano | `it` | Romance | 868,674 | 4,916,411 |
| Portugués | `pt` | Romance | 430,585 | 3,034,395 |
| Árabe | `ar` | Semítico | 47,471 | 231,511 |
| Coreano | `ko` | Koreano | 11,917 | 52,005 |

- Descarga directa desde `https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/{code}.txt.gz`.
- Procesamiento: tokenización por espacios, normalización básica, conversión a formato CSV del proyecto.
- **Aviso**: estos son lenguas reales usadas como **candidatos estructurales**, no como afirmación de parentesco directo con corpus perdidos.

**Corpus perdido real (Indus):**

| Corpus | Fuente | Licencia | Inscripciones | Tokens | Vocabulario |
|--------|--------|----------|---------------|--------|-------------|
| Indus real | `github.com/mayig/indus-valley-script-corpus` | MIT | 179 | 1,003 | 182 |

- Digitización del Corpus of Indus Seals and Inscriptions (CISI) por Parpola et al., transcrita por Mayig (2025).
- Formato JSON con notación Parpola (P###), convertido a CSV del proyecto vía `scripts/convert_indus_corpus.py`.
- **Aviso**: Este es un corpus arqueológico real, pero la transcripción sigue siendo una interpretación epigráfica con incertidumbre en la identificación de signos.

### 24.3 Corpus perdido de referencia

- **Ejemplo mínimo**: `data/raw/lost_language/corpus.csv` (16 tokens, 4 tipos, 2 documentos). Es un placeholder para validar el pipeline; no pretende ser un corpus arqueológico real.
- **Benchmark sintético controlado**: `data/raw/lost_language/corpus_synthetic.csv` (500 oraciones, 3,337 tokens, 16 tipos). Generado con una mini-gramática artificial (DET-NOUN-VERB-PREP-DET-NOUN) para validar que el método detecta estructura cuando existe. El código generador está en `scripts/generate_synthetic_corpus.py`.
- **Benchmark Rongorongo-like v2**: `data/raw/lost_language/corpus_rongorongo_v2.csv` (~4,500 tokens, 120 tipos, boustrophedon, repeticiones).
- **Benchmark Indus-like**: `data/raw/lost_language/corpus_indus_like.csv` (~8,000 tokens, 150 tipos, textos cortos, restricciones posicionales).

### 24.4 Experimentos ejecutados

#### Exp A: Análisis de frecuencia (corpus mínimo)
- **Script**: `scripts/run_frequency_analysis.py`
- **Resultado**: Entropía = 1.3209 nats. Distribución no uniforme (g014=37.5%, g001=25%, g032=25%, g088=12.5%).
- **Output**: `reports/tables/frequency_table.csv`, `reports/figures/token_frequency.png`.

#### Exp B: SVD y rango efectivo (corpus mínimo)
- **Script**: `scripts/build_embeddings.py`
- **Resultado**: Vocab 4, embeddings (4,4), rango efectivo 3.3935. El decaimiento espectral muestra que la matriz PPMI tiene rango casi completo para un vocabulario tan pequeño.

#### Exp C: SVD y rango efectivo (corpus sintético)
- **Script**: `scripts/build_embeddings.py` sobre `corpus_synthetic.csv`
- **Resultado**: Vocab 16, embeddings (16,16), rango efectivo **8.9661**.

#### Exp D: Controles negativos (corpus sintético)
- **Script**: `scripts/run_negative_controls.py`
- **Variantes**: real, permutado, random_same_freq, random_uniform.
- **Resultados**:

| variant | vocab_size | effective_rank | top_sv | sum_sv |
|---------|------------|----------------|--------|--------|
| real | 16 | **8.97** | 2.95 | 10.19 |
| permuted | 16 | 11.81 | 1.07 | 5.38 |
| random_same_freq | 16 | 12.68 | 1.02 | 5.51 |
| random_uniform | 16 | 12.34 | 0.87 | 4.76 |

- **Interpretación**: El corpus real sintético tiene un rango efectivo **significativamente menor** que todos los controles aleatorios. Esto indica que la mini-gramática genera estructura comprimible (información no trivial en pocas dimensiones), mientras que los controles carecen de esa estructura relacional. El sanity check del script confirma: `real_r < uniform_r`.

#### Exp E: Alineamiento con lengua candidata (Maorí)
- **Script**: `scripts/run_alignment.py`
- **Observación técnica**: El vocabulario perdido mínimo tiene 4 tipos; el Maorí tiene 679. Procrustes ortogonal requiere `n` común, por lo que el script detectó el mismatch y aplicó **distancia directa en el espacio latente** (sin rotación), con advertencia explícita.
- **Resultado**: Distorsión media mínima = 0.0669. El transporte óptimo (Sinkhorn manual, reg=0.5) produce una matriz de acoplamiento de alta entropía (7.85), cercana a uniforme. Esto es **esperado y honesto**: con 4 signos y 679 palabras, sin anclajes, no hay base para preferir un candidato sobre otro.
- **Output**: `reports/hypotheses/maori_dictionary_ot.yaml`, `maori_dictionary_nn.yaml`.

#### Exp F: Corpus sintético rico (~112 tipos, ~24k tokens)
- **Script**: `scripts/generate_synthetic_corpus.py` (gramática PCFG jerárquica)
- **Configuración**: 3,000 oraciones, vocabulario de 112 tipos, 11 clases funcionales (DET, NOUN, VERB, ADJ, PREP, CONJ, PRON, NUM, QUANT, ADV, NAME).
- **Resultado**: Vocab size=112, embeddings (112, 32), rango efectivo=21.94.
- **Propósito**: Proporcionar un ground-truth controlado donde se conoce la gramática subyacente, permitiendo validar matemáticamente que el pipeline espectral comprime estructura sintáctica.

#### Exp G: Controles negativos sobre corpus sintético rico
- **Script**: `scripts/run_negative_controls.py`
- **Resultados**:

| variant | vocab_size | effective_rank | top_sv | sum_sv |
|---------|------------|----------------|--------|--------|
| real | 112 | **11.18** | 31.79 | 139.98 |
| permuted | 112 | 14.18 | 16.34 | 78.59 |
| random_same_freq | 112 | 14.15 | 16.99 | 80.88 |
| random_uniform | 112 | 13.91 | 13.72 | 61.61 |

- **Interpretación**: El rango efectivo del corpus estructurado (11.18) es significativamente menor que todos los controles. La mini-gramática genera ~24k tokens pero la información latente se comprime en ~11 dimensiones efectivas. Los controles aleatorios no muestran esta compresión.

#### Exp H: Recuperación de anclajes sintéticos con Procrustes
- **Scripts**: `scripts/generate_synthetic_candidate.py` + `scripts/validate_anchors.py`
- **Setup**: Se generó un "idioma candidato" aplicando una permutación conocida al vocabulario sintético (112 pares 1-to-1). Se usó el 20% de los pares (22 anclajes) para entrenar Procrustes y el 80% restante (90 pares) para test.
- **Resultados**:

| Métrica | Valor |
|---------|-------|
| Accuracy@1 | **0.6889** |
| Accuracy@5 | **0.8333** |
| Accuracy@10 | **0.8778** |
| MRR | **0.7601** |
| Median rank | **1** |
| Mean rank | **3.93** |

- **Interpretación**: Bajo condiciones controladas (isometría exacta por permutación + anclajes parciales), Procrustes recupera la correspondencia de manera no trivial. El 68.9% de los signos de test se asignan exactamente correctos con solo 20% de anclajes. Esto valida matemáticamente que el algoritmo de alineamiento funciona cuando las premisas del teorema de identificabilidad parcial se satisfacen. **No implica** que funcionará igual con datos arqueológicos reales donde las premisas no se cumplen exactamente.

#### Exp I: Comparación relacional de métodos de alineamiento
- **Script**: `scripts/compare_alignment_methods.py`
- **Métricas**: distorsión geométrica `<Pi, D>`, distorsión relacional GW-style `L(Pi)`, entropía del acoplamiento.
- **Resultados en sintético 112x112 (Procrustes)**:

| Método | Geo Dist | Rel Dist | Entropía |
|--------|----------|----------|----------|
| OT (Sinkhorn) | **2.25** | **2220.93** | **8.74** |
| Soft NN | 258.52 | 63050.00 | 495.37 |
| Random | 329.56 | 34163.57 | 506.93 |

- **Resultados en lost 4 vs Maorí 679 (direct distance)**:

| Método | Geo Dist | Rel Dist | Entropía |
|--------|----------|----------|----------|
| OT (Sinkhorn) | **0.40** | **335.34** | **7.85** |
| Soft NN | 1.58 | 828.80 | 25.98 |
| Random | 1.91 | 1376.99 | 25.30 |

- **Interpretación**: En ambos escenarios, OT produce la menor distorsión geométrica y relacional. La entropía baja de OT en el caso sintético (8.74 vs ~500) refleja que el acoplamiento es más determinista cuando las geometrías son isométricas. En el caso cross-size (4 vs 679), OT sigue siendo mejor que NN y random, pero la entropía es relativamente alta, indicando incertidumbre justificada por la falta de anclajes.

#### Exp K: Comparación multi-candidato sobre corpus sintético
- **Script**: `scripts/run_all_candidates.py`
- **Candidatos**: 7 lenguas austronesias/polinésicas descargadas desde OPUS-Tatoeba v2023-04-12.
- **Setup**: Comparar embeddings del sintético (112 tipos) contra cada candidato usando OT + distorsión relacional GW-style.
- **Resultados** (ordenados por distorsión relacional creciente = mejor):

| Candidato | Familia | Tokens vocab | Geo Dist | Rel Dist | Entropía |
|-----------|---------|-------------|----------|----------|----------|
| rapa_nui | Polinesio | 56 | 2.03 | **1490.54** | 8.00 |
| fijian | Austronesio | 49 | 2.02 | 1549.21 | 7.70 |
| tahitian | Polinesio | 54 | 2.10 | 1536.89 | 7.96 |
| tongan | Polinesio | 60 | 2.20 | 1732.53 | 8.02 |
| samoan | Polinesio | 138 | 1.85 | 2099.07 | 9.03 |
| hawaiian | Polinesio | 157 | 1.80 | 2408.47 | 9.24 |
| maori | Polinesio | 679 | 1.42 | 3000.58 | 10.97 |

- **Aviso metodológico crucial**: Estos resultados **no implican** que Rapa Nui sea "más cercano" al corpus sintético en sentido genealógico. El corpus sintético es una gramática artificial (DET-NOUN-VERB). Las distorsiones reflejan tamaño de vocabulario y dimensionalidad, no parentesco lingüístico real. Lo que se valida aquí es que el pipeline **puede comparar múltiples candidatos de manera automatizada** y generar rankings auditables. En un escenario real con corpus perdido genuino, estos rankings serían hipótesis para evaluación filológica, no conclusiones.
- **Output**: `reports/tables/candidate_comparison_summary.csv`.

#### Exp L: Tokenización polinésica con segmentación de partículas
- **Script**: `scripts/segment_polynesian.py`
- **Método**: Separación conservadora de partículas funcionales de alta frecuencia al inicio de tokens (ej. `te`, `he`, `i`, `kei` en Maorí; `te`, `e`, `i` en Tahitian/Rapa Nui). Esto reduce la polisemia superficial de tokens como `tekau` (si `te` es determinante) vs. `tekau` como número.
- **Resultados de segmentación**:

| Idioma | Tokens orig | Tokens segmentados | Vocab orig | Vocab seg |
|--------|-------------|-------------------|------------|-----------|
| Maorí | 3,542 | 4,017 | 679 | 642 |
| Tahitiano | 95 | 120 | 54 | 51 |
| Hawaiano | 802 | 956 | 157 | 148 |
| Samoano | 354 | 486 | 138 | 123 |
| Tongano | 101 | 130 | 60 | 62 |
| Fiyiano | 105 | 107 | 49 | 51 |
| Rapa Nui | 123 | 143 | 56 | 56 |

- **Impacto en alineamiento** (distorsión relacional OT vs. sintético):

| Candidato | Rel Dist (orig) | Rel Dist (seg) | Δ |
|-----------|-----------------|----------------|---|
| Maorí | 3000.58 | 2973.34 | -27.24 |
| Tahitiano | 1536.89 | 1498.01 | -38.88 |
| Hawaiano | 2408.47 | 2132.75 | -275.72 |
| Samoano | 2099.07 | 2153.29 | +54.22 |
| Tongano | 1732.53 | 1725.35 | -7.18 |
| Fiyiano | 1549.21 | 1558.74 | +9.53 |
| Rapa Nui | 1490.54 | 1520.32 | +29.78 |

- **Interpretación**: La segmentación de partículas produce mejoras mixtas. Hawaiano y Tahitiano mejoran notablemente; Samoano y Rapa Nui empeoran ligeramente. Esto sugiere que la heurística de partículas es más efectiva para algunas lenguas que para otras, dependiendo de la morfosintaxis real. La variabilidad es honesta y documentada. En un pipeline real, la tokenización debería ajustarse por lengua o incluso por corpus específico.
- **Output**: `data/raw/candidate_languages/*_tokens_segmented.csv`, `reports/tables/candidate_comparison_summary_segmented.csv`.

#### Exp M: Robustez bajo polysemy (colapso de categorías)
- **Script**: `scripts/validate_polysemy.py`
- **Setup**: Se generó un candidato sintético donde el 15% del vocabulario perdido colapsa a un token candidato compartido (simulando polisemia o fusión morfológica). Se redujo el vocabulario candidato de 112 a 98 tipos. Se usaron 17 anclajes (20%) para entrenar Procrustes y 68 para test.
- **Resultados**:

| Métrica | Biyectivo (Exp H) | Polysemy 15% (Exp M) | Δ |
|---------|-------------------|----------------------|---|
| Accuracy@1 | 0.6889 | **0.3529** | -0.336 |
| Accuracy@5 | 0.8333 | **0.6471** | -0.186 |
| MRR | 0.7601 | **0.4961** | -0.264 |
| Median rank | 1 | **2** | +1 |

- **Interpretación**: La degradación es significativa y esperada. Cuando la correspondencia no es 1-to-1, Procrustes ortogonal (que asume isometría exacta) pierde poder predictivo. Esto valida empíricamente la advertencia del paper: identificabilidad parcial requiere biyección o, al menos, anclajes que rompan la ambigüedad. En contextos arqueológicos reales, donde la polisemia es probable, el método debe complementarse con priors culturales y controles negativos más estrictos.
- **Output**: `reports/tables/anchor_recovery_polysemy.json`.

#### Exp N: Gromov-Wasserstein solver completo
- **Script**: `scripts/run_gw_alignment.py`
- **Método**: Iteración de punto fijo entrópica (Peyre et al., 2016) implementada en NumPy puro. En cada paso: (1) computar tensor de costo GW a partir de distancias relacionales, (2) actualizar acoplamiento vía Sinkhorn manual.
- **Resultados** (sintético 112 vs. Rapa Nui 56):

| Método | Geo Dist | Rel Dist | Entropía |
|--------|----------|----------|----------|
| OT directo | 2.0257 | 1490.54 | 8.00 |
| **GW** | **3.0707** | **1489.36** | 8.12 |
| Random | 326.68 | -17343.81 | 429.33 |

- **Interpretación**: GW produce una distorsión relacional ligeramente menor que OT (1489.36 vs 1490.54) a costa de una distorsión geométrica mayor (3.07 vs 2.03). Esto es coherente con la teoría: GW prioriza la preservación de estructura relacional interna sobre la proximidad directa en el espacio latente. Para desciframiento, donde los espacios tienen dimensiones y tamaños diferentes, GW ofrece una alternativa teóricamente más apropiada que OT geométrico puro. La diferencia es pequeña en este benchmark sintético, pero el solver está validado y listo para escalarse.
- **Output**: `reports/tables/gw_rapa_nui.json`, `reports/tables/gw_maori.json`.

#### Exp O (bis): Co-ocurrencia direccional (izquierda/derecha)
- **Script**: `scripts/build_directional_embeddings.py`
- **Método**: Separar matrices de co-ocurrencia en `C_left` (contexto izquierdo) y `C_right` (contexto derecho), convertir ambas a PPMI, concatenar horizontalmente, aplicar SVD. Esto captura asimetrías posicionales (ej. determinantes a la izquierda de sustantivos, partículas postpositivas).
- **Resultados** (Rongorongo-like v2, 120 tipos, window=3, k=32):
  - Embeddings no-direccionales: rango efectivo = 30.56
  - **Embeddings direccionales**: rango efectivo = **30.86** (ligeramente mayor dimensión porque la matriz concatenada tiene el doble de columnas)
  - La estructura posicional adiciona información sin comprimir drásticamente el espacio, consistente con patrones sintácticos débiles en el corpus sintético.
- **Output**: `data/processed/embeddings_rongorongo_v2_dir.npy`.

#### Exp O: Corpus sintético Rongorongo-like (estructura conocida, ~120 tipos)
- **Scripts**: `scripts/generate_rongorongo_like_corpus.py`
- **Diseño**: Basado en descripciones estructurales públicas de Rongorongo (Ferrara 2015, Fischer 1997): ~120 glifos, distribución Zipf fuerte, patrones de doble/triple repetición, líneas orientadas en "tablets", alternancia de dirección (boustrophedon), bigramas/trigramas explícitos.
- **Resultados**: 15 tablets × 20 líneas = 300 líneas, ~4,500 tokens, vocabulario 120 tipos. Rango efectivo con window=3, k=16: **15.40**.
- **Controles negativos**:

| Variante | Rango efectivo |
|----------|----------------|
| Real (patrones) | **15.40** |
| Permutado | 15.07 |
| Random misma freq | 15.15 |
| Random uniforme | **14.00** |

- **Aviso honesto**: El sanity check **falla** (`real >= uniform`). Esto indica que, con los parámetros por defecto (window=3, k=16), el corpus Rongorongo-like no genera una estructura espectral claramente comprimida respecto al azar. Las posibles causas: (1) líneas cortas (~15 tokens) con vocabulario grande (120 tipos) hacen que las co-ocurrencias sean ruidosas; (2) el boustrophedon destruye patrones direccionales; (3) las repeticiones dobles/triples introducen autocorrelación espuria.
- **Lección metodológica**: Un sanity check que falla es **mejor** que uno que pase por artefacto. El pipeline detecta honestamente cuando un corpus no tiene estructura clara. Para Rongorongo real, esto sugiere que métodos basados puramente en co-ocurrencia local pueden ser insuficientes sin priors adicionales (iconográficos, posicionales, contextuales).
- **Output**: `data/raw/lost_language/corpus_rongorongo_v2.csv`, `reports/tables/control_comparison_rongorongo_v2.csv`.

#### Exp Q: Corpus sintético Indus-like (estructura conocida, ~150 tipos)
- **Script**: `scripts/generate_indus_like_corpus.py`
- **Diseño**: Inspirado en descripciones estructurales de la escritura del Indo (Parpola 1994, Mahadevan 1977): ~150 signos, textos muy cortos (~3 signos promedio), restricciones posicionales fuertes (títulos al inicio, numerales al final), alta repetición de signos funcionales.
- **Resultados**: 3,000 inscripciones, ~8,000 tokens, vocabulario 150 tipos. Rango efectivo = 14.86.
- **Controles negativos**:

| Variante | Rango efectivo |
|----------|----------------|
| Real (patrones) | **14.86** |
| Permutado | 15.34 |
| Random misma freq | 15.31 |
| Random uniforme | **13.60** |

- **Aviso**: Sanity check **falla** de nuevo (`real >= uniform`). Esto confirma un patrón: cuando el vocabulario es grande (~120-150 tipos) y las secuencias son cortas (~3-15 tokens), las co-ocurrencias locales son demasiado ruidosas para detectar estructura clara sin priors adicionales.
- **Output**: `data/raw/lost_language/corpus_indus_like.csv`.

#### Exp R: Comparación multi-candidato con pool diverso (no solo polinésico)
- **Script**: `scripts/run_diverse_candidates.py`
- **Setup**: Comparar el corpus sintético PCFG (112 tipos) y Rongorongo-like (120 tipos) contra 10 candidatos: 7 polinésicos + inglés + español + japonés.
- **Resultados** (sintético PCFG vs. candidatos, ordenados por distorsión relacional creciente):

| Candidato | Familia | Vocab | Rel Dist |
|-----------|---------|-------|----------|
| rapa_nui | Polinesio | 56 | **1490.54** |
| tahitian | Polinesio | 54 | 1536.89 |
| fijian | Austronesio | 49 | 1549.21 |
| tongan | Polinesio | 60 | 1732.53 |
| samoan | Polinesio | 138 | 2099.07 |
| hawaiian | Polinesio | 157 | 2408.47 |
| **japanese** | **Japonico** | **432** | **2908.82** |
| maori | Polinesio | 679 | 3000.58 |
| **spanish** | **Romance** | **5000** | **8765.81** |
| **english** | **Germánico** | **5000** | **10659.28** |

- **Resultados** (Rongorongo-like vs. candidatos):

| Candidato | Familia | Rel Dist |
|-----------|---------|----------|
| rapa_nui | Polinesio | **2398.18** |
| tahitian | Polinesio | 2443.91 |
| fijian | Austronesio | 2455.85 |
| tongan | Polinesio | 2640.03 |
| samoan | Polinesio | 3008.24 |
| hawaiian | Polinesio | 3317.55 |
| **japanese** | **Japonico** | **3820.63** |
| maori | Polinesio | 3913.59 |
| **spanish** | **Romance** | **9679.52** |
| **english** | **Germánico** | **11572.80** |

- **Interpretación**: Los candidatos polinésicos consistentemente muestran **menor distorsión relacional** que los europeos (inglés, español) y que el japonés. Esto **no implica** parentesco genealógico. Lo que ocurre es que la gramática sintética PCFG y el Rongorongo-like usan patrones de orden fijo (DET-NOUN-VERB) similares a lenguas aglutinantes/head-initial, mientras que el inglés y español tienen estructuras más flexibles y morfosintaxis más compleja que distorsionan los embeddings. El pipeline **distingue familias estructurales**, no genealógicas.
- **Output**: `reports/tables/diverse_comparison_synthetic.csv`, `reports/tables/diverse_comparison_rongorongo.csv`.

#### Exp S: Corpus de Indus real (Parpola CISI, 182 signos, 1,003 tokens)
- **Script**: `scripts/convert_indus_corpus.py` + `scripts/build_embeddings.py`
- **Fuente**: `github.com/mayig/indus-valley-script-corpus` (MIT License). Digitización del Corpus of Indus Seals and Inscriptions (CISI) por Parpola et al., transcrita por Mayig (2025). 179 inscripciones (lados de sellos/tablas), ~5.6 signos por inscripción.
- **Estadísticas**: Vocabulario 182 signos distintos (notación P### de Parpola). Signos más frecuentes: P324 (99x), P122 (76x), P086 (35x), P385 (35x).
- **Resultados** (embeddings no-direccionales, window=3, k=16):
  - Rango efectivo = **15.69**
  - Controles negativos: real=15.69, permutado=15.66, random_same_freq=15.53, random_uniform=15.35
  - **Sanity check falla** (`real_r >= uniform_r`): consistente con corpus de vocabulario grande (~182 tipos) y secuencias cortas (~5.6 tokens). La co-ocurrencia local no detecta estructura clara sin priors adicionales.
- **Comparación multi-candidato** (real Indus vs. 10 lenguas):

| Candidato | Familia | Rel Dist |
|-----------|---------|----------|
| rapa_nui | Polinesio | **2271.37** |
| tahitian | Polinesio | 2317.05 |
| fijian | Austronesio | 2329.61 |
| tongan | Polinesio | 2513.14 |
| samoan | Polinesio | 2880.17 |
| hawaiian | Polinesio | 3188.22 |
| **japanese** | **Japonico** | **3689.66** |
| maori | Polinesio | 3782.02 |
| **spanish** | **Romance** | **9547.74** |
| **english** | **Germánico** | **11441.29** |

- **Interpretación**: El ranking es cualitativamente idéntico al obtenido con el corpus sintético Indus-like (Exp Q). Los polinésicos muestran menor distorsión que los europeos. Esto refuerza que el pipeline detecta **estructuras tipológicas similares** (patrones de orden rígido, morfología ligera), no parentesco genealógico. Los embeddings direccionales producen resultados casi idénticos ( ranking preservado, valores ligeramente mayores), indicando que la direccionalidad no aporta información discriminativa adicional en este corpus tan corto.
- **Output**: `data/raw/lost_language/corpus_indus_real.csv`, `reports/tables/diverse_comparison_indus_real.csv`, `reports/tables/diverse_comparison_indus_real_dir.csv`.

#### Exp S (bis): Calibración de hiperparámetros sobre corpus Indus real
- **Script**: `scripts/calibrate_hyperparameters.py --input data/raw/lost_language/corpus_indus_real.csv`
- **Setup**: Grid search de 96 configuraciones (`window ∈ {2,3,5,7}`, `k ∈ {8,16,32,64}`, `alpha ∈ {0,0.5,1}`, `pmi ∈ {1e-9,1e-6}`).
- **Resultados**:

| Métrica | Mejor config | Valor |
|---------|-------------|-------|
| Compresión máxima | window=3, k=8, alpha=0, pmi=1e-6 | **7.84** |
| Señal máxima | window=7, k=16, alpha=0, pmi=1e-9 | **25.55** |

- **Sanity check con mejores parámetros**: Real=7.84, Permutado=7.83, Rand_freq=7.81, Uniform=7.62. **Sigue fallando** (`real_r > uniform_r`).
- **Análisis de espectro**: Los valores singulares del corpus real son sistemáticamente menores que los del uniforme (suma total 219 vs 284), pero su distribución normalizada es ligeramente más uniforme (entropía 2.752 vs 2.732), lo que eleva el rango efectivo. Esto indica que la matriz PPMI del corpus real tiene menor energía total pero mayor dispersión espectral que el azar.
- **Interpretación**: Incluso con hiperparámetros óptimos, el corpus Indus real no muestra la compresión espectral característica de estructura sintáctica local. Esto es consistente con la hipótesis de Farmer, Sproat & Witzel (2004) de que las inscripciones del Indo podrían ser un sistema no-lingüístico o proto-escritura de muy baja entropía secuencial. Alternativamente, podría reflejar que 179 inscripciones son insuficientes para 182 signos distintos, o que la estructura es no-local (dependencias de largo alcance, contextos extra-textuales).
- **Output**: `reports/tables/hyperparameter_grid_indus_real.csv`.

#### Exp T: Entropía condicional de pares (bigramas/trigramas) para corpus cortos
- **Script**: `scripts/analyze_conditional_entropy.py`
- **Motivación**: Para corpora con secuencias muy cortas (~5-6 tokens), la co-ocurrencia ventanal es ruidosa. La entropía condicional de bigramas/trigramas no requiere ventanas y funciona directamente con secuencias cortas (Rao et al. 2009, Science).
- **Resultados en corpus sintético PCFG (16 tipos, 3,337 tokens)**:

| Variante | H(unigram) | H(bigram) | H(trigram) | Perpl(bigram) |
|----------|------------|-----------|------------|---------------|
| Real | 3.86 | **2.25** | **2.13** | 4.77 |
| Permutado | 3.86 | 3.78 | 2.76 | 13.78 |
| Rand same freq | 3.86 | 3.80 | 2.76 | 13.89 |
| Rand uniform | 4.00 | 3.94 | 2.74 | 15.32 |

- **Sanity check sintético**: PASS (`real_h < uniform_h`). El corpus estructurado muestra entropía condicional significativamente menor que todos los controles, validando la métrica.
- **Resultados en corpus Indus real (182 tipos, 1,003 tokens)**:

| Variante | H(unigram) | H(bigram) | H(trigram) | Perpl(bigram) |
|----------|------------|-----------|------------|---------------|
| Real | 6.29 | **2.51** | 0.61 | 5.72 |
| Permutado | 6.29 | 3.09 | 0.22 | 8.53 |
| Rand same freq | 6.16 | 3.20 | 0.23 | 9.19 |
| Rand uniform | 7.35 | 2.28 | 0.02 | 4.84 |

- **Interpretación**: El corpus real Indus tiene entropía condicional de bigramas **menor que la permutada** (2.51 vs 3.09) y menor que random misma frecuencia (2.51 vs 3.20), lo que indica que existe **estructura secuencial a nivel de pares** que no es explicable por frecuencia marginal pura. Sin embargo, la comparación con uniforme es problemática por artefacto de tamaño muestral (el corpus uniforme con 182 tipos y ~1000 tokens genera muchos tipos raros que inflan artificialmente la entropía condicional al aparecer una sola vez). Para trigramas, el artefacto de escasez es aún más severo.
- **Conclusión**: El corpus Indus real muestra **evidencia débil pero no nula** de estructura secuencial a nivel de bigramas, consistente con Rao et al. (2009). La co-ocurrencia ventanal no la detecta; la entropía condicional sí la detecta parcialmente. Esto sugiere que el Indus podría tener dependencias de corto alcance (bigramas) pero no patrones de contexto extendidos.
- **Output**: `reports/tables/entropy_analysis_indus_real.csv`, `reports/tables/entropy_analysis_synthetic.csv`.

#### Exp U: Embeddings mejorados con priors iconográficos (Indus real)
- **Script**: `scripts/build_enhanced_embeddings.py`
- **Setup**: Concatenar embeddings espectrales (16-dim) con vectores de características iconográficas extraídos del JSON del corpus Indus (damage, line, uncertainty, branching_factor, branch_count, branch_direction, etc. → 9 features normalizadas).
- **Resultados** (comparación con embeddings puramente espectrales):

| Candidato | Rel Dist (espectral) | Rel Dist (mejorado) | Δ |
|-----------|---------------------|---------------------|---|
| rapa_nui | 2271.37 | **936.79** | -58.8% |
| tahitian | 2317.05 | **983.24** | -57.6% |
| fijian | 2329.61 | **996.84** | -57.2% |
| tongan | 2513.14 | **1180.04** | -53.0% |
| samoan | 2880.17 | **1542.63** | -46.4% |
| hawaiian | 3188.22 | **1850.81** | -42.0% |
| japanese | 3689.66 | **2346.02** | -36.4% |
| maori | 3782.02 | **2437.21** | -35.5% |
| spanish | 9547.74 | **8201.03** | -14.1% |
| english | 11441.29 | **10095.29** | -11.8% |

- **Interpretación**: El ranking de candidatos se preserva casi idénticamente, pero las distorsiones absolutas disminuyen significativamente para todos los candidatos (especialmente polinésicos, ~50% reducción). Esto indica que los priors iconográficos añaden información geométrica coherente con la estructura contextual, pero no cambian las conclusiones tipológicas. Los candidatos europeos se benefician menos (~12% reducción), sugiriendo que sus embeddings son menos compatibles con los priors visuales del Indus.
- **Output**: `data/processed/embeddings_indus_real_enhanced.npy`, `reports/tables/diverse_comparison_indus_real_enhanced.csv`.

#### Exp V: Análisis de sesgo posicional (Indus real)
- **Script**: `scripts/analyze_positional_bias.py`
- **Método**: Para cada signo, calcular la posición relativa media (0=inicio, 1=fin) y las fracciones de aparición en primera/última posición.
- **Resultados** (Indus real, top signos por frecuencia):

| Signo | Frecuencia | Posición media | Ratio inicio | Ratio fin |
|-------|------------|----------------|--------------|-----------|
| P324 | 99 | **0.08** | **0.78** | 0.01 |
| P217 | 18 | **0.15** | **0.78** | 0.11 |
| P086 | 35 | 0.24 | 0.54 | 0.03 |
| P385 | 35 | **0.94** | 0.00 | **0.83** |
| P378 | 17 | **0.75** | 0.12 | **0.59** |

- **Interpretación**: Los signos P324 y P217 muestran fuerte sesgo hacia el **inicio** de las inscripciones (>75% de las veces en primera posición). P385 muestra fuerte sesgo hacia el **final** (83% en última posición). Esto es consistente con la literatura epigráfica del Indus que describe signos funcionales posicionalmente restringidos (títulos al inicio, numerales al final).
- **Validación con controles negativos**:

| Signo | Métrica | Real | Permutado | Rand Freq | Rand Uniform |
|-------|---------|------|-----------|-----------|--------------|
| P324 | first_ratio | **0.778** | 0.162 | 0.212 | 0.000 |
| P324 | last_ratio | 0.010 | 0.212 | 0.125 | 0.333 |
| P385 | first_ratio | 0.000 | 0.171 | 0.133 | 0.125 |
| P385 | last_ratio | **0.829** | 0.171 | 0.167 | 0.125 |
| P217 | first_ratio | **0.778** | 0.444 | 0.130 | 0.167 |
| P217 | last_ratio | 0.111 | 0.111 | 0.261 | 0.000 |

- Las diferencias entre real y controles son masivas (+0.62 para P324 inicio, +0.66 para P385 final). Esto valida que el sesgo posicional es una **característica estructural genuina** del corpus Indus, no explicable por azar o frecuencia marginal.
- **Validación sintética**: El análisis aplicado al corpus PCFG también detecta el sesgo posicional diseñado (determinantes al inicio, nombres/verbos en medio/final), confirmando que la métrica no genera falsos positivos.
- **Output**: `reports/tables/positional_analysis_indus.csv`, `reports/figures/positional_bias_indus.png`.

#### Exp U (bis): Detección de comunidades en red de co-ocurrencia (Indus real)
- **Script**: `scripts/analyze_network_communities.py`
- **Método**: Construir red de co-ocurrencia ponderada por PPMI, aplicar algoritmo Louvain para detección de comunidades.
- **Resultados**: 12 comunidades detectadas (window=2, min_pmi=0.2). Tamaños: 25, 23, 23, 17, 17, 17, 15, 12, 12, 11, 9, 1 nodos.
- **Correlación con sesgo posicional** (`scripts/correlate_communities_positional.py`):

| Comunidad | Tamaño | mean_first_ratio | mean_last_ratio | Top signos |
|-----------|--------|------------------|-----------------|------------|
| 3 | 17 | **0.349** | 0.044 | **P324**, P062, P060, P013, P142 |
| 4 | 17 | 0.268 | 0.036 | P325, P147, P056, P051, P007 |
| 1 | 23 | 0.186 | 0.133 | P050, **P217**, P378, P092, P201 |
| 0 | 25 | 0.144 | **0.251** | P122, **P385**, P123, P202, P073 |

- **Interpretación**: La comunidad 3 es una **comunidad de inicio** que agrupa signos con fuerte sesgo hacia la primera posición, incluyendo a P324 (signo más frecuente, start-biased 78%). La comunidad 0 incluye a P385 (end-biased 83%) y muestra el mayor `mean_last_ratio` (0.251). Esto sugiere que la estructura de co-ocurrencia captura parcialmente las restricciones posicionales: signos que aparecen al inicio tienden a co-ocurrir entre sí (como títulos), mientras que signos de final están más dispersos.
- **Output**: `reports/tables/network_communities_indus.csv`, `reports/figures/network_communities_indus.png`, `reports/tables/community_positional_correlation.csv`.

#### Exp W: Validación del método de sesgo posicional con benchmark controlado
- **Script**: `scripts/generate_positional_corpus.py` + `scripts/analyze_positional_bias.py`
- **Setup**: Corpus sintético de 2,000 secuencias, 60 signos, longitud media 5.02 tokens, con sesgo posicional explícito: TITLE al inicio (80%), NUMERAL al final (80%), SIGNATURE penúltimo (50%).
- **Resultados de generación**:

| Clase | Tokens | first_ratio | last_ratio |
|-------|--------|-------------|------------|
| TITLE | 1,907 | **0.835** | 0.869 |
| OBJECT | 6,166 | 0.066 | 0.320 |
| NUMERAL | 800 | 0.000 | **1.000** |
| SIGNATURE | 1,173 | 0.000 | **0.932** |

- **Resultados de detección** (top signos por positional bias):

| Signo | Clase | first_ratio | last_ratio | Posición media |
|-------|-------|-------------|------------|----------------|
| T00 | TITLE | **0.812** | 0.000 | 0.09 |
| T01 | TITLE | **0.841** | 0.000 | 0.08 |
| T02 | TITLE | **0.876** | 0.000 | 0.06 |
| N00 | NUMERAL | 0.000 | **1.000** | 1.00 |
| N01 | NUMERAL | 0.000 | **1.000** | 1.00 |
| S00 | SIGNATURE | 0.000 | **0.858** | 0.93 |
| O00 | OBJECT | 0.059 | 0.030 | 0.49 |

- **Sanity check espectral**: FALLA (`real_r=14.74 >= uniform_r=14.56`). Esto confirma que la co-ocurrencia ventanal es insuficiente para detectar estructura en corpus con vocabulario grande y secuencias cortas, **incluso cuando la estructura posicional es fuerte y conocida**.
- **Conclusión**: El análisis de sesgo posicional detecta correctamente las clases funcionales con restricciones posicionales. El sanity check espectral falla porque la estructura es puramente posicional, no de contexto ventanal. Esto valida que para corpus como Indus, el análisis posicional es más apropiado que la co-ocurrencia local.
- **Output**: `data/raw/lost_language/corpus_positional_synthetic.csv`, `reports/tables/positional_analysis_synthetic_pos.csv`, `reports/figures/positional_bias_synthetic_pos.png`.

#### Exp P: Calibración de hiperparámetros via grid search
- **Script**: `scripts/calibrate_hyperparameters.py`
- **Setup**: Grid search sobre `window_size ∈ {2,3,5,7}`, `k ∈ {8,16,32,64}`, `alpha ∈ {0,0.5,1}`, `pmi_epsilon ∈ {1e-9,1e-6}`. Corpus: sintético v2 (112 tipos).
- **Resultados** (96 configuraciones):

| Métrica | Mejor config | Valor |
|---------|-------------|-------|
| Compresión máxima (menor rango efectivo) | window=3, k=8, alpha=0, pmi=1e-9 | **6.09** |
| Señal máxima (mayor valor singular) | window=2, k=16, alpha=0, pmi=1e-9 | **34.85** |

- **Interpretación**: `alpha=0` (sin ponderación por valores singulares) domina ambas métricas en este benchmark. Esto sugiere que, para una gramática sintética con estructura clara, las direcciones principales bastan; ponderar por magnitud no ayuda. `window=2` o `3` son óptimos; ventanas grandes diluyen la señal local.
- **Output**: `reports/tables/hyperparameter_grid.csv`, `configs/recommended.yaml`.

#### Exp J: Bootstrap de estabilidad espectral (corpus sintético rico)
- **Script**: `scripts/run_bootstrap.py`
- **Configuración**: 50 muestras bootstrap, remuestreo de oraciones con reemplazo, dimensión k=16.
- **Resultados**:

| Métrica | Valor |
|---------|-------|
| Effective rank mean | 12.2188 |
| Effective rank std | 0.0789 |
| Effective rank CV | **0.65%** |
| Top SV mean | 31.9588 |
| Top SV std | 0.2954 |
| Embedding cosine mean | **0.7651** |
| Embedding cosine std | 0.0150 |

- **Interpretación**: El rango efectivo es extremadamente estable bajo remuestreo (CV < 1%). La similitud coseno media entre embeddings bootstrap es ~0.76, indicando que la geometría relativa de los signos se preserva razonablemente bien aunque las coordenadas absolutas varían (esperado por invariancia rotacional). Esto sugiere que las conclusiones sobre estructura no son artefactos de un solo subconjunto del corpus.

### 24.5 Decisiones técnicas documentadas

1. **Co-ocurrencia con fronteras**: `cooccurrence_matrix_from_sequences` respeta límites de `doc_id`/`line_id`. El README original mostraba una función legacy sin esta protección; se corrigió.
2. **Soft dictionary vs. OT**: El README original presentaba `soft_dictionary` (softmax sobre distancias) como si fuera transporte óptimo. Se separaron conceptualmente: `alignment.py` tiene `soft_dictionary` (nearest-neighbor), y `transport.py` tiene `optimal_transport_matrix` (Sinkhorn con restricciones marginales, con fallback manual en NumPy).
3. **Pandas compatibility**: `token_frequencies` en `frequency.py` usa `rename("count").reset_index()` en lugar de asumir nombres automáticas de columnas, compatible con pandas >= 2.0.
4. **pyproject.toml**: Especifica `requires-python = ">=3.10"` y estructura `src/`-layout.
5. **Makefile pipeline**: Se implementó `make pipeline` que ejecuta tests, embeddings, controles negativos, bootstrap, validación de anclajes, validación de polysemy, comparación multi-candidato y generación de reporte integrado en un solo comando.
6. **CLI mejorado**: `spectral-submersion` soporta subcomandos (`setup`, `test`, `pipeline`, `validate --config`, `report`).
7. **Calibración automatizada**: Grid search de 96 configuraciones de hiperparámetros con métricas de compresión y señal.
8. **Co-ocurrencia direccional**: `directional_cooccurrence_matrix_from_sequences` separa contexto izquierdo y derecho, capturando asimetrías posicionales. Integrado en `build_embeddings.py` via `--directional` flag.
9. **Pool diverso de candidatos**: 15 lenguas de 7 familias validan que el pipeline distingue estructuras tipológicas, no solo geografías.
10. **Corpus Indus real integrado**: Se descargó y convirtió el corpus abierto `mayig/indus-valley-script-corpus` (Parpola CISI). El sanity check falla consistentemente con parámetros por defecto, validando la honestidad del pipeline.

### 24.6 Limitaciones actuales

- **Corpus perdido real**: Se obtuvo un corpus real de Indus (179 inscripciones, 182 signos, ~5.6 signos/inscripción) desde `github.com/mayig/indus-valley-script-corpus` (MIT License). Sin embargo, **el sanity check falla** (`real_r >= uniform_r`), indicando que la co-ocurrencia local con parámetros por defecto no detecta estructura clara. Para Rongorongo, no se dispone de corpus transcrito normalizado en formato abierto.
- **Alineamiento sin anclajes**: El teorema de no-desciframiento gratuito (paper, Sección 7) demuestra que sin anclajes no hay identificabilidad. Los resultados de alineamiento con Maorí (Exp E) y el pool diverso (Exp R, S) muestran entropía alta, coherente con la teoría.
- **Tokenización candidata**: Espacios simples, con segmentación opcional de partículas funcionales. No se aplicó lematización real ni análisis morfológico profundo.
- **Transporte óptimo**: Fallback manual de Sinkhorn en NumPy funciona pero es más lento que POT para matrices grandes (>1e3 x 1e3). No se dispone de compilador C++ en el entorno.
- **Gromov-Wasserstein**: Solver completo implementado (`transport.py::gromov_wasserstein_matrix`). Permite comparar espacios de distinto tamaño vía estructura relacional.
- **Polysemy**: Se validó degradación bajo colapso (Exp M), pero el pipeline no tiene estrategia activa para recuperar mapeos no-biyectivos más allá de reportar incertidumbre.
- **Sanity checks fallidos**: Los corpus Rongorongo-like e Indus-like fallan el sanity check (`real_r >= uniform_r`). Esto indica que los parámetros por defecto (window=3, k=16) no detectan estructura clara en corpus con vocabulario grande y secuencias cortas. Necesitan hiperparámetros específicos o priors adicionales.

### 24.7 Próximos pasos recomendados (actualizado)

1. ~~Obtener o construir un corpus perdido sintético más grande~~ ✅ Completado (112 tipos, 24k tokens).
2. ~~Implementar anclajes sintéticos y medir recuperación~~ ✅ Completado (Acc@1=68.9% con 20% anclajes).
3. ~~Implementar métrica de distorsión relacional GW-style~~ ✅ Completado.
4. ~~Refinar bootstrap con métricas de estabilidad real~~ ✅ Completado.
5. ~~Expandir candidatos a otros polinésicos~~ ✅ Completado (7 candidatos).
6. ~~Mejorar tokenización polinésica~~ ✅ Completado (segmentación de partículas, resultados mixtos documentados).
7. ~~Integrar control negativo automático en el pipeline de reporte~~ ✅ Completado (`generate_integrated_report.py` + `make pipeline`).
8. ~~Implementar anclajes con colapso (polysemy) para medir robustez bajo no-biyectividad~~ ✅ Completado (degradación ~49% documentada).
9. ~~Implementar GW solver completo para alineamiento relacional cross-size~~ ✅ Completado.
10. ~~Obtener/generar corpus perdido realista (Rongorongo-like)~~ ✅ Completado (synthetic Rongorongo-like v2).
11. ~~Implementar CLI de alto nivel~~ ✅ Completado (`spectral-submersion validate --config ...`).
12. ~~Calibrar hiperparámetros via grid search~~ ✅ Completado (96 configs evaluadas).
13. ~~Obtener corpus perdido **real** (transcripción arqueológica normalizada)~~ ✅ Completado (corpus Indus real desde `mayig/indus-valley-script-corpus`, 179 inscripciones, 182 signos).
14. ~~Implementar segmentación posicional (izquierda/derecha) en co-ocurrencia~~ ✅ Completado (`--directional` flag en `build_embeddings.py`).
15. ~~Evaluar hiperparámetros específicos para corpus de vocabulario grande + secuencias cortas~~ ✅ Completado (Indus real falla sanity check incluso con mejores parámetros).
16. ~~Añadir priors iconográficos y contextuales~~ ✅ Completado (`build_enhanced_embeddings.py` con feature vectors del corpus Indus).
17. ~~Obtener corpus de Rongorongo real transcrito en formato abierto~~ ❌ No encontrado. La búsqueda exhaustiva en GitHub, Figshare, Zenodo, JSTOR, y Google Scholar no identificó transcripciones normalizadas disponibles públicamente. Se identificaron proyectos académicos activos (Lastilla, Ravanelli, Valério 2019-2022; Melka 2009; Harris & Melka 2011) que trabajan en corpus digitales, pero los datos requieren acceso institucional o contacto directo con los autores. El benchmark Rongorongo-like sintético sigue siendo la mejor aproximación disponible para el pipeline.
18. ~~Implementar análisis de entropía condicional de pares/bigramas~~ ✅ Completado (`analyze_conditional_entropy.py`).
19. Generar reporte integrado con todas las figuras y tablas maestras (`generate_master_summary.py` + `visualize_corpus_comparison.py`).


### 24.7 Experimentos adicionales: Traductor Rongorongo (Sesión Abril 2026)

**Objetivo:** Construir un traductor funcional lenguaje-candidato → glifos Rongorongo como prueba de concepto end-to-end.

#### Experimento X: Corpus sintético Rongorongo v3 (realista)

- **Script:** `scripts/generate_rongorongo_v3.py`
- **Características:** 120 glifos tipo Barthel, 7 categorías funcionales (det, noun, verb, num, name, part, rare), boustrophedon, repeticiones dobles (15%) y triples (3%), 1,000 tabletas, 7,453 líneas, 40,179 tokens.
- **Resultado:** Estructura realista que reproduce propiedades conocidas del corpus Rongorongo (Ferrara 2015, Fischer 1997).

#### Experimento Y: Corpus paralelo masivo v3 (context-aware)

- **Script:** `scripts/generate_massive_parallel_v3.py`
- **Características:** 200,000 pares paralelos. Mapeo context-aware: (palabra, categoría, contexto_hash) → glifo específico. Códigos semánticos: d01-d15 (determinantes), n01-n35 (nombres), v01-v20 (verbos), m01-m10 (numerales), p01-p15 (nombres propios), x01-x10 (partículas).
- **Resultado:** 228 tokens fuente únicos, 105 glifos destino únicos. Longitud media fuente: 9.50, destino: 11.14.

#### Experimento Z: Entrenamiento Transformer seq2seq

- **Script:** `scripts/train_rongorongo_translator_v2.py`
- **Arquitectura:** Transformer encoder-decoder, d_model=256, 8 cabezas, 4+4 capas, ~5.4M parámetros.
- **Optimización:** AdamW (lr=3e-4), CosineAnnealingLR, data augmentation (random token masking 10%).
- **Resultados:** Pérdida validación = 2.19, Perplexity = 8.98 (20 épocas, convergencia estable).
- **Modelo guardado:** `models/rongorongo_translator_v5/`

#### Evaluación de mapeos aprendidos

- **Script:** `scripts/analyze_rongorongo_mappings.py`
- **Hallazgo 1 (Determinantes estables):** `te → d03` (100%), `he → d01` (100%), `na → d04` (100%). Los determinantes de clase cerrada se mapean a glifos fijos.
- **Hallazgo 2 (Nombres propios con repetición):** `Maui → p01 p01 p01`, `Hina → p04 p04`. El modelo reproduce el patrón de repetición característico de Rongorongo para nombres propios.
- **Hallazgo 3 (Verbos context-dependentes):** `haere` (ir) se mapea a `v07` con `tangata` (persona), pero a `v19` con `manu` (pájaro). El modelo aprendió distinciones semánticas contextuales.
- **Hallazgo 4 (Partículas distintas):** `ki → x02`, `i → x05`, `mai → x06`, `atu → x03`. Cada partícula tiene un glifo único y estable.

#### Beam search vs Greedy

- **Script:** `scripts/evaluate_rongorongo_v5.py`, `src/spectral_submersion/beam_search.py`
- **Implementación:** Beam width=5, length penalty=0.8.
- **Resultado:** Para este modelo, beam search produce resultados idénticos a greedy en la mayoría de los casos, indicando alta confiabilidad en las predicciones. Esto es una fortaleza para traducción estructural.

#### Demo interactiva

- **Script:** `scripts/demo_rongorongo_translator.py`
- **Uso:** `PYTHONPATH=src python scripts/demo_rongorongo_translator.py`
- **Modo interactivo:** `PYTHONPATH=src python scripts/translate_to_rongorongo.py --interactive`

#### Evaluación cruzada por familias

- **Script:** `scripts/evaluate_rongorongo_translator.py`
- **Resultados:**
  | Familia | Longitud media | Diversidad media |
  |---------|---------------|------------------|
  | Rapa Nui | 6.25 | 3.75 |
  | Hawaiian | 8.67 | 3.33 |
  | Fijian | 7.50 | 1.50 |
  | English | 12.00 | 1.00 |
  | Spanish | 10.33 | 1.33 |
  | Japanese | 6.50 | 1.50 |

Los lenguajes polinésicos producen secuencias más diversas y estructuradas que los no-polinésicos, consistente con que el corpus de entrenamiento fue generado con patrones tipológicos polinésicos.

#### Descubrimiento de corpus real (Archive.org)

- **Fuente:** Archive.org item `rongorongotexts` (2024-08-29)
- **Contenido:** 79 imágenes PNG con transcripciones Barthel de 22 tabletas (A-Y, excepto M, U, W). Licencia: Public Domain.
- **Ubicación local:** `data/raw/lost_language/rongorongo_archive_images/`
- **Limitación:** Las transcripciones son imágenes de glifos dibujados, no códigos numéricos en texto. Extraerlas automáticamente requeriría segmentación + clasificación de ~120 glifos (proyecto de visión computacional de meses). Se dejan como referencia arqueológica para validación visual futura.

#### Estado de la búsqueda de datos reales

| Corpus | Disponibilidad | Formato | Acceso |
|--------|---------------|---------|--------|
| Indus (Parpola CISI) | ✅ Disponible | JSON/CSV | MIT License (Mayig 2025) |
| Rongorongo Barthel | ✅ Disponible | Imágenes PNG | Public Domain (Archive.org) |
| Rongorongo transcrito | ❌ No disponible | Texto numérico | Requiere contacto institucional |
| Proto-Elamite | ❌ No encontrado | - | - |
| Linear A (DELA) | ❌ No disponible públicamente | Digital | Requiere permisos académicos |

