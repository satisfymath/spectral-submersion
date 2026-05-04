# Llamada a la Comunidad Científica

## Spectral Submersion: un marco auditable para estudiar sistemas simbólicos no descifrados

---

## Qué es este proyecto

**Spectral Submersion** es un proyecto de investigación de código abierto que aplica métodos matemáticos rigurosos al análisis de sistemas simbólicos de bajo recurso: Rongorongo, escritura del Indo, corpus sintéticos de control y otros casos donde la evidencia es fragmentaria.

La posición metodológica del proyecto es deliberadamente conservadora: no afirmamos desciframientos definitivos sin evidencia externa. Producimos hipótesis estadísticas auditables, con incertidumbre explícita, controles negativos y rutas claras de falsación.

### Enfoque

- **Estadística estructural**: frecuencia, co-ocurrencia, entropía, valores singulares
- **Geometría latente**: PMI/PPMI, SVD, embeddings espectrales
- **Alineamiento**: Procrustes, transporte óptimo, Gromov-Wasserstein
- **Auditabilidad**: controles negativos, bootstrap, niveles de claim y reportes reproducibles

### Resultados clave

| Corpus | Finding |
|--------|---------|
| Rongorongo | Señales estructurales que merecen revisión experta y reproducción independiente |
| Indus | Estructura limitada sin priors adicionales; se requiere mejor cobertura de datos |
| Sintéticos | Benchmarks útiles para validar que el pipeline detecta señales conocidas |

### Estado científico actual

El paper técnico actual deja una restricción importante: sin anclas externas, Rongorongo no debe subir de C0-C1. La ruta responsable para avanzar es el **anclaje icónico inverso**:

1. reconstruir un conjunto verificable de referentes del mundo Rapa Nui;
2. comparar glifos con referentes visuales mediante embeddings;
3. validar primero en escrituras conocidas;
4. bloquear claims si la validación externa falla.

La implementación actual genera candidatos iconográficos auditables para revisión, pero todavía no autoriza claims C2.5 porque la validación cross-script queda por debajo del umbral definido en el paper. Ese bloqueo es una virtud del marco: evita convertir rankings visuales en significado.

---

## Cómo puedes contribuir

### Investigación
- Revisión de metodología estadística
- Propuestas de nuevos métodos de alineamiento
- Validación de resultados en otros corpora
- Lectura crítica del paper y de los niveles de claim

### Desarrollo
- Optimización de código (SVD, Sinkhorn)
- Nuevos módulos de análisis de imagen para glifos
- Integración con bases de datos de glifos existentes
- Empaquetado reproducible de experimentos
- Validación de encoders visuales en escrituras descifradas

### Datos
- Curación de transcripciones y metadatos
- Revisión de licencias y procedencia
- Nuevas lenguas candidatas y corpus comparables
- Paquetes de datos descargables con checksums
- Referentes visuales del mundo Rapa Nui con bibliografía y licencia
- Búsqueda de datos Rongorongo públicos, privados o restringidos, siempre con permisos claros

### Comunidad
- Moderación del Discord
- Onboarding de colaboradores
- Notas de reuniones, roadmap y seguimiento de issues

### Retroalimentación
- Revisión de paper técnico
- Sugerencias de métricas
- Conexión con expertos del dominio

---

## Guías de contribución

1. **Fork** el repositorio
2. Crea una **branch** para tu feature: `git checkout -b feature/nueva-metrica`
3. Agrega **tests** para nueva funcionalidad
4. Asegúrate que corran: `make test`
5. Pull Request con descripción clara

Ver `CONTRIBUTING.md` para detalles.

---

## Discord y coordinación

La comunidad se organiza en Discord para conversación diaria, lectura de papers, soporte a nuevos contribuidores y grupos de trabajo: https://discord.gg/ueg94XVw

GitHub seguirá siendo la fuente de verdad para código, issues, PRs, decisiones técnicas y releases.

Canales recomendados:

- `#start-here`: reglas, links y primeros pasos
- `#announcements`: releases, reuniones y decisiones
- `#research-methods`: estadística, SVD, transporte óptimo, controles
- `#rongorongo`: datos, fuentes, transcripción y contexto Rapa Nui
- `#data-curation`: licencias, manifest, checksums y Google Drive
- `#engineering`: pipeline, tests, reproducibilidad y performance
- `#paper-review`: revisión metodológica y citas
- `#results-replication`: reproducir experimentos y reportar divergencias

Ver `DISCORD_SETUP.md` para la estructura completa.

---

## Datos descargables

Los datos pesados, modelos y artefactos de experimentos deben publicarse fuera del repositorio en un paquete versionado, idealmente en Google Drive, Zenodo o un release externo equivalente. El repo debe incluir manifiestos, checksums e instrucciones de reproducción.

Carpeta Drive actual: https://drive.google.com/drive/u/1/folders/1vbnpyyy-YtVYyd-gf_-XwvHLHqIgQq8h

Ver `DATA_ACCESS.md` para la propuesta de estructura. Antes de publicar datos ampliamente, revisar manifest, fuentes, licencias y permisos.

Para entender el estado técnico actual del paper y del plan de anclaje icónico, ver `RESEARCH_STATE.md`.

### Búsqueda comunitaria de datos Rongorongo

El cuello de botella central del proyecto es la falta de datos suficientes y verificables. Invitamos a la comunidad a ayudar a encontrar, catalogar y solicitar acceso a:

- fotografías de alta resolución de tablillas y objetos Rongorongo;
- transcripciones Barthel, variantes modernas y metadata paleográfica;
- modelos 3D, fotogrametría o escaneos de museos;
- corpus RR-corpus/Lastilla/Ravanelli/Valério u otros materiales institucionales;
- catálogos de museos, procedencia, dimensiones, estado de conservación y orientación;
- petroglifos y referentes visuales Rapa Nui con licencia clara;
- corpus Rapanui y polinésicos comparables para contexto cultural.

No se deben subir datos privados o restringidos al repo, Discord o Drive público sin permiso explícito. Si encuentras una fuente, abre un issue usando la plantilla de data source y documenta solo metadata, contacto, enlace y estado de licencia.

Ver `DATA_WISHLIST.md`.

---

## Nota personal

Además de la motivación científica, existe una motivación humana detrás del proyecto. Está documentada en `WHY_THIS_EXISTS.md`. Esa nota no forma parte de la evidencia ni de los claims del paper; es el norte personal que explica por qué vale la pena construir esto con paciencia.

---

## Proyectos relacionados a conectar

- **Lackadaisical-Security/rongorongo-deciphered-public** — 740 glifos Rongorongo con lexicon
- **mayig/indus-valley-script-corpus** — Corpus Indus real (MIT)
- **OPUS-Tatoeba** — Lenguas candidatas (CC-BY 2.0 FR)

---

## Contacto

- **Autor**: David Alexander Astudillo Muñoz
- **Email**: david@baskerville.ai
- **GitHub**: @davidastudillo
- **LinkedIn**: davidastudillo

---

## Disclaimer

Este proyecto **no afirma haber descifrado** ningún lenguaje. Genera hipótesis estadísticas que requieren validación expertal. Todo resultado incluye incertidumbre documentada.

El objetivo es construir herramientas auditables que **ayuden** a la comunidad científica a explorar hipótesis, no reemplazarla.

---

*"No buscamos una traducción milagrosa. Buscamos una geometría estadística defendible y una metodología falsable."*
