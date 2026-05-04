# Roadmap

Este roadmap organiza el proyecto para que pueda crecer sin perder rigor metodológico.

## Fase 0: lanzamiento público

- Publicar post de LinkedIn con claims conservadores.
- Crear Discord con canales, reglas y onboarding.
- Publicar `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `DATA_ACCESS.md` y licencia.
- Crear una lista inicial de `good first issue`.
- Preparar primer paquete de datos descargable.

## Fase 1: datos reproducibles

- Auditar fuentes, licencias y procedencia.
- Mantener `DATA_WISHLIST.md` con datos públicos, privados y restringidos por reunir.
- Crear issues para leads de datos Rongorongo sin subir archivos no autorizados.
- Separar datos pequeños de artefactos pesados.
- Generar `manifest.csv` y `checksums.sha256`.
- Subir release `v0.1-public-launch` a Google Drive.
- Documentar qué archivos son generados y qué scripts los reproducen.

## Fase 2: pipeline robusto

- Consolidar targets de Makefile que sí corren de extremo a extremo.
- Reducir scripts duplicados o experimentales.
- Agregar smoke tests para los comandos principales.
- Versionar configs de experimentos.
- Registrar seeds, commit SHA y ambiente por run.

## Fase 3: anclaje icónico inverso

- Reconstruir `world-rapanui-1500` con referentes, fuentes y licencias.
- Validar encoders visuales primero en escrituras descifradas.
- Mejorar el benchmark cross-script hasta superar el umbral C2.5 definido en el paper.
- Separar candidatos visuales de claims iconográficos admitidos.
- Publicar ledger de hipótesis con evidencia, contraevidencia y nivel de claim.

## Fase 4: validación científica

- Reproducir controles negativos principales.
- Definir umbrales de claim en docs y tests.
- Separar señales estructurales de hipótesis semánticas.
- Revisar resultados con expertos del dominio.
- Convertir hallazgos en issues rastreables.

## Fase 5: comunidad e innovación

- Sesiones semanales de lectura o replicación.
- Grupos de trabajo: datos, métodos, ingeniería, paper.
- Publicar notas de reuniones y decisiones.
- Crear datasets/model cards cuando corresponda.
- Evaluar publicación archivada en Zenodo.

## Principio rector

El proyecto debe ser interesante sin ser sensacionalista: mejor una hipótesis auditable que una traducción espectacular imposible de defender.
