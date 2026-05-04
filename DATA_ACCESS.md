# Acceso a Datos y Plan de Release

Este repositorio contiene código, documentación y artefactos seleccionados. Para un lanzamiento público profesional, los datos grandes, modelos entrenados y resultados generados deben compartirse mediante un release externo versionado en vez de hacer crecer indefinidamente el repositorio Git.

Hosting recomendado:

- Google Drive para acceso rápido de la comunidad.
- Zenodo o repositorio institucional para releases citables y archivables.
- GitHub Releases solo para metadata pequeña y enlaces.

Carpeta Drive actual:

- https://drive.google.com/drive/u/1/folders/1vbnpyyy-YtVYyd-gf_-XwvHLHqIgQq8h

Nota: usar esta carpeta como staging/release externo. Antes de abrir cualquier paquete al público, revisar `manifest.csv` y completar `source`, `license` y `notes`.

## Estructura propuesta para Google Drive

```text
spectral-submersion-data/
  README_DATA.md
  manifest.csv
  checksums.sha256
  licenses/
  v0.1-public-launch/
    core/
    raw/
    processed/
    models/
    runs/
    reports/
```

## Niveles de release

| Nivel | Contenido | Propósito |
|------|----------|---------|
| `core` | Corpora pequeños, configs, reportes seleccionados | Onboarding rápido |
| `raw` | Corpora públicos y snapshots de fuentes | Reproducibilidad |
| `processed` | Embeddings, vocabularios, tablas derivadas | Experimentos más rápidos |
| `models` | Checkpoints de modelos entrenados | Demo y evaluación |
| `runs` | Outputs, figuras y resúmenes de experimentos | Auditoría |

## Esquema del manifest

Usa un CSV con estas columnas:

```csv
path,size_bytes,sha256,release_tier,source,license,notes
```

Ejemplo:

```csv
data/raw/lost_language/corpus_rongorongo_v3.csv,12345,<sha256>,raw,generated,project,Mention generation script
```

## Qué no subir

- Datos privados.
- Credenciales, API keys o archivos `.env`.
- Datasets con licencia incierta o acceso restringido.
- Notas personales no relacionadas con la investigación.
- Archivos sin fuente, licencia o script de generación trazable.

## Datos privados o restringidos

El proyecto necesita reunir la mayor cantidad posible de datos Rongorongo, pero no todo puede publicarse de inmediato. Algunos materiales pueden estar bajo copyright, custodia institucional, permisos de museo, acuerdos académicos o sensibilidad cultural.

Usaremos cuatro niveles de acceso:

| Nivel | Descripción | Puede ir al repo/Drive público |
|-------|-------------|-------------------------------|
| `open` | Licencia clara y redistribución permitida | Sí |
| `permissioned` | Uso permitido con autorización escrita | Solo si la autorización lo permite |
| `restricted` | Acceso académico, museo, comunidad o convenio | No; registrar solo metadata |
| `do-not-share` | Privado, sensible o sin permiso | No |

Regla práctica: si no hay licencia clara, no se sube el archivo. Se puede abrir un issue con metadata, enlace a la fuente, institución custodiante y estado de permisos.

Ver `DATA_WISHLIST.md` para la lista viva de datos que falta reunir.

## Checklist de publicación en Drive

1. Crear una carpeta llamada `spectral-submersion-data`.
2. Crear una subcarpeta versionada, por ejemplo `v0.1-public-launch`.
3. Subir cada nivel como carpeta separada.
4. Agregar `README_DATA.md`, `manifest.csv`, `checksums.sha256` y notas de licencia.
5. Configurar el acceso como "Anyone with the link can view".
6. Copiar el enlace público en:
   - `README.md`
   - `COMMUNITY_CALL.md`
   - `linkedin_post.md`
   - GitHub release notes
7. Abrir un issue por cada dataset con licencia incierta.

## Primer release público sugerido

Para el primer lanzamiento LinkedIn/Discord, conviene publicar un paquete acotado:

- configs necesarios para ejemplos reproducibles
- corpora públicos pequeños o medianos
- resúmenes y figuras seleccionadas
- sin llaves privadas ni notas personales no publicables
- sin bundle grande de modelos salvo que exista model card y notas de licencia

## Generar manifest local

Después de decidir los archivos exactos, genera el manifest con:

```bash
python scripts/build_data_manifest.py --output data_release_manifest.csv
```

Revisa `data_release_manifest.csv`, completa las columnas `source`, `license` y `notes` donde falte contexto, y súbelo junto con `checksums.sha256`.
