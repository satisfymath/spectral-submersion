# Petición de mano en rongorongo — línea de máxima certeza

**Fecha:** 2026-09-18 · **Tablilla:** `reports/figures/tablilla_propuesta.png`
**Scripts:** `scripts/proposal_certainty_tournament.py` (rondas 1 y 2, JSON en `reports/proposal_certainty_tournament_r{1,2}.json`), `scripts/render_tablet.py`

## Frase

| Capa | Contenido |
|---|---|
| Español | Tú eres mi esposa; te amo para toda la vida |
| Rapanui (glosa, vocabulario del traductor) | *ko kou taku vahine, e aroha au ki a kou, mo te ora tonu* |
| Barthel (v6 beam+fusión LM real) | `002 002 004 004 280 280 450 063 063 004 004 002 002 004 004 430 022` |

## Qué certeza tiene, y de qué tipo

El rongorongo no está descifrado; nadie puede leer esta línea. La certeza que sí se puede medir es de **consistencia del sistema**:

| Métrica | Valor | Significado |
|---|---|---|
| Acuerdo entre modelos (v6 vs v6_noaug, entrenados por separado) | 16/17 glifos compartidos (0.94); difieren en un 004 repetido y un 022 final | El mapa no depende de una corrida de entrenamiento |
| Estabilidad ante el decodificador (13 configuraciones × 2 modelos) | núcleo `002 002 004 004 280 280 450 063 063 … 430 022` en 24/26 | No es un artefacto del beam ni del peso del LM |
| Clase iconográfica (taxonomía de Barthel) | *vahine* → 280 280 (antropomorfos 200–399), *aroha* → 450 (400–599) | La única capa con consenso académico: las figuras 200–399 representan personas |
| bits/glifo bajo el LM trigram de las tablillas reales | 6.14 (líneas reales: 6.28 ± 0.33, z = −0.41) | Estadísticamente indistinguible de una línea real |
| Bigramas atestiguados en el corpus real | 15/16 (0.94) | Casi cada transición existe en las tablillas A–F |
| Fórmulas ya grabadas en la tablilla del poema | `002 002`, `004 004`, `430 022` (*ora tonu*, cierre de L9) | Coherente con la tablilla que acompaña |
| Trazado real disponible | 7/7 códigos (002, 004, 280, 450, 063, 430, 022) | Renderizable con glifos reales |

Finalista B (segunda): *makemake au ki a kou mo te ora tonu* («te elijo para toda la vida») → `041h 670 580 001 006 006 001 006 006 430 022`: acuerdo exacto entre los dos modelos con la decodificación por defecto, pero el relleno de partículas cambia con el decodificador (núcleo estable 22/26). Abre con la fórmula de *makemake* (041h 670 580), ya en L6 y L9 del poema.

## Advertencia de gramática rapanui
La glosa está construida dentro del vocabulario de 232 tipos del traductor, no verificada por un hablante. *moe* = dormir/casarse, *vahine* = esposa/mujer, *aroha* = amor, *ora tonu* = vida perpetua, *taku* = mi, *kou* = tú.
