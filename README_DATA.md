# Spectral Submersion - Data Release v0.1

Public launch data package for the Spectral Submersion project.

## Contents

| Tier | Path | Description |
|------|------|-------------|
| `core` | `configs/` | Pipeline configuration files (YAML) |
| `raw` | `data/raw/` | Tokenized candidate language corpora from OPUS-Tatoeba, synthetic and real lost-language corpora |
| `raw` | `data/external/` | Raw Tatoeba downloads, Rongorongo RR corpus XML transcriptions, iconic referent images (Rapa Nui ca. 1500) |
| `processed` | `data/processed/` | Spectral embeddings (NumPy), vocabularies (JSON), derived tables |
| `models` | `models/` | Trained translator model checkpoints (PyTorch) |
| `runs` | `runs/` | Experiment outputs: iconic grounding PhD pipeline, cross-script validation, audit reports |
| `reports` | `reports/` | Tables (CSV/JSON), figures (PNG), hypothesis reports |

## Quick start

```bash
# Extract
unzip spectral-submersion-data-v0.1-public-launch.zip
cd spectral-submersion-data

# Verify integrity
sha256sum -c checksums.sha256

# Browse structure
ls v0.1-public-launch/
```

## Release tiers

- **core**: Onboarding essentials (configs).
- **raw**: Public corpora and source snapshots for reproducibility.
- **processed**: Embeddings, vocabularies, derived tables for faster experiments.
- **models**: Trained model checkpoints for demo and evaluation.
- **runs**: Full experiment outputs, figures, and summaries for audit.
- **reports**: Tables and figures from published analyses.

## License

See `licenses/` for individual dataset licenses. Unless otherwise noted:
- Code: MIT (see `LICENSE` in main repo)
- Data: varies by source (Tatoeba data: CC-BY 2.0 FR; generated synthetic data: project license)
- Referent images: used for research purposes; original sources retain copyright

## Links

- GitHub: https://github.com/satisfymath/spectral-submersion
- Paper: https://github.com/satisfymath/spectral-submersion/blob/main/submersion_espectral_lenguajes_perdidos_paper.pdf
- Discord: https://discord.gg/ueg94XVw
