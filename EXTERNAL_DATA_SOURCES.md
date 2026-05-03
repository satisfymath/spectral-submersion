# External Data Sources for Polynesian Languages

## Available (downloaded/free)

### Currently in repository
- `data/raw/candidate_languages/maori_tokens.csv` - 3,543 lines
- `data/raw/candidate_languages/haw_tokens.csv` - 803 lines (too small)
- Other Polynesian languages: very small corpora (<1000 lines each)

## Available externally (requires download)

### Leipzig Wortschatz (wortschatz.uni-leipzig.de)
- **Maori**: Multiple year-corpora (2011-2017), sizes: 10K, 30K, 100K sentences
- **Tonga**: 2,524 sentences, 41,821 tokens, 6,310 types (2012)

### Rapa Nui Corpus (rapanui.polycorpora.org)
- **Parallel texts** with translations
- Glossed texts (restricted access)
- Project 2012-2014 (requires institutional access)

### Zenodo - Polynesian Segmented Data (DOI: 10.5281/zenodo.1689909)
- 210 basic vocabulary concepts
- 31 Polynesian languages
- Segmented for cognate detection
- CC-BY-4.0 license

## Gap Analysis

| Language | Current tokens | Ideal tokens | Gap |
|----------|-------------|------------|-----|
| Maori | ~3,500 | >50,000 | Need enrichment |
| Hawaiian | ~800 | >50,000 | Critical |
| Rapa Nui | ~100 | >50,000 | Critical |
| Samoan | ~350 | >50,000 | Critical |

## Recommendation

1. Download Maori from Leipzig Wortschatz (can increase from 3,543 to 100K lines)
2. The Zenodo segmented data could provide cognate anchors for alignment
3. Rapa Nui corpus at rapanui.polycorpora.org has parallel texts but access is restricted