# Real Iconic Grounding Run

This run uses real RR-corpus SVG glyph paths and real local referent images.
It is an iconographic candidate-generation run, not a decipherment.

## Summary

- Glyph classes analyzed: 448
- Referents with real images: 17
- Mean top-1 iconicity: 0.722
- AnchorPower at threshold 0.6: 0.970
- Bootstrap assignment stability: 0.820
- NegCtrlGap: 144.910 (very_strong_check_leakage)
- Cross-script Acc@5 supplied: 0.286
- C2.5 admitted top-1 candidates: 0

## Top Candidate Rows

| glyph | referent | iconicity | failed criteria |
|---|---:|---:|---|
| `078` | `toki` | 0.851 | cross_script_validation |
| `493` | `tiger_shark` | 0.844 | cross_script_validation |
| `211` | `moray_eel` | 0.829 | cross_script_validation |
| `325` | `sun` | 0.822 | cross_script_validation |
| `030` | `moai` | 0.821 | cross_script_validation |
| `410` | `hand` | 0.821 | cross_script_validation |
| `461` | `tiger_shark` | 0.820 | cross_script_validation |
| `205` | `moray_eel` | 0.819 | cross_script_validation |
| `770` | `moai` | 0.819 | cross_script_validation |
| `245` | `hand` | 0.817 | cross_script_validation |
| `697` | `hand` | 0.817 | cross_script_validation |
| `665` | `moai` | 0.814 | cross_script_validation |
| `543` | `moray_eel` | 0.813 | cross_script_validation |
| `202` | `moray_eel` | 0.813 | cross_script_validation |
| `660` | `hand` | 0.812 | cross_script_validation |
| `595` | `moray_eel` | 0.812 | cross_script_validation |
| `376` | `tiger_shark` | 0.811 | cross_script_validation |
| `207` | `fairy_tern` | 0.810 | cross_script_validation |
| `242` | `sun` | 0.810 | cross_script_validation |
| `460` | `fairy_tern` | 0.809 | cross_script_validation |

## Interpretation Guardrail

C2.5 is intentionally blocked in this run because real cross-script validation has not yet been provided. The very high negative-control gap is also flagged as `very_strong_check_leakage`, so these rankings should be treated as candidates for inspection and validation, not as semantic claims.
