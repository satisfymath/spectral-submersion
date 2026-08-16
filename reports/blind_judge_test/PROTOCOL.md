# Blind discrimination test - protocol

- Items: 20 real lines (tablets A-F, length 6-16) + 20 v6 beam+fusion
  outputs from held-out gloss sentences, shuffled (seed 42).
- Judges: k >= 3, epigraphers or advanced amateurs familiar with Barthel
  transcriptions. Each judges all 40 items independently.
- Task: label each sequence real / synthetic + confidence 1-5.
- Analysis (script `analyze_blind_judges.py` [CALCULAR al recibir hojas]):
  per-judge ROC-AUC using confidence as score; Fleiss kappa for
  inter-judge agreement; bootstrap 95% CI (B=2000) on pooled AUC.
- Interpretation guard (C1/C2/C3): AUC ~= 0.5 means judges cannot
  discriminate v6 output from real lines at the sequence-statistics
  level. It does NOT mean the output is a valid text - only that this
  discrimination channel carries no signal (claim level C1 about the
  generator, no semantic claim).
- Status: PENDIENTE - requires human judges.
