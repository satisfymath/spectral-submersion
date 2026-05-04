"""Joint SVD of structural + iconographic features for Rongorongo.

Combines the 10-dim structural features with 8-dim iconographic features
into a joint 18-dim representation, then applies SVD and evaluates via
the inverted sanity check.

Key question: does combining iconographic features with structural features
improve detection of linguistic structure beyond structural features alone?
"""

import numpy as np
from pathlib import Path
from scipy.linalg import svd
from scipy.stats import entropy


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    E_struct = np.load("data/processed/structural_features_real.npy")
    E_struct_perm = np.load("data/processed/structural_features_permuted.npy")
    E_struct_unif = np.load("data/processed/structural_features_random_uniform.npy")
    E_icon = np.load("data/processed/iconographic_features_rongorongo_real.npy")
    n_common = min(
        E_struct.shape[0],
        E_struct_perm.shape[0],
        E_struct_unif.shape[0],
        E_icon.shape[0],
    )
    E_struct = E_struct[:n_common]
    E_struct_perm = E_struct_perm[:n_common]
    E_struct_unif = E_struct_unif[:n_common]
    E_icon = E_icon[:n_common]
    print(f"Using {n_common} common rows (truncated to match uniform baseline)")

    print(f"Structural features: {E_struct.shape}")
    print(f"Iconographic features: {E_icon.shape}")
    print(f"Permuted structural: {E_struct_perm.shape}")
    print(f"Uniform structural: {E_struct_unif.shape}")

    # Standardize each feature to zero mean, unit variance
    def standardize(E):
        mu = E.mean(axis=0, keepdims=True)
        sigma = E.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        return (E - mu) / sigma

    E_struct_s = standardize(E_struct)
    E_struct_perm_s = standardize(E_struct_perm)
    E_struct_unif_s = standardize(E_struct_unif)
    E_icon_s = standardize(E_icon)

    # ====== SVD of structural features only ======
    print("\n=== SVD: Structural Features Only (10-dim) ===")
    U_s, s_s, Vt_s = svd(E_struct_s, full_matrices=False)
    r_eff_struct = np.sum(s_s**2) / np.sum(s_s[0] ** 2)
    print(f"  r_eff = {r_eff_struct:.2f}")
    for k in [2, 5, 10]:
        frac = np.sum(s_s[:k] ** 2) / np.sum(s_s**2) * 100
        print(f"  Top-{k} explains {frac:.1f}% of variance")

    # Permuted baseline
    U_sp, s_sp, Vt_sp = svd(E_struct_perm_s, full_matrices=False)
    r_eff_struct_perm = np.sum(s_sp**2) / np.sum(s_sp[0] ** 2)
    print(f"  permuted r_eff = {r_eff_struct_perm:.2f}")

    # Uniform baseline
    U_su, s_su, Vt_su = svd(E_struct_unif_s, full_matrices=False)
    r_eff_struct_unif = np.sum(s_su**2) / np.sum(s_su[0] ** 2)
    print(f"  uniform r_eff = {r_eff_struct_unif:.2f}")
    print(
        f"  Inverted check: real({r_eff_struct:.2f}) > perm({r_eff_struct_perm:.2f}) > unif({r_eff_struct_unif:.2f}) ? "
        f"{'PASS' if r_eff_struct > r_eff_struct_perm > r_eff_struct_unif else 'FAIL'}"
    )

    # ====== SVD of iconographic features only ======
    print("\n=== SVD: Iconographic Features Only (8-dim) ===")
    U_i, s_i, Vt_i = svd(E_icon_s, full_matrices=False)
    r_eff_icon = np.sum(s_i**2) / np.sum(s_i[0] ** 2)
    print(f"  r_eff = {r_eff_icon:.2f}")
    for k in [2, 5, 8]:
        frac = np.sum(s_i[:k] ** 2) / np.sum(s_i**2) * 100
        print(f"  Top-{k} explains {frac:.1f}% of variance")

    # ====== Joint SVD: Concatenated features ======
    print("\n=== Joint SVD: Structural + Iconographic (18-dim) ===")
    E_joint = np.hstack([E_struct_s, E_icon_s])

    # Shuffle iconographic features to create baseline
    rng = np.random.default_rng(42)
    E_icon_shuf = E_icon_s[rng.permutation(E_icon_s.shape[0])]

    # Permuted structural features baseline
    E_joint_perm = np.hstack([E_struct_perm_s, E_icon_s])

    # Random uniform structural baseline
    E_joint_unif = np.hstack([E_struct_unif_s, E_icon_s])

    # Shuffled iconographic baseline (structural real + iconographic random)
    E_joint_icon_shuf = np.hstack([E_struct_s, E_icon_shuf])

    # Joint SVDs
    for label, E in [
        ("real+icon", E_joint),
        ("perm+icon", E_joint_perm),
        ("unif+icon", E_joint_unif),
        ("real+icon_shuf", E_joint_icon_shuf),
    ]:
        U, s, Vt = svd(E, full_matrices=False)
        r_eff = float(np.sum(s**2) / np.sum(s[0] ** 2))
        frac2 = float(np.sum(s[:2] ** 2) / np.sum(s**2) * 100)
        frac5 = float(np.sum(s[:5] ** 2) / np.sum(s**2) * 100)
        print(
            f"  {label:20s}: r_eff={r_eff:.2f}, top-2={frac2:.1f}%, top-5={frac5:.1f}%"
        )
        print(
            f"  {label:20s}: r_eff={r_eff:.2f}, top-2={frac2:.1f}%, top-5={frac5:.1f}%"
        )

    print(
        f"\n  Inverted check: real+icon({r_eff:.2f}) should be > perm+icon and > real+icon_shuf"
    )

    # ====== Weighted combination ======
    print("\n=== Weighted Combinations ===")
    results = []
    for w_struct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_icon = 1.0 - w_struct
        # Pad iconographic features with zeros to match structural dimensions for weighted combination
        # Actually use concatenation approach: combine via SVD
        E_combined = np.hstack([w_struct * E_struct_s, w_icon * E_icon_s])

        # Baseline: permuted structural + iconographic
        E_combined_perm = np.hstack([w_struct * E_struct_perm_s, w_icon * E_icon_s])
        E_combined_unif = np.hstack([w_struct * E_struct_unif_s, w_icon * E_icon_s])

        _, s_real, _ = svd(E_combined, full_matrices=False)
        _, s_perm, _ = svd(E_combined_perm, full_matrices=False)
        _, s_unif, _ = svd(E_combined_unif, full_matrices=False)

        r_real = np.sum(s_real**2) / np.sum(s_real[0] ** 2)
        r_perm = np.sum(s_perm**2) / np.sum(s_perm[0] ** 2)
        r_unif = np.sum(s_unif**2) / np.sum(s_unif[0] ** 2)

        passed = r_real > r_perm > r_unif
        results.append(
            {
                "w_struct": w_struct,
                "w_icon": w_icon,
                "r_real": r_real,
                "r_perm": r_perm,
                "r_unif": r_unif,
                "pass": passed,
            }
        )
        print(
            f"  w_struct={w_struct:.2f}, w_icon={w_icon:.2f}: "
            f"r_real={r_real:.2f}, r_perm={r_perm:.2f}, r_unif={r_unif:.2f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # ====== Correlation between feature sets ======
    print("\n=== Feature Correlations ===")
    corr_matrix = np.corrcoef(E_struct_s.T, E_icon_s.T)
    n_s, n_i = E_struct_s.shape[1], E_icon_s.shape[1]
    cross_corr = corr_matrix[:n_s, n_s:]
    feature_names_s = [
        "log_freq",
        "type_token",
        "first_ratio",
        "last_ratio",
        "pos_mean",
        "pos_std",
        "rep_rate",
        "mean_run_len",
        "succ_entropy",
        "freq_rank_norm",
    ]
    feature_names_i = [
        "width",
        "height",
        "aspect_ratio",
        "path_complexity",
        "stroke_density",
        "n_components",
        "fill_ratio",
        "symmetry",
    ]

    print(f"  Max |correlation| = {np.max(np.abs(cross_corr)):.4f}")
    print(f"  Mean |correlation| = {np.mean(np.abs(cross_corr)):.4f}")

    # Find strongest cross-correlations
    flat_idx = np.argsort(np.abs(cross_corr.ravel()))[::-1]
    print("  Top 5 cross-correlations:")
    for idx in flat_idx[:5]:
        i_s, i_i = divmod(idx, n_i)
        print(
            f"    {feature_names_s[i_s]:15s} <-> {feature_names_i[i_i]:20s}: r={cross_corr[i_s, i_i]:.4f}"
        )

    # Save
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "joint_svd_results.csv", index=False)
    np.save(out_dir / "cross_correlation_matrix.npy", cross_corr)
    print(f"\nResults saved to {out_dir}/joint_svd_results.csv")


if __name__ == "__main__":
    main()
