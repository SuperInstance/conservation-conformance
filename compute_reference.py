#!/usr/bin/env python3
"""Compute reference results from the test vector using Python + numpy/scipy."""

import json
import sys
import numpy as np

def main():
    with open("test_vectors.json") as f:
        tv = json.load(f)

    W = np.array(tv["adjacency_matrix"], dtype=np.float64)
    attr = np.array(tv["attributes"], dtype=np.float64)
    n = W.shape[0]

    # Degrees
    degrees = W.sum(axis=1)
    D = np.diag(degrees)

    # Unnormalized Laplacian: L = D - W
    L = D - W

    print("=== Adjacency Matrix W ===")
    for row in W:
        print("  " + "  ".join(f"{v:.15e}" for v in row))

    print("\n=== Degree Matrix D (diagonal) ===")
    print("  " + "  ".join(f"{d:.15e}" for d in degrees))

    print("\n=== Laplacian L = D - W ===")
    for row in L:
        print("  " + "  ".join(f"{v:.15e}" for v in row))

    # Symmetrize L to avoid numerical noise
    L_sym = (L + L.T) / 2.0

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    print("\n=== Eigenvalues (sorted ascending) ===")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i} = {ev:.15e}")

    print("\n=== Eigenvectors (columns of V) ===")
    for k in range(n):
        print(f"  v_{k}: " + "  ".join(f"{v:.15e}" for v in eigenvectors[:, k]))

    # Conservation ratios
    ratios = []
    for k in range(n):
        phi = eigenvectors[:, k]
        projection = phi * attr
        gradient = np.diff(projection)
        mean = np.mean(gradient)
        var = np.mean((gradient - mean) ** 2)
        ratios.append(var)
        print(f"\n  CR({k}) = {var:.15e}  (λ_{k} = {eigenvalues[k]:.15e})")

    # Spectral gap (largest gap between consecutive eigenvalues)
    gaps = np.diff(eigenvalues)
    sg = float(np.max(gaps))
    print(f"\n=== Spectral Gap ===")
    print(f"  {sg:.15e}")
    print(f"  Gaps: " + "  ".join(f"{g:.15e}" for g in gaps))

    # Cheeger constant approximation: λ₂ / 2
    if len(eigenvalues) >= 2:
        cheeger = float(eigenvalues[1]) / 2.0
    else:
        cheeger = 0.0
    print(f"\n=== Cheeger Constant (λ₂/2 approximation) ===")
    print(f"  {cheeger:.15e}")

    # Spectral fingerprint hash
    import hashlib
    rounded = np.round(eigenvalues, 6).tobytes()
    fp_hash = hashlib.sha256(rounded).hexdigest()
    print(f"\n=== Fingerprint Hash (SHA-256 of rounded eigenvalues) ===")
    print(f"  {fp_hash}")

    # Spectral entropy
    total = float(np.sum(np.abs(eigenvalues)))
    if total > 1e-15:
        probs = np.abs(eigenvalues) / total
        probs = probs[probs > 1e-15]
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0
    eff_dim = float(np.exp(entropy))
    print(f"\n=== Spectral Entropy ===")
    print(f"  {entropy:.15e}")
    print(f"\n=== Effective Dimension ===")
    print(f"  {eff_dim:.15e}")

    # Save expected results
    expected = {
        "name": tv["name"],
        "laplacian_type": tv["laplacian_type"],
        "adjacency_matrix": W.tolist(),
        "laplacian": L.tolist(),
        "degrees": degrees.tolist(),
        "eigenvalues": [float(x) for x in eigenvalues],
        "eigenvectors": [[float(x) for x in eigenvectors[:, k]] for k in range(n)],
        "conservation_ratios": [float(x) for x in ratios],
        "spectral_gap": sg,
        "cheeger_constant": cheeger,
        "fingerprint_hash": fp_hash,
        "spectral_entropy": entropy,
        "effective_dimension": eff_dim,
        "gap_profile": [float(x) for x in gaps],
        "attributes": attr.tolist(),
        "n": n,
    }

    with open("expected_results.json", "w") as f:
        json.dump(expected, f, indent=2)

    print("\n✅ Reference results saved to expected_results.json")

if __name__ == "__main__":
    main()
