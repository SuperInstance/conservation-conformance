# conservation-conformance

Cross-language conformance test suite for the Conservation Spectral SDK — verifies that Python, C, JavaScript, and Rust implementations produce identical spectral results.

## What This Gives You

- **Reference test vectors** — Precomputed adjacency matrix with expected eigenvalues, eigenvectors, conservation ratios, spectral gap, and Cheeger constant
- **Four language runners** — Automated comparison against Python, C (gcc), JavaScript (Node), and Rust implementations
- **Configurable tolerances** — Eigenvalue (1e-6), eigenvector (1e-4), conservation ratio (1e-4), and scalar (1e-6) tolerances
- **Pass/fail summary** — Clear ✅/❌ output per implementation with per-value detail

## Quick Start

```bash
# Run conformance tests against all implementations
python run_conformance.py
```

Output:

```
╔══════════════════════════════════════════════════════════════╗
║  Conservation Spectral SDK — Cross-Language Conformance     ║
╚══════════════════════════════════════════════════════════════╝

📊 CONFORMANCE SUMMARY
  Python       ✅ PASS
  C            ✅ PASS
  JS           ✅ PASS
  Rust         ✅ PASS

🎉 ALL IMPLEMENTATIONS CONFORM
```

## How It Works

1. Loads [`expected_results.json`](expected_results.json) — reference vectors computed by [`compute_reference.py`](compute_reference.py)
2. For each language, builds the same 5-vertex chord-progression graph
3. Runs Laplacian construction → eigendecomposition → conservation analysis
4. Compares results against reference within tolerance
5. Reports per-value agreement and overall pass/fail

## Files

| File | Purpose |
|------|---------|
| [`expected_results.json`](expected_results.json) | Reference eigenvalues, eigenvectors, ratios, scalars |
| [`test_vectors.json`](test_vectors.json) | Input adjacency matrix and attributes |
| [`compute_reference.py`](compute_reference.py) | Generate reference results from the Python SDK |
| [`run_conformance.py`](run_conformance.py) | Main conformance runner (Python, C, JS, Rust) |

## How It Fits

This is the **conformance layer** for the conservation spectral ecosystem:

- **Rust**: [conservation-spectral](https://github.com/SuperInstance/conservation-spectral)
- **Python**: [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python)
- **TypeScript**: [conservation-spectral-js](https://github.com/SuperInstance/conservation-spectral-js)
- **Ada**: [conservation-spectral-ada](https://github.com/SuperInstance/conservation-spectral-ada)

All implementations must pass this suite before release.

## Requirements

- Python ≥ 3.10 with `numpy`, `scipy`, and the `conservation-spectral` package installed
- `gcc` for C tests
- `node` for JavaScript tests
- `cargo` for Rust tests

## License

MIT
