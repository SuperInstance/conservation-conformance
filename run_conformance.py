#!/usr/bin/env python3
"""
Cross-language conformance test runner for Conservation Spectral SDK.
"""

import json
import subprocess
import sys
import os
import shutil
import string

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(BASE)

EIGENVALUE_TOL = 1e-6
EIGENVECTOR_TOL = 1e-4
CONSERVATION_TOL = 1e-4
SCALAR_TOL = 1e-6


def load_expected():
    with open(os.path.join(BASE, "expected_results.json")) as f:
        return json.load(f)


def compare_eigenvalues(expected, actual):
    if len(expected) != len(actual):
        return False, f"    Length mismatch: {len(expected)} vs {len(actual)}"
    ok_all = True
    lines = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        diff = abs(e - a)
        ok = diff < EIGENVALUE_TOL
        if not ok: ok_all = False
        lines.append(f"    λ_{i}: exp={e:.10e} got={a:.10e} diff={diff:.2e} {'✅' if ok else '❌'}")
    return ok_all, "\n".join(lines)


def compare_eigenvectors(expected_vecs, actual_vecs, n):
    ok_all = True
    lines = []
    for k in range(n):
        ev, av = expected_vecs[k], actual_vecs[k]
        dot = sum(e * a for e, a in zip(ev, av))
        deviation = abs(abs(dot) - 1.0)
        ok = deviation < EIGENVECTOR_TOL
        if not ok: ok_all = False
        lines.append(f"    v_{k}: |dot|={abs(dot):.10f} dev={deviation:.2e} {'✅' if ok else '❌'}")
    return ok_all, "\n".join(lines)


def compare_conservation_ratios(expected, actual):
    if len(expected) != len(actual):
        return False, f"    Length mismatch: {len(expected)} vs {len(actual)}"
    ok_all = True
    lines = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        if i == 0:
            lines.append(f"    CR({i}): exp={e:.10e} got={a:.10e} (skipped — zero eigenvalue)")
            continue
        diff = abs(e - a)
        ok = diff < CONSERVATION_TOL
        if not ok: ok_all = False
        lines.append(f"    CR({i}): exp={e:.10e} got={a:.10e} diff={diff:.2e} {'✅' if ok else '❌'}")
    return ok_all, "\n".join(lines)


def compare_scalar(expected, actual, name):
    diff = abs(expected - actual)
    ok = diff < SCALAR_TOL
    return ok, f"  {name}: exp={expected:.10e} got={actual:.10e} diff={diff:.2e} {'✅' if ok else '❌'}"


def run_comparison(expected, actual, n):
    details = []
    all_pass = True

    ok, d = compare_eigenvalues(expected["eigenvalues"], actual["eigenvalues"])
    details.append("  Eigenvalues:"); details.append(d); all_pass &= ok

    ok, d = compare_eigenvectors(expected["eigenvectors"], actual["eigenvectors"], n)
    details.append("  Eigenvectors:"); details.append(d); all_pass &= ok

    ok, d = compare_conservation_ratios(expected["conservation_ratios"], actual["conservation_ratios"])
    details.append("  Conservation Ratios:"); details.append(d); all_pass &= ok

    for key in ["spectral_gap", "cheeger_constant"]:
        ok, d = compare_scalar(expected[key], actual[key], key)
        details.append(d); all_pass &= ok

    return all_pass, "\n".join(details)


def test_python(expected):
    print("\n" + "=" * 60 + "\n🐍 Testing Python implementation\n" + "=" * 60)
    n = expected["n"]

    script = (
        "import sys, json, numpy as np\n"
        f"sys.path.insert(0, '{WORKSPACE}/conservation-spectral-python/src')\n"
        "from conservation_spectral import (TensionGraph, build_laplacian, eigendecompose,\n"
        "    conservation_ratio, spectral_gap)\n"
        "\n"
        f"with open('{BASE}/expected_results.json') as f:\n"
        "    exp = json.load(f)\n"
        "W = np.array(exp['adjacency_matrix'])\n"
        "attr = np.array(exp['attributes'])\n"
        "n = exp['n']\n"
        "\n"
        "g = TensionGraph(directed=False)\n"
        "labels = ['C', 'G', 'Am', 'F', 'Dm']\n"
        "for i in range(n): g.add_vertex(labels[i])\n"
        "for i in range(n):\n"
        "    for j in range(i+1, n):\n"
        "        if W[i][j] > 0: g.add_edge(labels[i], labels[j], W[i][j])\n"
        "\n"
        "lap = build_laplacian(g, normalized=False, laplacian_type='unnormalized')\n"
        "eigen = eigendecompose(lap, laplacian_type='unnormalized')\n"
        "\n"
        "result = {\n"
        "    'eigenvalues': [float(x) for x in eigen.eigenvalues],\n"
        "    'eigenvectors': [[float(x) for x in eigen.eigenvectors[:, k]] for k in range(n)],\n"
        "    'conservation_ratios': [float(conservation_ratio(eigen, attr, k)) for k in range(n)],\n"
        "    'spectral_gap': float(spectral_gap(eigen.eigenvalues)),\n"
        "    'cheeger_constant': float(eigen.eigenvalues[1]) / 2.0 if len(eigen.eigenvalues) >= 2 else 0.0,\n"
        "}\n"
        "print(json.dumps(result))\n"
    )

    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f"  Python error: {r.stderr[:500]}"
        actual = json.loads(r.stdout.strip())
    except Exception as e:
        return False, f"  Python exception: {e}"

    return run_comparison(expected, actual, n)


def test_c(expected):
    print("\n" + "=" * 60 + "\n⚙️  Testing C implementation\n" + "=" * 60)
    n = expected["n"]
    adj = expected["adjacency_matrix"]
    attr = expected["attributes"]
    header = os.path.join(WORKSPACE, "conservation-spectral-c", "conservation_spectral.h")

    attr_str = ", ".join(f"{v:.1f}" for v in attr)
    edges_lines = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] > 0:
                edges_lines.append(f"    cs_graph_add_edge(g, {i}, {j}, {adj[i][j]:.1f});")
    edges_str = "\n".join(edges_lines)

    c_code = (
        '#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n'
        '#define CS_IMPLEMENTATION\n'
        f'#include "{header}"\n\n'
        'int main() {\n'
        f'    int n = {n};\n'
        '    cs_graph *g = cs_graph_create(n);\n'
        f'    double attr[] = {{{attr_str}}};\n'
        '    for (int i = 0; i < n; i++)\n'
        '        cs_graph_add_vertex(g, i, attr[i]);\n'
        f'{edges_str}\n'
        '    cs_laplacian lap = cs_build_laplacian(g, false);\n'
        '    cs_eigen eig = cs_eigendecompose(&lap, 0);\n\n'
        '    printf("{\\n");\n'
        '    printf("  \\"eigenvalues\\": [");\n'
        '    for (int i = 0; i < n; i++) {\n'
        '        if (i > 0) printf(", ");\n'
        '        printf("%.15e", eig.eigenvalues[i]);\n'
        '    }\n'
        '    printf("],\\n  \\"eigenvectors\\": [");\n'
        '    for (int k = 0; k < n; k++) {\n'
        '        if (k > 0) printf(", ");\n'
        '        printf("[");\n'
        '        for (int i = 0; i < n; i++) {\n'
        '            if (i > 0) printf(", ");\n'
        '            printf("%.15e", eig.eigenvectors[k * n + i]);\n'
        '        }\n'
        '        printf("]");\n'
        '    }\n'
        '    printf("],\\n  \\"conservation_ratios\\": [");\n'
        '    for (int k = 0; k < n; k++) {\n'
        '        if (k > 0) printf(", ");\n'
        '        printf("%.15e", cs_conservation_ratio(&eig, attr, n, k));\n'
        '    }\n'
        '    printf("],\\n  \\"spectral_gap\\": %.15e,\\n", cs_spectral_gap(&eig));\n'
        '    double cheeger = (eig.n >= 2) ? eig.eigenvalues[1] / 2.0 : 0.0;\n'
        '    printf("  \\"cheeger_constant\\": %.15e\\n}\\n", cheeger);\n\n'
        '    cs_laplacian_free(&lap);\n'
        '    cs_eigen_free(&eig);\n'
        '    cs_graph_free(g);\n'
        '    return 0;\n'
        '}\n'
    )

    c_file = os.path.join(BASE, "_test_c.c")
    exe_file = os.path.join(BASE, "_test_c")

    try:
        with open(c_file, "w") as f:
            f.write(c_code)
        r = subprocess.run(["gcc", "-O2", "-o", exe_file, c_file, "-lm"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f"  C compile error: {r.stderr[:500]}"
        r = subprocess.run([exe_file], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f"  C runtime error: {r.stderr[:500]}"
        actual = json.loads(r.stdout.strip())
    except Exception as e:
        return False, f"  C exception: {e}"
    finally:
        for fp in [c_file, exe_file]:
            if os.path.exists(fp): os.unlink(fp)

    return run_comparison(expected, actual, n)


def test_js(expected):
    print("\n" + "=" * 60 + "\n🟨 Testing JavaScript implementation\n" + "=" * 60)
    n = expected["n"]
    adj_json = json.dumps(expected["adjacency_matrix"])
    attr_json = json.dumps(expected["attributes"])

    js_code = (
        f'const {{ jacobiEigen }} = require("{WORKSPACE}/conservation-spectral-js/dist/eigen.js");\n\n'
        f'const n = {n};\n'
        f'const adj = {adj_json};\n'
        f'const attr = new Float64Array({attr_json});\n\n'
        'const W = new Float64Array(n * n);\n'
        'for (let i = 0; i < n; i++)\n'
        '    for (let j = 0; j < n; j++)\n'
        '        W[i * n + j] = adj[i][j];\n\n'
        'const degrees = new Float64Array(n);\n'
        'for (let i = 0; i < n; i++) {\n'
        '    let d = 0;\n'
        '    for (let j = 0; j < n; j++) d += W[i * n + j];\n'
        '    degrees[i] = d;\n'
        '}\n\n'
        'const L = new Float64Array(n * n);\n'
        'for (let i = 0; i < n; i++)\n'
        '    for (let j = 0; j < n; j++)\n'
        '        L[i * n + j] = (i === j ? degrees[i] : 0) - W[i * n + j];\n\n'
        'const eigen = jacobiEigen(L, n, n * n * 10);\n\n'
        'function conservationRatio(vectors, attr, k) {\n'
        '    const phi = vectors[k];\n'
        '    const n = attr.length;\n'
        '    const projected = new Float64Array(n);\n'
        '    for (let i = 0; i < n; i++) projected[i] = attr[i] * phi[i];\n'
        '    const diffs = new Float64Array(n - 1);\n'
        '    for (let i = 0; i < n - 1; i++) diffs[i] = projected[i + 1] - projected[i];\n'
        '    let mean = 0;\n'
        '    for (let i = 0; i < diffs.length; i++) mean += diffs[i];\n'
        '    mean /= diffs.length;\n'
        '    let variance = 0;\n'
        '    for (let i = 0; i < diffs.length; i++) variance += (diffs[i] - mean) ** 2;\n'
        '    variance /= diffs.length;\n'
        '    return variance;\n'
        '}\n\n'
        'const ratios = [];\n'
        'for (let k = 0; k < n; k++) ratios.push(conservationRatio(eigen.vectors, attr, k));\n\n'
        'let maxGap = 0;\n'
        'for (let i = 1; i < n; i++) {\n'
        '    const gap = eigen.values[i] - eigen.values[i - 1];\n'
        '    if (gap > maxGap) maxGap = gap;\n'
        '}\n'
        'const cheeger = eigen.values.length >= 2 ? eigen.values[1] / 2.0 : 0.0;\n\n'
        'const result = {\n'
        '    eigenvalues: Array.from(eigen.values),\n'
        '    eigenvectors: eigen.vectors.map(v => Array.from(v)),\n'
        '    conservation_ratios: ratios,\n'
        '    spectral_gap: maxGap,\n'
        '    cheeger_constant: cheeger\n'
        '};\n'
        'console.log(JSON.stringify(result));\n'
    )

    js_file = os.path.join(BASE, "_test_js.js")
    try:
        with open(js_file, "w") as f:
            f.write(js_code)
        r = subprocess.run(["node", js_file], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, f"  JS error: {r.stderr[:500]}"
        actual = json.loads(r.stdout.strip())
    except Exception as e:
        return False, f"  JS exception: {e}"
    finally:
        if os.path.exists(js_file): os.unlink(js_file)

    return run_comparison(expected, actual, n)


def test_rust(expected):
    print("\n" + "=" * 60 + "\n🦀 Testing Rust implementation\n" + "=" * 60)
    n = expected["n"]
    adj = expected["adjacency_matrix"]
    attr = expected["attributes"]

    adj_flat = ", ".join(f"{adj[i][j]:.1f}" for i in range(n) for j in range(n))
    attr_str = ", ".join(f"{v:.1f}" for v in attr)

    test_dir = os.path.join(BASE, "_rust_test")
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.makedirs(os.path.join(test_dir, "src"))

    with open(os.path.join(test_dir, "Cargo.toml"), "w") as f:
        f.write(f'''[package]
name = "conformance_test"
version = "0.1.0"
edition = "2021"

[dependencies]
conservation-spectral-core = {{ path = "{WORKSPACE}/conservation-spectral" }}
''')

    main_rs = (
        'use conservation_spectral_core::laplacian::build_laplacian;\n'
        'use conservation_spectral_core::eigen::eigendecompose;\n'
        'use conservation_spectral_core::LaplacianType;\n\n'
        'fn main() {\n'
        f'    let n: usize = {n};\n'
        f'    let adj: Vec<f64> = vec![{adj_flat}];\n'
        f'    let attr: Vec<f64> = vec![{attr_str}];\n\n'
        '    let lap = build_laplacian(&adj, n, |_i: usize, _j: usize| 1.0, LaplacianType::Unnormalized);\n'
        '    let eigen = eigendecompose(&lap);\n\n'
        '    let evals: Vec<String> = eigen.eigenvalues.iter().map(|v| format!("{:.15e}", v)).collect();\n'
        '    let evecs: Vec<String> = eigen.eigenvectors.iter().map(|v| {\n'
        '        let row: Vec<String> = v.iter().map(|x| format!("{:.15e}", x)).collect();\n'
        '        format!("[{}]", row.join(", "))\n'
        '    }).collect();\n\n'
        '    let ratios: Vec<String> = (0..eigen.eigenvalues.len()).map(|k| {\n'
        '        let phi = &eigen.eigenvectors[k];\n'
        '        let m = phi.len().min(attr.len());\n'
        '        let projected: Vec<f64> = (0..m).map(|i| phi[i] * attr[i]).collect();\n'
        '        let gradients: Vec<f64> = projected.windows(2).map(|w| w[1] - w[0]).collect();\n'
        '        let gn = gradients.len() as f64;\n'
        '        let mean: f64 = gradients.iter().sum::<f64>() / gn;\n'
        '        let variance: f64 = gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gn;\n'
        '        format!("{:.15e}", variance)\n'
        '    }).collect();\n\n'
        '    let mut max_gap = 0.0_f64;\n'
        '    for w in eigen.eigenvalues.windows(2) {\n'
        '        let gap = w[1] - w[0];\n'
        '        if gap > max_gap { max_gap = gap; }\n'
        '    }\n'
        '    let cheeger = if eigen.eigenvalues.len() >= 2 { eigen.eigenvalues[1] / 2.0 } else { 0.0 };\n\n'
        '    println!("{{");\n'
        '    println!("  \\"eigenvalues\\": [{}],", evals.join(", "));\n'
        '    println!("  \\"eigenvectors\\": [{}],", evecs.join(", "));\n'
        '    println!("  \\"conservation_ratios\\": [{}],", ratios.join(", "));\n'
        '    println!("  \\"spectral_gap\\": {:.15e},", max_gap);\n'
        '    println!("  \\"cheeger_constant\\": {:.15e}", cheeger);\n'
        '    println!("}}");\n'
        '}\n'
    )

    try:
        with open(os.path.join(test_dir, "src", "main.rs"), "w") as f:
            f.write(main_rs)
        r = subprocess.run(["cargo", "run", "--release"], capture_output=True, text=True,
                           timeout=120, cwd=test_dir)
        if r.returncode != 0:
            return False, f"  Rust error: {r.stderr[:800]}"
        actual = json.loads(r.stdout.strip())
    except Exception as e:
        return False, f"  Rust exception: {e}"
    finally:
        if os.path.exists(test_dir): shutil.rmtree(test_dir)

    return run_comparison(expected, actual, n)


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Conservation Spectral SDK — Cross-Language Conformance     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    expected = load_expected()
    n = expected["n"]
    print(f"\nTest: {expected['name']} ({n} vertices, {expected['laplacian_type']})")
    print(f"Tolerances: eigenvalues={EIGENVALUE_TOL}, eigenvectors={EIGENVECTOR_TOL}, "
          f"conservation={CONSERVATION_TOL}, scalars={SCALAR_TOL}")

    results = {}
    for name, test_fn in [("Python", test_python), ("C", test_c), ("JS", test_js), ("Rust", test_rust)]:
        ok, details = test_fn(expected)
        results[name] = ok
        print(details)

    print("\n" + "=" * 60)
    print("📊 CONFORMANCE SUMMARY")
    print("=" * 60)
    all_pass = True
    for impl, passed in results.items():
        print(f"  {impl:12s} {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed: all_pass = False

    print("=" * 60)
    if all_pass:
        print("🎉 ALL IMPLEMENTATIONS CONFORM")
    else:
        print("⚠️  SOME IMPLEMENTATIONS DIVERGE — see details above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
