"""
End-to-end FSCI pipeline demo, using a synthetic circular equilibrium
so it runs anywhere without a real TCABR g-file.

Usage
-----
    python examples/run_pipeline_example.py

To run against a real TCABR discharge instead, replace the two
``build_synthetic_*`` calls with:

    eq = fsci.load_equilibrium("/path/to/g000000.eqdsk")
    optics = fsci.build_optical_geometry(XC, YC, S_HARDWARE)

where ``S_HARDWARE`` should be the real, Zemax-derived impact
parameters of the 32 TCABR channels (still a placeholder/synthetic
array as of this writing — see the project README).
"""

import numpy as np

import fsci


def main():
    print("Building a synthetic circular equilibrium (Module 2 stand-in)...")
    eq = fsci.build_synthetic_equilibrium()

    print("Building a synthetic 8-channel optical geometry (Module 4 stand-in)...")
    optics = fsci.build_synthetic_optics(eq, n_channels=8)

    print("Discretising into 10 flux-surface shells (Module 3)...")
    grid = fsci.build_flux_grid(n_shells=10)

    print("Tracing rays and building the geometric matrix G (Module 5)...")
    inv = fsci.build_geometric_matrix(optics, grid, eq)
    print(f"  G shape: {inv.data.shape}, ds = {inv.ds * 1000:.3f} mm")
    print(f"  Max length in plasma: {inv.length_in_plasma.max():.4f} m")

    print("Running regression + independent allocation validation...")
    reg_report = fsci.run_regression_tests(optics, eq, inv)
    alloc_report = fsci.run_independent_allocation_validation(optics, eq, inv)
    print(f"  Regression invariants passed: {reg_report.passed}")
    print(f"  Independent allocation check passed: {alloc_report.passed} "
          f"(max divergence: {alloc_report.max_divergence:.2e} m)")

    print("Synthesising fake signals from a known emissivity profile (Module 7)...")
    true_emissivity = np.linspace(1.0, 0.05, grid.n_shells)
    signals = inv.data @ true_emissivity

    print("Inverting with NNLS, TSVD and Tikhonov...")
    for solver in (
        fsci.NNLSSolver(),
        fsci.TSVDSolver(truncation_index=4),
        fsci.TikhonovSolver(lambda_param=0.05),
    ):
        result = solver.solve(inv.data, signals)
        print(f"  {result.method_name}: chi^2 = {result.chi_square:.3e}")

    print("\nDone. See fsci.visualization.plot_inversion_validation for the "
          "4-panel diagnostic dashboard (requires a display / notebook to view).")


if __name__ == "__main__":
    main()
