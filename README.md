# FSCI — Flux-Surface Constrained Inversion (TCABR)

[![CI](https://github.com/dougnovaes/tcabr-fsci/actions/workflows/ci.yml/badge.svg)](https://github.com/dougnovaes/tcabr-fsci/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pipeline that reconstructs the radial emissivity profile of the
TCABR tokamak plasma from a multichannel spectrometer, using the real
magnetic equilibrium (GEQDSK) instead of the circular-shell assumption
of classic onion-peeling / Pearce inversion.

## Why not classic tomography?

TCABR's diagnostic is a single fan of 32 equatorial lines of sight —
one viewing angle, not a multi-angle projection set. Under that
constraint, 2D pixel-based tomography is severely underdetermined.
FSCI instead assumes what plasma physics already guarantees: particle
and heat transport along magnetic field lines is fast enough that
emissivity is approximately constant on a flux surface, ε = ε(ψ_N).
That collapses the unknowns from thousands of 2D pixels to a few dozen
1D flux shells — turning an intractable inverse problem into an
overdetermined (but still ill-conditioned) one, solved here with
regularised methods (Tikhonov, TSVD, NNLS) rather than direct
inversion.

This is a specific implementation of a method family established in
the fusion-diagnostics literature since the 1980s under names such as
*generalized/asymmetric Abel inversion* (for non-circular flux
surfaces) and *flux-surface-constrained profile inversion* — see
`docs/` or the project history for literature pointers. "FSCI" is this
repository's own name for its particular implementation, not a
claim of a new method.

## Architecture

```
                 Core Geometry Utilities
        (ray-tracing, LCFS masking, equatorial limits)
                          │
            ┌─────────────┼─────────────┐
            │                           │
       Module 5                    Module 6
   (build matrix G)           (visual validation)
            │                           │
            └─────────────┬─────────────┘
                           │
                      Module 7
                      (inversion)
```

| Module | File | Responsibility |
|---|---|---|
| M1 | `profiles.py` | Physical profiles (n_e, T_e, ...) vs. ψ_N |
| M2 | `equilibrium.py` | GEQDSK read: ψ_N(R,Z), LCFS, magnetic axis |
| M3 | `flux_grid.py` | Discretisation of ψ_N into inversion shells |
| M4 | `optics.py` | 32-channel line-of-sight geometry |
| Core | `geometry_core.py` | Ray tracing shared by M5 and M6 |
| M5 | `inversion_matrix.py` | Geometric transfer matrix G |
| M6 | `visualization.py` | 4-panel validation dashboard |
| M7 | `solvers.py` | NNLS / TSVD / Tikhonov inversion |
| — | `validation.py` | Regression invariants + independent allocation check |
| — | `synthetic.py` | Synthetic equilibrium/optics for testing & demos |

### Physics axiom (see `fsci/optics.py`)

All lines of sight are assumed strictly equatorial (Z = 0) and
perpendicular to their nominal measurement radii. This licenses
reducing the 3D diagnostic geometry to the 2D equatorial plane — it is
a **design assumption** of the TCABR diagnostic, not something derived
or verified by the code.

### Engineering notes worth knowing before modifying this code

- **Matrix orientation**: `psi_N` is stored natively as `(len(R), len(Z))`
  throughout the domain layer. `.T` is used *only* inside plotting
  calls. An earlier attempt to "unify" this by transposing at read
  time broke the pipeline silently, because TCABR's square (257×257)
  mesh makes a shape-based orientation check vacuously true. Do not
  reintroduce a shape-based orientation test.
- **`length_in_plasma`** in `InversionMatrix` is defined as `sum(G[j,:])`,
  not an independent geometric count — see `validation.py` for the
  genuinely independent cross-check.
- **`run_independent_allocation_validation`** is independent of the
  ψ_N interpolator, but *not* independent of the ray-tracing geometry
  itself (it shares `calculate_ray_limits`/`generate_lcfs_mask` with
  Module 5). A fully independent geometric check would require a
  second, unrelated ray-tracing implementation.

## Status

- Modules 1–7 implemented and covered by a regression test suite based
  on physical invariants (positivity, length conservation, ψ_N ≈ 0 at
  the magnetic axis, core-vs-edge monotonicity) rather than
  hardcoded reference values — these should hold for any valid GEQDSK
  file, not just TCABR's current equilibrium.
- **`S_HARDWARE`** (the 32 channels' real impact parameters) is not
  yet wired in: the pipeline currently runs against synthetic optics
  for testing/demo purposes. Real Zemax-derived hardware coordinates
  are pending integration.
- No experimental validation against real TCABR discharge data yet.

## Installation

```bash
git clone https://github.com/dougnovaes/tcabr-fsci.git
cd tcabr-fsci
pip install -e ".[dev]"        # add [interactive] too for the Jupyter dashboard
```

## Quick start

```bash
python examples/run_pipeline_example.py
```

This runs the whole pipeline — equilibrium, discretisation, optics,
ray tracing, matrix construction, validation, and all three inversion
solvers — against a synthetic circular equilibrium, so it works
without any TCABR data file. To run against a real discharge:

```python
import fsci

eq = fsci.load_equilibrium("path/to/g000000.eqdsk")
optics = fsci.build_optical_geometry(xc, yc, S_HARDWARE)  # your real hardware array
grid = fsci.build_flux_grid(n_shells=20)
inv = fsci.build_geometric_matrix(optics, grid, eq)

result = fsci.TikhonovSolver(lambda_param=0.05).solve(inv.data, measured_signals)
```

## Testing

```bash
pytest tests/ -v
```

All tests run against a synthetic circular equilibrium
(`fsci.synthetic`) — no proprietary TCABR data is required or
committed to this repository (see `.gitignore`).

## License

MIT — see [LICENSE](LICENSE).
