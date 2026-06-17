# Contributing to HolographicQC

## Development Setup

```bash
git clone https://github.com/ahmed19999520-alt/holographic-qc.git
cd holographic-qc
pip install -e ".[all,dev]"
pre-commit install
```

## Code Standards

All Python code must pass:

```bash
black holographic_qc/ tests/
isort holographic_qc/ tests/
pytest tests/ -v
```

No test may use mock data for physics assertions. All physical constants must
match SI values from CODATA 2018.

## Adding a New Material

1. Create `holographic_qc/materials/your_material.py`
2. Implement the interface: `fermi_velocity`, `bulk_gap_eV`, `central_charge`, `xi`,
   `t2_standard_ns(T)`, `t2_holographic_ns(T, L)`, `material_parameters_dict()`
3. Register in `holographic_qc/materials/__init__.py` and `MATERIALS_REGISTRY`
4. Add tests in `tests/test_materials.py`

## Adding a New Algorithm

1. Create `holographic_qc/algorithms/your_algorithm.py`
2. Register in `holographic_qc/algorithms/__init__.py`
3. Add tests in `tests/test_your_algorithm.py`
4. All gate counts and circuit depths must be computed from first principles

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit with descriptive messages
4. Open a pull request against `main`

## Reporting Issues

Include: platform, Python version, full traceback, minimal reproducible example.