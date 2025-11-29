# JAXIGA

JAX-based Isogeometric Analysis library for solving partial differential equations.

## Installation

Install the package in editable mode using pip:

```bash
pip install -e .
```

This will install the `jaxiga` package along with its dependencies:
- jax
- jaxlib
- numpy
- scipy
- matplotlib
- tqdm

## Package Structure

```
JAXIGA_pv/
├── src/jaxiga/           # Main package directory
│   ├── utils/            # Core utility functions
│   └── utils_iga/        # IGA-specific utilities
├── examples/             # Example scripts
│   ├── poisson/          # Poisson equation examples
│   ├── darcy/            # Darcy flow examples
│   ├── linear_elasticity/# Linear elasticity examples
│   └── phase_field/      # Phase field examples
└── pyproject.toml        # Package configuration
```

## Usage

After installation, you can import modules from the `jaxiga` package:

```python
from jaxiga.utils.bfgs import minimize as bfgs_minimize
from jaxiga.utils.preprocessing_DEM_1d import generate_quad_pts_weights_1d
from jaxiga.utils_iga.materials import MaterialElast2D
```

## Running Examples

Navigate to any example directory and run the scripts:

```bash
cd examples/poisson
python Poisson1D_DEM.py
```

All examples have been updated to use the installed `jaxiga` package.