# igaops

**Matrix-free weighted quadrature operators for Isogeometric Analysis**

`igaops` provides reusable numerical building blocks for Isogeometric Analysis (IGA) based on **weighted quadrature** (WQ) and **matrix-free** (MF) techniques. The algorithms are largely based on developments carried out during the author's PhD thesis and on methods described in the published literature (see [Bibliography](#bibliography)).

The library is written in pure Python and relies only on `numpy`, `scipy`, and `geomdl`. It is designed to be **solver- and application-agnostic**: it exposes numerical operators rather than a complete simulation pipeline. Boundary conditions, material models, and problem-specific solvers are intentionally out of scope.

## What this package provides

- **Weighted quadrature rules** — computation of weights and points from univariate knot vectors.

- **L-mode products** — efficient tensor-product contractions between multi-dimensional arrays (sum-factorisation / L-mode matrix–vector products).

- **Matrix-free operators** — stiffness and mass matrix–vector products performed without explicitly assembling the global matrix, exploiting the tensor-product structure of IGA basis functions.

- **Matrix assembly** — classical sparse assembly for cases where an explicit matrix is needed, for example for direct solvers or debugging.

- **Preconditioners** — construction and application of the fast diagonalisation preconditioner for IGA, based on the Sylvester equation approach of Sangalli & Tani.

- **Geometry wrapper** — a thin class around `geomdl` objects that exposes matrix-free evaluation of the Jacobian, Hessian, and related geometric quantities needed by the operators above.

## Scope

`igaops` currently focuses on tensor-product B-spline and NURBS discretizations and provides low-level numerical tools that can be used to build IGA solvers and applications.

It does **not** aim to provide a complete finite element or simulation framework.

### What this package does not provide

Boundary condition handling, material libraries, load vectors, and time integration schemes are deliberately excluded. These components are application-specific and are best developed at a higher level, for example within [YETI](https://lamcos.insa-lyon.fr), the laboratory's Fortran/Python simulation framework.

## Requirements

- Python ≥ 3.10
- `numpy`
- `scipy`
- `geomdl`

The required dependencies are installed automatically.

Optional dependencies for post-processing and visualisation:

- `matplotlib`
- `pandas`
- `seaborn`

It is also recommended to install [ParaView](https://www.paraview.org/) for `.vtk` output visualisation.

## Installation

Install the latest released version from PyPI:

```bash
pip install igaops
```

To include the optional visualisation dependencies:

```bash
pip install "igaops[viz]"
```

For a development install from a local clone:

```bash
git clone https://github.com/JoaquinCORNEJO/igaops.git

cd igaops

pip install -e ".[viz]"
```

## Development

This project uses [Black](https://github.com/psf/black) for code formatting, [mypy](https://mypy.readthedocs.io/) for static type checking, and `pytest` for testing.

For a local development environment:

```bash
pip install -e ".[dev]"
```

Format the source code with:

```bash
black src/
```

Run static type checking with:

```bash
mypy src/
```

Run the test suite with:

```bash
pytest
```

## Citation

If you use `igaops` in academic work, please cite the relevant publications and/or thesis describing the methods implemented in the library.

## License

`igaops` is distributed under the **GNU Lesser General Public License v2.1 or later (LGPL-2.1-or-later)**.

See the [LICENSE](LICENSE) file for the complete license text.

## Contact

Questions and feedback are welcome:

[joaquin.cofu@gmail.com](mailto:joaquin.cofu@gmail.com)

## Bibliography

The algorithms implemented in `igaops` are based on the following references.

### B-Splines and NURBS

- L. Piegl — *The NURBS Book*

- J. Cottrell, T. J. R. Hughes, Y. Bazilevs — *Isogeometric Analysis: Toward Integration of CAD and FEA*

### Weighted quadrature and matrix-free methods

- F. Calabrò, G. Sangalli, M. Tani — *Fast formation of isogeometric Galerkin matrices by weighted quadrature*

- G. Sangalli, M. Tani — *Matrix-free weighted quadrature for a computationally efficient isogeometric k-method*

- R. Hiemstra et al. — *Fast formation and assembly of finite element matrices with application to isogeometric linear elasticity*

### Preconditioners and fast diagonalisation

- G. Sangalli, M. Tani — *Isogeometric preconditioners based on fast solvers for the Sylvester equation*

- M. Montardini — *Preconditioners for isogeometric analysis* (PhD thesis)

### Tensor products and sum-factorisation

- T. Kolda, B. Bader — *Tensor decompositions and applications*

- P. Antolin et al. — *Efficient matrix computation for tensor-product isogeometric analysis*