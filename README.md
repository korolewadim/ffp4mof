# FFP4MOF

FFP4MOF is a python library aimed at calculating force field precursors (FFPs) for metal-organic frameworks (MOFs) using a machine approach. It uses a CIF file as input data and produces a JSON file containing calculated FFPs, including partial charges, polarizabilities, dispersion coefficients, QDO parameters, and electron cloud parameters. The output file can be read using the pymatgen library.

```python
from ffp4mof.predict import get_ffps

get_ffps("filename.cif")
```

```python
from pymatgen import Structure

Structure.from_file("filename.json")
```

## Installation

FFP4MOF can be installed via the following command
```
pip install git+https://github.com/korolewadim/ffp4mof.git
```

The saved models are available at <a href="https://doi.org/10.5281/zenodo.5500642"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5500642.svg" alt="DOI"></a>

## Citation

If you find this code useful, please cite the following manuscript:

    Korolev, V. V., Nevolin, Y. M., Manz, T. A., & Protsenko, P. V. (2021).
    Parametrization of Non-Bonded Force Field Terms for Metal-Organic Frameworks Using Machine Learning Approach.
    arXiv preprint arXiv:2107.06044.
