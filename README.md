# FFP4MOF

FFP4MOF is a python library aimed at calculating force field precursors (FFPs) for metal-organic frameworks (MOFs) using a machine learning approach. It uses a CIF file as input data and produces a JSON file containing calculated FFPs, including partial charges, polarizabilities, dispersion coefficients, QDO parameters, and electron cloud parameters. The output file can be read using the [pymatgen](https://pymatgen.org) library, calculated FFPs are available via `as_dataframe()` method.

```python
from ffp4mof.predict import get_ffps

get_ffps("filename.cif")
```

```python
from pymatgen import Structure

structure = Structure.from_file("filename.json")
ffps = structure.as_dataframe()
```

## Installation

We strongly recommend the [Anaconda](https://www.anaconda.com) distribution of Python (3.9 release is required). You can create and activate a new conda environment with commands
```
conda create --name ffp4mof python=3.9
conda activate ffp4mof
```

FFP4MOF can be installed via the following command
```
pip install git+https://github.com/korolewadim/ffp4mof.git
```

The saved models are available at <a href="https://doi.org/10.5281/zenodo.5500642"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5500642.svg" alt="DOI"></a>. The unzipped `saved_models.zip` archive should be placed in the `ffp4mof` folder.

## Citation

If you find this code useful, please cite the following manuscript:

    Korolev, V. V., Nevolin, Y. M., Manz, T. A., & Protsenko, P. V. (2021).
    Parametrization of Nonbonded Force Field Terms for Metal–Organic Frameworks Using Machine Learning Approach.
    Journal of Chemical Information and Modeling, https://doi.org/10.1021/acs.jcim.1c01124.
