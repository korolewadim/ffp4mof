from os.path import join, dirname, abspath
from numpy import mean
from joblib import load as joblib_load
from pickle import load as pickle_load
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from ffp4mof.featurize import get_features


AVAILABLE_FORCE_FIELD_PRECURSORS = [
    "partial_charge",
    "fluctuating_polarizability",
    "FF_polarizability",
    "C6_coefficient",
    "QDO_mass",
    "QDO_charge",
    "QDO_frequency",
    "a_electron_parameter",
    "b_electron_parameter",
]


LOG10_FORCE_FIELD_PRECURSORS = [
    "fluctuating_polarizability",
    "FF_polarizability",
    "C6_coefficient",
]


def _get_ffp(features, ffp_type):
    scaler = joblib_load(join(dirname(abspath(__file__)), "scalers", ffp_type, "scaler.gz"))
    scaled_features = scaler.transform(features)
    targets = []

    for i in range(5):
        model = pickle_load(open(join(dirname(abspath(__file__)), "models", ffp_type, f"best_model_{i}.pickle"), "rb"))
        targets.append(model.predict(scaled_features))

    targets = mean(targets, axis=0)

    return targets


def get_ffps(filename, ffps_to_calc=None):
    a = AseAtomsAdaptor()
    structure = a.get_structure(read(filename))
    structure_name = filename.split("/")[-1][:-4]
    features = get_features(structure)
    ffps_dict = {}
    ffps_to_calc = AVAILABLE_FORCE_FIELD_PRECURSORS if ffps_to_calc is None else ffps_to_calc

    for ffp_type in ffps_to_calc:
        assert ffp_type in AVAILABLE_FORCE_FIELD_PRECURSORS
        ffp_values = _get_ffp(features, ffp_type)
        if ffp_type == "partial_charge":
            ffp_values = ffp_values - sum(ffp_values) / ffp_values.size
        if ffp_type in LOG10_FORCE_FIELD_PRECURSORS:
            ffp_values = 10 ** (ffp_values)
        structure.add_site_property(ffp_type, ffp_values.tolist())
        
    structure.to("json", f"{structure_name}.json")
