from os.path import join, dirname, abspath
from itertools import combinations
from json import load
from pandas import read_csv
from numpy import array, vstack, hstack
from numpy import zeros, fill_diagonal, where
from pymatgen.core.periodic_table import Element
from ase.io import read
from ase.data import atomic_numbers, covalent_radii, cccbdb_ip
from matminer.featurizers.site import CrystalNNFingerprint, AGNIFingerprints, OPSiteFingerprint

from ffp4mof.matfeaturizers import VoronoiModifiedFingerprint


def get_op_site_fingerprints(structure):
    opsite = OPSiteFingerprint()
    opsite_fingerprints = vstack([opsite.featurize(structure, i) for i in range(len(structure))])
    return opsite_fingerprints


def get_voronoi_fingerprints(structure):
    voronoi_fingerprints = array(VoronoiModifiedFingerprint().featurize_structure(structure))
    return voronoi_fingerprints


def get_agni_fingerprints(structure):
    agni = AGNIFingerprints(directions=(None,))
    agni_fingerprints = vstack([agni.featurize(structure, i) for i in range(len(structure))])
    return agni_fingerprints


def get_crystal_nn_fingerprints(structure):
    crystal_nn = CrystalNNFingerprint.from_preset('cn')
    crystal_nn_fingerprints = vstack([crystal_nn.featurize(structure, i) for i in range(len(structure))])
    return crystal_nn_fingerprints


def get_adj_dist_matrices(structure, tol=0.5):
    dist_matrix = structure.distance_matrix
    adj_matrix = zeros(dist_matrix.shape)
    sites = [site for site in structure]

    for i, j in combinations(range(len(structure)), 2):
        if dist_matrix[i][j] < 6.1:
            element_1 = Element(sites[i].specie.symbol)
            element_2 = Element(sites[j].specie.symbol)
            
            max_distance = covalent_radii[atomic_numbers[element_1.symbol]] + \
            covalent_radii[atomic_numbers[element_2.symbol]] + tol
                
            if dist_matrix[i][j] < max_distance:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1

    fill_diagonal(adj_matrix, 0)

    return array(adj_matrix, dtype=int), dist_matrix


def get_site_descrs(structure):
    adj, dist = get_adj_dist_matrices(structure)
    ionization_energies = [Element.from_Z(z).ionization_energy for z in structure.atomic_numbers]
    electronegativities = [ELECTRONEGATIVITIES_DICT[str(z)] for z in structure.atomic_numbers]

    site_descrs = []

    for i in range(len(structure)):
        zero_sphere, first_sphere, second_sphere = [], [], []

        zero_sphere.append(ionization_energies[i])
        zero_sphere.append(electronegativities[i])

        first_neighbors = where(adj[i] > 0)[0].tolist()
        first_sphere.append(len(first_neighbors))
        first_sphere.append(sum([ionization_energies[n] for n in first_neighbors]) / len(first_neighbors))
        first_sphere.append(sum([electronegativities[n] for n in first_neighbors]) / len(first_neighbors))
        first_sphere.append(sum([dist[i][n] for n in first_neighbors]) / len(first_neighbors))

        second_neighbors = set(hstack([where(adj[n] > 0)[0].tolist() for n in first_neighbors]).tolist())
        second_neighbors = list(second_neighbors - set(first_neighbors) - set([i]))
        second_sphere.append(len(second_neighbors))
        second_sphere.append(sum([ionization_energies[n] for n in second_neighbors]) / len(second_neighbors))
        second_sphere.append(sum([electronegativities[n] for n in second_neighbors]) / len(second_neighbors))
        second_sphere.append(sum([dist[i][n] for n in second_neighbors]) / len(second_neighbors))

        all_spheres = zero_sphere + first_sphere + second_sphere
        site_descrs.append(all_spheres)

    site_descrs = array(site_descrs)
    return site_descrs


def get_features(structure):
    features = hstack((
        get_agni_fingerprints(structure),
        get_crystal_nn_fingerprints(structure),
        get_site_descrs(structure),
        get_op_site_fingerprints(structure),
        get_voronoi_fingerprints(structure),
    ))
    
    return features


with open(join(dirname(abspath(__file__)), "electronegativities.json")) as json_file:
    ELECTRONEGATIVITIES_DICT = load(json_file)
