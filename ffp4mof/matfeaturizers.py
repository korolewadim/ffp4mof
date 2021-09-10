"""
matminer Copyright (c) 2015, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with
the distribution.

(3) Neither the name of the University of California, Lawrence
Berkeley National Laboratory, U.S. Dept. of Energy nor the names of
its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance
of the source code ("Enhancements") to anyone; however, if you
choose to make your Enhancements available either publicly, or
directly to Lawrence Berkeley National Laboratory or its
contributors, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license:
a  non-exclusive, royalty-free perpetual license to install, use,
modify, prepare derivative works, incorporate into other computer
software, distribute, and sublicense such enhancements or derivative
works thereof, in binary and source code form.
"""

from numpy import zeros, concatenate
from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats
from matminer.utils.caching import get_all_nearest_neighbors

class VoronoiModifiedFingerprint(BaseFeaturizer):
    """
    Voronoi tessellation-based features around target site.
    Calculate the following sets of features based on Voronoi tessellation
    analysis around the target site:
    Voronoi indices
        n_i denotes the number of i-edged facets, and i is in the range of 3-10.
        e.g.
        for bcc lattice, the Voronoi indices are [0,6,0,8,...];
        for fcc/hcp lattice, the Voronoi indices are [0,12,0,0,...];
        for icosahedra, the Voronoi indices are [0,0,12,0,...];
    i-fold symmetry indices
        computed as n_i/sum(n_i), and i is in the range of 3-10.
        reflect the strength of i-fold symmetry in local sites.
        e.g.
        for bcc lattice, the i-fold symmetry indices are [0,6/14,0,8/14,...]
            indicating both 4-fold and a stronger 6-fold symmetries are present;
        for fcc/hcp lattice, the i-fold symmetry factors are [0,1,0,0,...],
            indicating only 4-fold symmetry is present;
        for icosahedra, the Voronoi indices are [0,0,1,0,...],
            indicating only 5-fold symmetry is present;
    Weighted i-fold symmetry indices
        if use_weights = True
    Voronoi volume
        total volume of the Voronoi polyhedron around the target site
    Voronoi volume statistics of sub_polyhedra formed by each facet + center
        stats_vol = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi area
        total area of the Voronoi polyhedron around the target site
    Voronoi area statistics of the facets
        stats_area = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi nearest-neighboring distance statistics
        stats_dist = ['mean', 'std_dev', 'minimum', 'maximum']
    Args:
        cutoff (float): cutoff distance in determining the potential
                        neighbors for Voronoi tessellation analysis.
                        (default: 6.5)
        use_symm_weights(bool): whether to use weights to derive weighted
                                i-fold symmetry indices.
        symm_weights(str): weights to be used in weighted i-fold symmetry
                           indices.
                           Supported options: 'solid_angle', 'area', 'volume',
                           'face_dist'. (default: 'solid_angle')
        stats_vol (list of str): volume statistics types.
        stats_area (list of str): area statistics types.
        stats_dist (list of str): neighboring distance statistics types.
    """

    def __init__(self, cutoff=6.5,
                 use_symm_weights=False, symm_weights='solid_angle',
                 stats_vol=None, stats_area=None, stats_dist=None):
        self.cutoff = cutoff
        self.use_symm_weights = use_symm_weights
        self.symm_weights = symm_weights
        self.stats_vol = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_vol is None else copy.deepcopy(stats_vol)
        self.stats_area = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_area is None else copy.deepcopy(stats_area)
        self.stats_dist = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_dist is None else copy.deepcopy(stats_dist)

    def featurize(self, n_w):
        """
        Get Voronoi fingerprints of site with given index in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Voronoi fingerprints.
                -Voronoi indices
                -i-fold symmetry indices
                -weighted i-fold symmetry indices (if use_symm_weights = True)
                -Voronoi volume
                -Voronoi volume statistics
                -Voronoi area
                -Voronoi area statistics
                -Voronoi dist statistics
        """

        # Prepare storage for the Voronoi indices
        voro_idx_list = zeros(8, int)
        voro_idx_weights = zeros(8)
        vol_list = []
        area_list = []
        dist_list = []

        # Get statistics
        for nn in n_w:
            if nn['poly_info']['n_verts'] <= 10:
                # If a facet has more than 10 edges, it's skipped here.
                voro_idx_list[nn['poly_info']['n_verts'] - 3] += 1
                vol_list.append(nn['poly_info']['volume'])
                area_list.append(nn['poly_info']['area'])
                dist_list.append(nn['poly_info']['face_dist'] * 2)
                if self.use_symm_weights:
                    voro_idx_weights[nn['poly_info']['n_verts'] - 3] += \
                        nn['poly_info'][self.symm_weights]

        symm_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_symm_weights:
            symm_wt_list = voro_idx_weights / sum(voro_idx_weights)
            voro_fps = list(concatenate((voro_idx_list, symm_idx_list,
                                           symm_wt_list), axis=0))
        else:
            voro_fps = list(concatenate((voro_idx_list,
                                           symm_idx_list), axis=0))

        voro_fps.append(sum(vol_list))
        voro_fps.append(sum(area_list))
        voro_fps += [PropertyStats().calc_stat(vol_list, stat_vol)
                     for stat_vol in self.stats_vol]
        voro_fps += [PropertyStats().calc_stat(area_list, stat_area)
                     for stat_area in self.stats_area]
        voro_fps += [PropertyStats().calc_stat(dist_list, stat_dist)
                     for stat_dist in self.stats_dist]
        return voro_fps
    
    def featurize_structure(self, struct):
        n_w = get_all_nearest_neighbors(VoronoiNN(cutoff=self.cutoff, allow_pathological=True), struct)
        features = [self.featurize(n_w[i]) for i in range(len(struct.sites))]
        return features

    def feature_labels(self):
        labels = ['Voro_index_%d' % i for i in range(3, 11)]
        labels += ['Symmetry_index_%d' % i for i in range(3, 11)]
        if self.use_symm_weights:
            labels += ['Symmetry_weighted_index_%d' % i for i in range(3, 11)]
        labels.append('Voro_vol_sum')
        labels.append('Voro_area_sum')
        labels += ['Voro_vol_%s' % stat_vol for stat_vol in self.stats_vol]
        labels += ['Voro_area_%s' % stat_area for stat_area in self.stats_area]
        labels += ['Voro_dist_%s' % stat_dist for stat_dist in self.stats_dist]
        return labels

    def citations(self):
        voronoi_citation = (
            '@book{okabe1992spatial,  '
            'title  = {Spatial tessellations}, '
            'author = {Okabe, Atsuyuki}, '
            'year   = {1992}, '
            'publisher = {Wiley Online Library}}')
        symm_idx_citation = (
            '@article{peng2011, '
            'title={Structural signature of plastic deformation in metallic '
            'glasses}, '
            'author={H L Peng, M Z Li and W H Wang}, '
            'journal={Physical Review Letters}, year={2011}}, '
            'pages = {135503}, volume = {106}, issue = {13}, '
            'doi = {10.1103/PhysRevLett.106.135503}}')
        nn_stats_citation = (
            '@article{Wang2019, '
            'title = {A transferable machine-learning framework linking '
            'interstice distribution and plastic heterogeneity in metallic '
            'glasses}, '
            'author = {Qi Wang and Anubhav Jain}, '
            'journal = {Nature Communications}, year = {2019}, '
            'pages = {5537}, volume = {10}, '
            'doi = {10.1038/s41467-019-13511-9}, '
            'url = {https://www.nature.com/articles/s41467-019-13511-9}}')
        return [voronoi_citation, symm_idx_citation, nn_stats_citation]

    def implementors(self):
        return ['Qi Wang']
