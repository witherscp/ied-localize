"""Subject class"""

from glob import glob
from itertools import product
from math import isnan
from warnings import filterwarnings

import numpy as np
import pandas as pd

from .constants import *
from .helpers import *

filterwarnings(action="ignore", message='All-NaN slice encountered')

class Subject:

    def __init__(self, subj, n_parcs=600, n_networks=17, max_length=10,
                 dist=45, dual_spike=False, use_weighted=False):

        # set values
        self.subj = subj
        self.parcs = n_parcs
        self.networks = n_networks
        self.seq_len = max_length
        self.dist = dist

        # set subject-specific directories
        self.dirs = {
            k.lower()[:-4]: (v / self.subj) for k,v in data_directories.items()
        }
        if dual_spike:
            self.dirs['ied'] = Path(str(data_directories['IED_DIR']) + "_dualSpike") / self.subj
        self._update_ied_subdirs()
        self._update_mri_subdirs()

        # get arrays and dataframes that are constant
        self.elec_labels_df = self._fetch_elec_labels_df()
        self.elec2parc_df = self._fetch_elec2parc_df()
        self.elec2lobe_df = self._fetch_elec2lobe_df()
        self.parc_minEuclidean_byElec = self._fetch_parc_minEuclidean_byElec()
        self.node2parc_df_dict = self._fetch_node2parc_df_dict()
        self.elec2hemi_df = self._fetch_elec2hemi_df()
        self.elec_euc_arr = self._fetch_elec_euc_arr()

        # update attributes
        self._update_num_clusters()
        self._update_cluster_num_sequences()
        self._update_cluster_types()
        self._update_source_parcels(dist=dist, use_weighted=use_weighted)
        self._update_engel_class()

        if self.engel_class not in ("no_resection","no_outcome","deceased"):
            self.node2rsxn_df_dict = self._fetch_node2rsxn_df_dict()

    def _update_ied_subdirs(self):
        """Add IED subdirectories."""

        dir_name = f"Schaefer_{self.parcs}P_{self.networks}N"

        for val, dir in [('seqs', 'sequence_classification'),
                         ('raw_fc', 'raw_fc'),
                         ('sc', f'sc/{dir_name}'),
                         ('source_loc', f'source_localization/{dir_name}')]:

            self.dirs[val] = self.dirs['ied'] / dir

    def _update_mri_subdirs(self):
        """Add MRI subdirectories."""

        for val, dir in [('surf', 'surf/xhemi/std141/orig'),
                         ('general', 'surf/xhemi/std141/orig/general'),
                         ('align_elec_alt', f'icEEG/align_elec_alt')]:

            self.dirs[val] = self.dirs['mri'] / dir

    def _fetch_elec2hemi_df(self):
        """Fetch elec_to_hemi dataframe to store as self.elec2hemi_df.

        Returns:
            pd.DataFrame: df with columns 'elec' and 'hemi'
        """

        fpath = self.dirs['sc'] / "elec_to_hemi.csv"
        return pd.read_csv(fpath)

    def _fetch_elec_euc_arr(self):
        """Fetch elec_euc_arr to store as self.elec_euc_arr.
        
        Returns:
            np.array: n_elec x n_elec array of Euclidean distances
        """

        fpath = self.dirs['sc'] / "elec_euc_withinHemi.csv"
        arr = np.loadtxt(fpath, dtype=float)
        arr[np.diag_indices_from(arr)] = np.NaN
        
        return arr

    def fetch_sequences(self, cluster=None):
        """Fetch electrode sequences and lag times once they have been saved
        as .csv files in self.dirs['seqs'].

        Args:
            cluster (int, optional): Cluster of interest. Defaults to None.

        Returns:
            seqs, delays: np.array of electrode names and np.array of delay
                times for every sequence
        """

        assert isinstance(cluster, (int, type(None)))

        suffix = ''
        if isinstance(cluster, int):
            suffix = f"_cluster{cluster}"

        seqs_file = self.dirs['seqs'] / (f"elecSequences_max{self.seq_len}"
                                         f"{suffix}.csv")
        seqs = np.loadtxt(seqs_file, dtype=str, delimiter=",")

        delays_file = self.dirs['seqs'] / (f"delaySequences_max{self.seq_len}"
                                           f"{suffix}.csv")
        delays = np.loadtxt(delays_file, dtype=float,delimiter=",")

        return seqs, delays

    def _fetch_parc_minEuclidean_byElec(self):
        """Fetch array of minimum Euclidean distances between parcels and
        electrodes.

        Returns:
            np.array: array Euclidean distances with shape (n_parcs, n_elecs)
        """

        fpath = self.dirs['sc'] / "parc_minEuclidean_byElec.csv"
        return np.loadtxt(fpath, delimiter=',', dtype=float)

    def _fetch_node2parc_df_dict(self):
        """Fetch node to parcel look-up table for given hemisphere.

        Returns:
            dict: dictionary of dataframes with "node" and "parcel" columns for
                each hemisphere
        """

        node2parc_df_dict = {}

        for hemi, use_idx in [("LH", 1),
                              ("RH", 2)]:

            # load node2parc df
            node2parc_path = self.dirs['sc'] / "node_to_parc.csv"
            node2parc_df = pd.read_csv(node2parc_path, usecols=[0,use_idx],
                                       names=['node', 'parcel'], skiprows=1)

            # add to dict
            node2parc_df_dict[hemi] = node2parc_df

        return node2parc_df_dict

    def _fetch_elec2parc_df(self):
        """Fetch elec to parcel look-up table.

        Returns:
            pd.DataFrame: dataframe with "elecLabel" and "parcNumber" columns
        """

        return pd.read_csv((self.dirs['sc'] / "elec_to_parc.csv"))

    def _fetch_elec2lobe_df(self):
        """Fetch elec to lobe look-up table.

        Returns:
            pd.DataFrame: dataframe with "elecLabel" and "Lobe" columns
        """

        return pd.read_csv((self.dirs['sc'] / "elec_to_lobe.csv"))

    def _fetch_elec_labels_df(self):
        """Fetch df of all electrode names for conversion to index.

        Returns:
            pd.DataFrame: dataframe with "chanName" as column
        """

        # load order of electrodes
        fpath = self.dirs['sc'] / "elec_labels.csv"
        return pd.read_csv(fpath, header=None, names=["chanName"])

    def fetch_normalized_parc2prop_df(self, cluster, dist=45,
                                      only_geo=False, only_wm=False):
        """Fetch df with conversion table of parcel number to proportion of
        sequences explained.

        Args:
            cluster (int): number of cluster
            dist (int, optional): Geodesic search distance in mm. Defaults to
                45.
            only_geo (bool, optional): Use geodesic only method. Defaults to
                False.
            only_wm (bool, optional): Use white matter only method. Defaults to
                False.

        Returns:
            pd.DataFrame: dataframe with index as parcel number and column as
                'propExplanatorySpikes'
        """

        # if not using combination method, only one input can be set to True
        assert not (only_geo and only_wm)

        method = ""
        if only_geo:
            method = "_geodesic"
        elif only_wm:
            method = "_whiteMatter"

        fname = (f"*{method}_normalizedCounts_within{dist}_max{self.seq_len}"
                 f"_cluster{cluster}.csv")
        fpath_lst = glob(str(self.dirs['source_loc'] / fname))

        if not (only_geo or only_wm):
            for fpath in fpath_lst:
                if ("whiteMatter" in fpath) or ("geodesic" in fpath):
                    continue
                else:
                    parc2prop_path = fpath
                    break
        else:
            parc2prop_path = fpath_lst[0]

        df = pd.read_csv(parc2prop_path)

        return df.set_index("parcNumber")

    def _fetch_node2rsxn_df_dict(self):
        """Create dictionary of node to resection lookup tables for each 
        hemisphere.

        Returns:
            dict: dictionary with keys = hemi and values = dataframe
        """

        # create dictionary of node2rsxn_df for each hemisphere of interest
        node2rsxn_df_dict = {}

        for hemi in "LH", "RH":
            node2rsxn_path = self.dirs['sc'] / f"{hemi.upper()}_node_to_resection.txt"
            node2rsxn_df = pd.read_csv(node2rsxn_path,
                                       delim_whitespace=True,
                                       header=None,
                                       names=['node','is_resected']
                                    )
            node2rsxn_df_dict[hemi] = node2rsxn_df

        return node2rsxn_df_dict

    def fetch_geodesic_travel_times(self):
        """Fetch geodesic travel times based on geodesic velocities and
        distances from electrodes to nodes on the std.141 mesh.

        Returns:
            tuple: minGeo_maxSpeed_time, minGeo_minSpeed_time,
                maxGeo_minSpeed_time, maxGeo_maxSpeed_time (four np.arrays
                with shape (n_nodes, n_elecs))
        """

        # load and compute estimated lag times based on Geodesic distances

        fdir = self.dirs['sc']
        temp_minGeo = pd.read_csv((fdir / "node_minGeo_byElec.csv"), header=None)
        temp_maxGeo = pd.read_csv((fdir / "node_maxGeo_byElec.csv"), header=None)

        minGeo = temp_minGeo.to_numpy(copy=True)
        maxGeo = temp_maxGeo.to_numpy(copy=True)

        # set all minimum values > dist to np.NaN and all max values to dist that
        # have a min value less than dist
        minGeo[minGeo > self.dist] = np.NaN
        maxGeo[np.isnan(minGeo)] = np.NaN
        maxGeo[maxGeo > self.dist] = self.dist

        minGeo_maxSpeed_time = minGeo / MAX_GEO_VEL
        minGeo_minSpeed_time = minGeo / MIN_GEO_VEL
        maxGeo_minSpeed_time = maxGeo / MIN_GEO_VEL
        maxGeo_maxSpeed_time = maxGeo / MAX_GEO_VEL

        return (minGeo_maxSpeed_time,
                minGeo_minSpeed_time,
                maxGeo_minSpeed_time,
                maxGeo_maxSpeed_time
        )

    def fetch_wm_travel_times(self):
        """Fetch white matter min/max BL times.

        Returns:
            tuple: minBL_time, maxBL_time (np.arrays of shape
                (n_parcs, n_parcs))
        """

        # load and compute estimated lag times based on WM bundle lengths
        fdir = self.dirs['sc']
        BL = np.loadtxt((fdir / "BL.csv"), delimiter=",", dtype=float)
        sBL = np.loadtxt((fdir / "sBL.csv"), delimiter=",", dtype=float)

        BL[BL == 0] = np.nan
        sBL[sBL == 0] = np.nan

        maxBL = BL + sBL
        minBL = BL - sBL
        if minBL[minBL < 0].size > 0:
            minBL[minBL < 0] = 0

        maxBL_time = maxBL / MIN_WM_VEL
        minBL_time = minBL / MAX_WM_VEL
        minBL_time[np.diag_indices_from(minBL_time)] = 0
        maxBL_time[np.diag_indices_from(maxBL_time)] = 20

        return minBL_time, maxBL_time

    def _update_num_clusters(self):
        """Update self.num_clusters value."""

        # load cluster summary df
        in_file = f"cluster_summary_max{self.seq_len}.csv"
        try:
            df = pd.read_csv((self.dirs['seqs'] / in_file))
            self.num_clusters = max(set(df['Cluster Number']))
        except FileNotFoundError:
            self.num_clusters = "Clustering has not been run"

    def _update_cluster_types(self):
        """Update self.valid_clusters and self.soz_clusters."""

        # check that clustering is complete
        assert isinstance(self.num_clusters, int)

        # load bad_clusters
        in_file = "ied_bad_clusters.csv"
        bad_df = pd.read_csv((data_directories['IED_ANALYSIS_DIR'] / in_file))

        # update self.valid_clusters
        try:
            subj_bad_clusters = bad_df.loc[bad_df.subject == self.subj]['badClusters'].iloc[0]
            bad_clusters = subj_bad_clusters.strip("][").split(',')
            if bad_clusters[0] == '':
                bad_clusters = []
            else:
                bad_clusters = [int(clust) for clust in bad_clusters]
            valid_clusters = [
                i for i in range(1,self.num_clusters+1) if i not in bad_clusters
            ]
            self.valid_clusters = [i for i in valid_clusters if self.cluster_nseqs[i] >= 100]
        except IndexError:
            self.valid_clusters = 'Subject not found in ied_bad_clusters.csv'

        # load soz_clusters
        in_file = "ied_soz_clusters.csv"
        soz_df = pd.read_csv((data_directories['IED_ANALYSIS_DIR'] / in_file))

        # update self.soz_clusters
        try:
            subj_soz_clusters = soz_df.loc[soz_df.subj == self.subj]['sozClusters'].iloc[0]
            soz_clusters = subj_soz_clusters.strip("][").split(',')
            if soz_clusters[0] == '':
                self.soz_clusters = []
            else:
                self.soz_clusters = [int(clust) for clust in soz_clusters if self.cluster_nseqs[int(clust)] >= 100]
        except IndexError:
            self.soz_clusters = 'Subject not found in ied_soz_clusters.csv'

    def _update_cluster_num_sequences(self):
        """Update self.cluster_nseqs value."""

        fpath = self.dirs['seqs'] / f"cluster_summary_max{self.seq_len}.csv"
        df = pd.read_csv(fpath)

        cluster_seq_dict = {}
        for cluster in range(1,self.num_clusters+1):
            n_seqs = df.loc[df['Cluster Number'] == cluster,
                            'Number of Sequences'].iloc[0]
            cluster_seq_dict[cluster] = n_seqs

        self.cluster_nseqs = cluster_seq_dict

    def _update_source_parcels(self, dist=45, use_weighted=False):
        """Update self.valid_sources_all with a set of every possible source
        for each cluster. Update self.valid_sources_one with a single source
        for each cluster (choosing the one that is closest to the most frequent
        lead electrode).

        Args:
            dist (int, optional): geodesic search distance. Defaults to 45.
        """


        valid_sources_all, valid_sources_one = {}, {}

        for cluster in self.valid_clusters:

            parc2prop_df = self.fetch_normalized_parc2prop_df(cluster, dist=dist)

            # get all top source parcels (within 5% of the top source)
            top_parc = parc2prop_df.sort_values(by=['propExplanatorySpikes'],
                                                ascending=False).head(1)
            top_proportion = top_parc['propExplanatorySpikes'].iloc[0]

            if top_proportion > 0.5:
                valid_sources_all[cluster] = set(
                    parc2prop_df.loc[
                        parc2prop_df['propExplanatorySpikes'] > (top_proportion - .05)
                    ].index
                )
            else:
                valid_sources_all[cluster] = set()

            # as a tiebreaker, choose source parcel closest to most frequent
            # leading electrode
            if len(valid_sources_all[cluster]) > 1:

                if use_weighted:
                    min_dist = np.Inf
                    for source in valid_sources_all[cluster]:
                        d = self.compute_weighted_source2elec_dist(cluster,
                                                                   source=source,
                                                                   lead_only=True,
                                                                   use_all_seqs=True)
                        if min_dist > d:
                            min_dist = d
                            best_source = source

                    valid_sources_one[cluster] = set([best_source])
                else:
                    # compute most frequent lead electrode
                    seqs, _ = self.fetch_sequences(cluster)
                    lead = compute_top_lead_elec(seqs)

                    # get lead index
                    lead_idx = self.get_elec_idx(lead)

                    # find closest parcel to lead_idx
                    top_parc_idxs = [parc - 1 for parc in valid_sources_all[cluster]]
                    min_loc = np.argmin(self.parc_minEuclidean_byElec[top_parc_idxs, lead_idx])
                    valid_sources_one[cluster] = set([top_parc_idxs[min_loc] + 1])

            else:
                valid_sources_one[cluster] = valid_sources_all[cluster]

        self.valid_sources_all = valid_sources_all
        self.valid_sources_one = valid_sources_one

    def _update_engel_class(self):
        """Update self.engel_class and self.engel_months."""


        # load hemi df
        engel_fpath = data_directories['IED_ANALYSIS_DIR'] / "ied_subj_engelscores.csv"
        engel_df = pd.read_csv(engel_fpath)

        engel_dict = {'MoreThan24 Engel Class': 'MoreThan24 Months',
                      'Mo24 Engel Class': 24,
                      'Mo12 Engel Class': 12
                    }

        # iterate through engel_df columns from longest time to shortest
        for col, time in engel_dict.items():
            # get class from df
            engel_class = engel_df[engel_df["Patient"] == self.subj][col].iloc[0]

            # set months
            if type(time) is str:
                engel_months = engel_df[engel_df["Patient"] == self.subj][time].iloc[0]
            elif engel_class in ["no_resection","no_outcome","deceased"]:
                engel_months = np.nan
            else:
                engel_months = time

            # if class is not left blank then set class and months
            try:
                if isnan(engel_class):
                    continue
            except TypeError:
                self.engel_class = engel_class
                self.engel_months = engel_months
                return

        # since none of the class columns were filled-in, set months to np.nan
        engel_months = np.nan

        self.engel_class = engel_class
        self.engel_months = engel_months

    def get_elec_idx(self, elec):
        """Get index of electrode using self.elec_labels_df.

        Args:
            elec (str): electrode name

        Returns:
            int: electrode index
        """

        return self.elec_labels_df[self.elec_labels_df['chanName'] == elec].index[0]

    def compute_lead_elec_parc2prop_df(self, cluster):
        """Compute a lookup table of parcel to proportion explained for leading
        electrodes.

        Args:
            cluster (int): cluster number

        Returns:
            pd.DataFrame: dataframe with index = 'parcNumber' and
                column = 'propExplanatorySpikes'
        """

        elec_seqs, _ = self.fetch_sequences(cluster)
        lead_elecs = elec_seqs[:,0]

        # use normalized parc2prop file as template for lead_elec_df
        lead_elec_df = self.fetch_normalized_parc2prop_df(cluster)
        lead_elec_df['propExplanatorySpikes'] = 0

        parc_counts = {}
        for elec in lead_elecs:
            # get electrode parcel indices
            parc_idxs = convert_elec_to_parc(self.elec2parc_df, elec)

            # update count for each parcel
            for parc_idx in parc_idxs:
                parc_counts.setdefault(parc_idx,0)
                parc_counts[parc_idx] += 1

        # normalize based on total number of sequences
        n_seqs = elec_seqs.shape[0]
        normalized_counts = {parc:(count / n_seqs) for parc, count in parc_counts.items()}

        # iterate through normalized counts
        for parc_idx, prop in normalized_counts.items():
            # update lead_elec_df proportion
            lead_elec_df.iloc[parc_idx] = prop

        return lead_elec_df

    def compute_resected_prop(self, parcels):
        """Retrieve the resected proportion of a parcel for a given hemisphere.

        Args:
            parcels (list): list of int parcels in range (1,self.parcs+1)

        Returns:
            list: list of float proportions of parcels resected
        """

        resected_props = []

        for parcel in parcels:

            hemi = get_parcel_hemi(parcel, self.parcs)
            if hemi == "RH":
                parcel -= int(self.parcs / 2)

            # get hemi_specific df
            hemi_node2parc = self.node2parc_df_dict[hemi]
            hemi_node2rsxn = self.node2rsxn_df_dict[hemi]

            # mask out nodes for parcel and find proportion of nodes resected
            parc_nodes = hemi_node2parc[hemi_node2parc.parcel == parcel].index
            resected_prop = hemi_node2rsxn.iloc[parc_nodes].mean()['is_resected']

            # add to list
            resected_props.append(resected_prop)

        return resected_props

    def compute_rsxn_source_arr(self, n_cluster, all_sources=False):
        """Create n_node array with a gray resection zone and green TN/TP
        regions or red FN/FP parcels.

        Args:
            n_cluster (int): cluster number
            all_sources (bool, optional): plot every possible source 
                (not just the single best source). Defaults to False.

        Returns:
            tuple: np.array of values for creating niml.dset, str hemisphere
        """


        if all_sources:
            sources = self.valid_sources_all[n_cluster]
        else:
            sources = self.valid_sources_one[n_cluster]

        hemi = get_parcel_hemi(list(sources)[0], self.parcs)

        # retrieve array of 0s (non-resected) and 1s (resected)
        rsxn_arr = self.node2rsxn_df_dict[hemi.upper()]['is_resected'].to_numpy(dtype=float).copy()

        # get node2parc_df
        node2parc_df = self.node2parc_df_dict[hemi.upper()]

        # get resection props
        rsxn_props = self.compute_resected_prop(sources)

        for source,rsxn_prop in zip(sources,rsxn_props):

            if hemi.lower() == "rh":
                mask = (node2parc_df['parcel'] == (source - int(self.parcs/2)))
            else:
                mask = (node2parc_df['parcel'] == source)

            accuracy = get_prediction_accuracy(self.engel_class, rsxn_prop)

            # set values depending on concordance
            if accuracy.startswith('T'):
                rsxn_arr[mask] = 2.6
            elif accuracy.startswith('F'):
                rsxn_arr[mask] = 4

        return rsxn_arr, hemi

    def compute_localizing_seq_idxs(self, cluster, source, only_geo=False,
                                    only_wm=False):
        """Return an array of indices for which a source successfully localizes
        the sequences of a given cluster.

        Args:
            cluster (int): cluster number
            source (int): source parcel
            only_geo (bool, optional): Use geodesic only localization method.
                Defaults to False.
            only_wm (bool, optional): Use white matter only localization 
                method. Defaults to False.

        Returns:
            np.array: array of indices for which the sequences localize to a 
                given source (find the sequences with seqs[seq_indices])
        """

        method = ""
        if only_geo:
            method = "_geodesic"
        elif only_wm:
            method = "_whiteMatter"

        fname = (f"*{method}_sourceParcels_within{self.dist}_max{self.seq_len}"
                 f"_cluster{cluster}.csv")
        fpath_lst = glob(str(self.dirs['source_loc'] / fname))

        if not (only_geo or only_wm):
            for fpath in fpath_lst:
                if ("whiteMatter" in fpath) or ("geodesic" in fpath):
                    continue
                else:
                    parc2prop_path = fpath
                    break
        else:
            parc2prop_path = fpath_lst[0]

        seq_sources = np.loadtxt(parc2prop_path, dtype=float, delimiter=",")

        # return all indices where sequence localized to source
        return np.squeeze(np.argwhere(np.sum(seq_sources == source, axis=1)),
                          axis=1)

    def get_cluster_lobe(self, cluster):
        """Return the majority lobe for all spikes in a given cluster.

        Args:
            cluster (int): cluster number

        Returns:
            str: lobe name (frontal, temporal, parietal, occipital, insula,
                or multilobar)
        """

        seqs, delays = self.fetch_sequences(cluster)
        elec_count_df = retrieve_lead_counts(self.elec_labels_df,
                                             seqs,
                                             delays
                                            )
        elec_count_dict = elec_count_df['Within 100ms'].to_dict()

        lobe_count_dict = {}
        for elec, count in elec_count_dict.items():
            lobes = convert_elec_to_lobe(self.elec2lobe_df, elec)
            for lobe in lobes:
                # ignore electrodes without a parcel assigned
                if lobe == '':
                    continue
                lobe_count_dict.setdefault(lobe,0)
                lobe_count_dict[lobe] += count

        total_counts = sum(lobe_count_dict.values())
        top_lobe = max(lobe_count_dict, key=lobe_count_dict.get)
        
        # require top lobe to include majority of all spikes
        if lobe_count_dict[top_lobe] >= (total_counts / 2):
            return top_lobe
        else:
            return "multilobar"

    def compute_weighted_source2elec_dist(self, cluster, source=None,
                                          lead_only=False, use_geo=False,
                                          use_all_seqs=False):
        """Compute the distance between a cluster and source, using a variety
        of possible weighted schema.

        Args:
            cluster (int): cluster number
            source (int, optional): source parcel (uses valid_source_one if 
                None). Defaults to None.
            lead_only (bool, optional): only use weighted distances to lead 
                electrodes. Defaults to False.
            use_geo (bool, optional): find weighted geodesic distance, not 
                Euclidean. Defaults to False.
            use_all_seqs (bool, optional): use every sequence of the cluster, 
                not just localizing sequences based on the source parcel using 
                combination method. Defaults to False.

        Returns:
            float: weighted distance metric
        """

        if source == None:
            source = list(self.valid_sources_one[cluster])[0]
        else:
            assert source in range(self.parcs)

        seqs, delays = self.fetch_sequences(cluster=cluster)

        if use_all_seqs:
            count_df = retrieve_lead_counts(self.elec_labels_df,
                                            seqs,
                                            delays)
        else:
            source_idxs = self.compute_localizing_seq_idxs(cluster=cluster, source=source)
            count_df = retrieve_lead_counts(self.elec_labels_df,
                                            seqs[source_idxs,:],
                                            delays[source_idxs,:])

        if lead_only:
            col='Leader'
        else:
            col='Within 100ms'

        count_dict = count_df[col].to_dict()

        if use_geo:
            temp_minGeo = pd.read_csv((self.dirs['sc'] / "node_minGeo_byElec.csv"),
                                    header=None)
            minGeo = temp_minGeo.to_numpy(copy=True)

        dist_sum, count_total = 0, 0

        for elec, count in count_dict.items():
            if count == 0:
                continue
            elec_idx = self.get_elec_idx(elec)
            if use_geo:
                dist = compute_elec2parc_geo(self.node2parc_df_dict,
                                             minGeo,
                                             elec_idx,
                                             source)
            else:
                dist = compute_elec2parc_euc(self.parc_minEuclidean_byElec,
                                             elec_idx,
                                             source)

            dist_sum += (dist * count)
            count_total += count

        return dist_sum / count_total
    
    def compute_farthest_elec_dists(self, cluster, seq_indices,
                                    source=None, use_geo=True):

        if source == None:
            source = list(self.valid_sources_one[cluster])[0]
        else:
            assert source in range(self.parcs)

        seqs, _ = self.fetch_sequences(cluster=cluster)
        
        if seq_indices.size == 0:
            return np.array(())
        else:
            seqs = seqs[seq_indices]

        if use_geo:
            temp_minGeo = pd.read_csv((self.dirs['sc'] / "node_minGeo_byElec.csv"),
                                    header=None)
            minGeo = temp_minGeo.to_numpy(copy=True)

        max_dists = np.zeros(seqs.shape[0])

        for i in range(seqs.shape[0]):
            row = seqs[i,:]
            elecs = [elec for elec in row if elec != "nan"]
            
            max_dist = 0
            for elec in elecs:
                elec_idx = self.get_elec_idx(elec)
                if use_geo:
                    dist = compute_elec2parc_geo(self.node2parc_df_dict,
                                                 minGeo,
                                                 elec_idx,
                                                 source)
                else:
                    dist = compute_elec2parc_euc(self.parc_minEuclidean_byElec,
                                                 elec_idx,
                                                 source)

                if dist > max_dist: max_dist = dist
            
            max_dists[i] = max_dist

        return max_dists
    
    def compute_jaro_similarities(self, cluster):
        """Compute a Jaro-Winkler similarity matrix based on an array of 
        electrode sequences. Saves out the matrix for future usage.
        
        Args:
            cluster (int): cluster number

        Returns:
            np.array: n_seq x n_seq similarity matrix
        """


        seqs, _ = self.fetch_sequences(cluster)
        n_sequences = seqs.shape[0]
        
        jaro_similarities = np.zeros((n_sequences, n_sequences))

        # retrieve similarities, ignoring 'nan' values
        for i in range(n_sequences):
            seq_i = [elec for elec in seqs[i,:] if elec != 'nan']
            for j in range(i, n_sequences):
                seq_j = [elec for elec in seqs[j,:] if elec != 'nan']
                similarity = jaro.jaro_winkler_metric(seq_i, seq_j)
                jaro_similarities[i,j] = similarity

        # symmetrize final matrix and convert to distance matrix
        jaro_similarities = np.fmax(jaro_similarities.T, jaro_similarities)
        
        if cluster != None:
            fpath = self.dirs['seqs'] / (f"cluster{cluster}_similarityMatrix_"
                                         f"max{self.seq_len}.csv")
        else:
            fpath = self.dirs['seqs'] / (f"similarityMatrix_"
                                         f"max{self.seq_len}.csv")
        
        np.savetxt(fpath, 
                   X=jaro_similarities,
                   fmt="%.3f",
                   delimiter=','
                )
        
        return jaro_similarities
        
    def retrieve_jaro_similarities(self, cluster):
        """Retrieve a jaro similarity matrix for a given cluster

        Args:
            cluster (int): cluster number (None if all sequences)

        Returns:
            np.array: Jaro-Winkler similarity matrix
        """
    
        if cluster != None:
            fpath = self.dirs['seqs'] / (f"cluster{cluster}_similarityMatrix_"
                                         f"max{self.seq_len}.csv")
        else:
            fpath = self.dirs['seqs'] / (f"similarityMatrix_"
                                         f"max{self.seq_len}.csv")
        
        if fpath.exists():
            similarities_arr = np.loadtxt(fpath,
                                          dtype=float,
                                          delimiter=",")
        else:
            similarities_arr = self.compute_jaro_similarities(cluster)
        
        return similarities_arr
    
    def compute_proportion_neighbors(self, cluster, neighbor_thresh=12.):
        """Compute the average proportion of neighbor electrodes included 
        within sequences in a cluster. Hypothesis is that a greater proportion
        correlates with being closer to epileptogenic zone.

        Args:
            cluster (int): cluster number
            neighbor_thresh (float): max euc distance to be considered neighbor

        Returns:
            float: average proportion of neighbor electrodes firing
        """
        
        seqs, _ = self.fetch_sequences(cluster)
        n_sequences = seqs.shape[0]
        
        intersection_total = 0
        neighbors_total = 0
        
        # iterate through sequences, ignoring 'nan' values
        for i in range(n_sequences):
            seq = [elec for elec in seqs[i,:] if elec != 'nan']
            elec_idxs = [self.get_elec_idx(elec) for elec in seq]
            
            # Get the list of all neighbor indices within neighbor threshold
            neighbors = set(np.argwhere(
                self.elec_euc_arr[elec_idxs,:] < neighbor_thresh
            )[:,1])
            
            # find the number of intersecting electrodes (neighbors in sequence)
            intersection = neighbors.intersection(set(elec_idxs))
            intersection_total += len(intersection)
            neighbors_total += len(neighbors)
        
        return intersection_total / neighbors_total