"""Subject class"""

from glob import glob
from io import StringIO
from math import isnan
import shlex
import subprocess
from warnings import filterwarnings

filterwarnings("ignore", category=FutureWarning)
filterwarnings(action="ignore", message="All-NaN slice encountered")

import jaro
from nilearn import surface
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from .constants import data_directories, MIN_GM_VEL, MAX_GM_VEL, MIN_WM_VEL, MAX_WM_VEL
from .helpers import *


class Subject:
    def __init__(
        self,
        subj,
        n_parcs=600,
        n_networks=17,
        max_length=10,
        dist=45,
        use_weighted=True,
        use_best=True,
        cutoff=0.0,
        in_progress=False,
    ):
        """Create an instance of Subject for analysis of IED spike data

        Args:
            subj (str): subject p-number
            n_parcs (int, optional): Schaefer parcellation number. Defaults to
                600.
            n_networks (int, optional): Schaefer networks number. Defaults to
                17.
            max_length (int, optional): Max length of sequences detected.
                Defaults to 10.
            dist (int, optional): Max geodesic search distance for
                localization. Defaults to 45.
            use_weighted (bool, optional): For narrowing down sources from all
                to one, weight all lead electrodes rather than taking the most
                frequent. Defaults to True.
            use_best (bool, optional): For narrowing down sources from all
                to one, only compare the parcels tied for highest proportion
                explained (don't consider the top 5%). Defaults to True.
            cutoff (float, optional): the minimum proportion of sequences
                explained for a cluster to be considered localized. Defaults to
                0.5.
            in_progress (bool, optional): Subject still needs to be localized; 
                do not initialize attributes that have not been created yet. 
                Defaults to False.
        """

        # set values
        self.subj = subj
        self.parcs = n_parcs
        self.networks = n_networks
        self.seq_len = max_length
        self.dist = dist

        # set subject-specific directories
        self.dirs = {
            k.lower()[:-4]: (v / self.subj) for k, v in data_directories.items()
        }
        self._update_ied_subdirs()
        self._update_mri_subdirs()

        # get arrays and dataframes that are constant
        self.elec2index_dict = self._fetch_elec2index_dict()
        self.elec2parc_dict = self._fetch_elec2parc_dict()
        self.elec2lobe_dict = self._fetch_elec2lobe_dict()
        self.elec2hemi_dict = self._fetch_elec2hemi_dict()
        self.parc2node_dict = self._fetch_parc2node_dict()
        self.node2parc_hemi_dict = self._fetch_node2parc_hemi_dict()

        self.parc_minEuclidean_byElec = self._fetch_parc_minEuclidean_byElec()
        self.elec_euc_arr = self._fetch_elec_euc_arr()

        # update attributes
        self._update_num_clusters()
        self._update_cluster_num_sequences()
        self._update_cluster_hemispheres()
        self._update_engel_class()

        # if localized, update additional attributes
        if not in_progress:
            self._update_cluster_types()
            self._update_source_parcels(
                dist=dist, use_weighted=use_weighted, use_best=use_best, cutoff=cutoff
            )
            if self.engel_class not in ("no_resection", "no_outcome", "deceased"):
                self.node2rsxn_df_dict = self._fetch_node2rsxn_df_dict()
                self.resection_thresh = self._fetch_resection_thresh()

    def _update_ied_subdirs(self):
        """Add IED subdirectories."""

        dir_name = f"Schaefer_{self.parcs}P_{self.networks}N"

        for val, dir in [
            ("seqs", "sequence_classification"),
            ("raw_fc", "raw_fc"),
            ("sc", f"sc/{dir_name}"),
            ("source_loc", f"source_localization/{dir_name}"),
        ]:

            self.dirs[val] = self.dirs["ied"] / dir

    def _update_mri_subdirs(self):
        """Add MRI subdirectories."""

        for val, dir in [
            ("surf", "surf/xhemi/std141/orig"),
            ("general", "surf/xhemi/std141/orig/general"),
            ("align_elec_alt", "icEEG/align_elec_alt"),
            ("docs", "icEEG/__docs"),
        ]:

            self.dirs[val] = self.dirs["mri"] / dir

    def _fetch_elec_euc_arr(self):
        """Fetch elec_euc_arr to store as self.elec_euc_arr.

        Returns:
            np.array: n_elec x n_elec array of Euclidean distances
        """

        fpath = self.dirs["sc"] / "elec_euc_withinHemi.csv"
        arr = np.loadtxt(fpath, dtype=float)
        arr[np.diag_indices_from(arr)] = np.NaN

        return arr

    def _remove_noparcel_elecs(self, cluster):
        """Save sequences without elecs that have no parcels because they were
        not used for source localization and should be ignored for analysis.

        Args:
            cluster (int): cluster number
        """

        suffix = ""
        if isinstance(cluster, int):
            suffix = f"_cluster{cluster}"

        seqs_file = self.dirs["seqs"] / (
            f"elecSequences_max{self.seq_len}" f"{suffix}.csv"
        )
        seqs = np.loadtxt(seqs_file, dtype=str, delimiter=",")

        delays_file = self.dirs["seqs"] / (
            f"delaySequences_max{self.seq_len}" f"{suffix}.csv"
        )
        delays = np.loadtxt(delays_file, dtype=float, delimiter=",")

        new_seqs = []
        new_delays = []

        for i in range(seqs.shape[0]):

            new_seqs.append([])  # make new list in new_seqs and new_delays
            new_delays.append([])

            row = seqs[i, :]
            elecs = [elec for elec in row if elec != "nan"]
            for j, elec in enumerate(elecs):
                parcs = self.elec2parc_dict[elec]
                if len(parcs) > 0:
                    new_seqs[-1].append(elec)
                    new_delays[-1].append(delays[i, j])

            new_delays[-1] = [lag - new_delays[-1][0] for lag in new_delays[-1]]

        # convert new_seqs and new_delays to numpy arrays
        new_seqs_arr = output_lst_of_lsts(new_seqs, my_dtype=object)
        new_delays_arr = output_lst_of_lsts(new_delays)

        out_fpath = self.dirs["seqs"] / (
            "elecSequences_withParc_max" f"{self.seq_len}{suffix}.csv"
        )
        np.savetxt(out_fpath, X=new_seqs_arr, delimiter=",", fmt="%s")

        out_fpath = self.dirs["seqs"] / (
            "delaySequences_withParc_max" f"{self.seq_len}{suffix}.csv"
        )
        np.savetxt(out_fpath, X=new_delays_arr, delimiter=",")

    def _fetch_resection_thresh(self):
        """Fetch the proportion of a parcel resected threshold. 0.5 unless the
        subject has no parcels half-resected, in which case this will return
        the maximum.

        Returns:
            float: 0.5 or max(all_resected_proportions); whichever is smallest
        """

        resection_props = np.array(self.compute_resected_prop(range(1, self.parcs + 1)))

        # account for parcels with no parcels > 0.5 resected
        return min(max(resection_props), 0.5)

    def _fetch_parc_minEuclidean_byElec(self):
        """Fetch array of minimum Euclidean distances between parcels and
        electrodes.

        Returns:
            np.array: array Euclidean distances with shape (n_parcs, n_elecs)
        """

        fpath = self.dirs["sc"] / "parc_minEuclidean_byElec.csv"
        return np.loadtxt(fpath, delimiter=",", dtype=float)

    def _fetch_parc2node_dict(self):
        """Fetch parcel to node dictionary.

        Returns:
            dict: keys = parcel number, values = array of nodes
        """

        parc2node_dict = {}

        for hemi, use_idx in [("LH", 1), ("RH", 2)]:

            fpath = self.dirs["sc"] / "node_to_parc.csv"
            df = pd.read_csv(
                fpath,
                usecols=[0, use_idx],
                names=["node", "parcel"],
                skiprows=1,
                dtype=int,
            )

            modifier = 0
            if hemi == "RH":
                modifier = self.parcs // 2

            for parc in range(1, (self.parcs // 2) + 1):
                node_arr = df.loc[df.parcel == parc, "node"].to_numpy()
                parc2node_dict[parc + modifier] = node_arr

        return parc2node_dict

    def _fetch_node2parc_hemi_dict(self):
        """Fetch node to parcel hemi dict.

        Returns:
            dict: keys = hemi, values = node2parc_dict;
                  node2parc_dict: keys = node, values = parcel
        """

        node2parc_hemi_dict = {}

        for hemi, use_idx in [("LH", 1), ("RH", 2)]:

            fpath = self.dirs["sc"] / "node_to_parc.csv"
            df = pd.read_csv(
                fpath,
                usecols=[0, use_idx],
                names=["node", "parcel"],
                skiprows=1,
                dtype=int,
            )

            modifier = 0
            if hemi == "RH":
                modifier = self.parcs // 2

            df_dict = df.set_index("node").to_dict()["parcel"]
            node2parc_dict = {
                node: (parc + modifier if parc != 0 else parc)
                for node, parc in df_dict.items()
            }

            node2parc_hemi_dict[hemi] = node2parc_dict

        return node2parc_hemi_dict

    def _fetch_elec2hemi_dict(self):
        """Fetch elec to hemi dictionary.

        Returns:
            dict: keys = elec, values = hemi
        """

        fpath = self.dirs["sc"] / "elec_to_hemi.csv"
        df = pd.read_csv(fpath)
        df.hemi = df.hemi.str.upper()
        return df.set_index("elec").to_dict()["hemi"]

    def _fetch_elec2parc_dict(self):
        """Fetch elec to parcel dictionary.

        Returns:
            dict: keys = elec, values = hemi
        """
        df = pd.read_csv((self.dirs["sc"] / "elec_to_parc.csv"))
        df_dict = df.set_index("elecLabel").to_dict()["parcNumber"]
        return {elec: convert_parcs(parcs) for elec, parcs in df_dict.items()}

    def _fetch_elec2lobe_dict(self):
        """Fetch elec to lobe dictionary.

        Returns:
            dict: keys = elec, values = lobe
        """

        fpath = self.dirs["sc"] / "elec_to_lobe.csv"
        df = pd.read_csv(fpath)
        df_dict = df.set_index("elecLabel").to_dict()["Lobe"]
        return {elec: convert_lobes(lobes) for elec, lobes in df_dict.items()}

    def _fetch_elec2index_dict(self):
        """Fetch elec to index dictionary.

        Returns:
            dict: keys = elec, values = index
        """

        # load order of electrodes
        fpath = self.dirs["sc"] / "elec_labels.csv"
        df = pd.read_csv(fpath, header=None, names=["chanName"])
        df["index"] = df.index
        return df.set_index("chanName").to_dict()["index"]

    def _fetch_node2rsxn_df_dict(self):
        """Create dictionary of node to resection lookup tables for each
        hemisphere.

        Returns:
            dict: dictionary with keys = hemi and values = dataframe
        """

        # create dictionary of node2rsxn_df for each hemisphere of interest
        node2rsxn_df_dict = {}

        for hemi in "LH", "RH":
            node2rsxn_path = self.dirs["sc"] / f"{hemi.upper()}_node_to_resection.txt"
            node2rsxn_df = pd.read_csv(
                node2rsxn_path,
                delim_whitespace=True,
                header=None,
                names=["node", "is_resected"],
            )
            node2rsxn_df_dict[hemi] = node2rsxn_df

        return node2rsxn_df_dict

    def _update_num_clusters(self):
        """Update self.num_clusters value."""

        # load cluster summary df
        in_file = f"cluster_summary_max{self.seq_len}.csv"
        try:
            df = pd.read_csv((self.dirs["seqs"] / in_file))
            self.num_clusters = max(set(df["Cluster Number"]))
        except FileNotFoundError:
            self.num_clusters = "Clustering has not been run"

    def _update_cluster_types(self):
        """Update self.valid_clusters and self.soz_clusters."""

        # check that clustering is complete
        assert isinstance(self.num_clusters, int)

        # load bad_clusters
        in_file = "ied_bad_clusters.csv"
        bad_df = pd.read_csv((data_directories["IED_ANALYSIS_DIR"] / in_file))

        # update self.valid_clusters
        try:
            subj_bad_clusters = bad_df.loc[bad_df.subject == self.subj][
                "badClusters"
            ].iloc[0]
            bad_clusters = subj_bad_clusters.strip("][").split(",")
            if bad_clusters[0] == "":
                bad_clusters = []
            else:
                bad_clusters = [int(clust) for clust in bad_clusters]
            valid_clusters = [
                i for i in range(1, self.num_clusters + 1) if i not in bad_clusters
            ]
            self.valid_clusters = [
                i for i in valid_clusters if self.cluster_nseqs[i] >= 100
            ]
        except IndexError:
            self.valid_clusters = "Subject not found in ied_bad_clusters.csv"

        # load soz_clusters
        in_file = "ied_soz_clusters.csv"
        soz_df = pd.read_csv((data_directories["IED_ANALYSIS_DIR"] / in_file))

        # update self.soz_clusters
        try:
            subj_soz_clusters = soz_df.loc[soz_df.subj == self.subj][
                "sozClusters"
            ].iloc[0]
            soz_clusters = subj_soz_clusters.strip("][").split(",")
            if soz_clusters[0] == "":
                self.soz_clusters = []
            else:
                self.soz_clusters = [
                    int(clust)
                    for clust in soz_clusters
                    if self.cluster_nseqs[int(clust)] >= 100
                ]
        except IndexError:
            self.soz_clusters = "Subject not found in ied_soz_clusters.csv"

    def _update_cluster_num_sequences(self):
        """Update self.cluster_nseqs value."""

        fpath = self.dirs["seqs"] / f"cluster_summary_max{self.seq_len}.csv"
        df = pd.read_csv(fpath)

        cluster_seq_dict = {}
        for cluster in range(1, self.num_clusters + 1):
            n_seqs = df.loc[
                df["Cluster Number"] == cluster, "Number of Sequences"
            ].iloc[0]
            cluster_seq_dict[cluster] = n_seqs

        self.cluster_nseqs = cluster_seq_dict

    def _update_cluster_hemispheres(self):
        """Update self.cluster_hemis value"""

        fpath = self.dirs["seqs"] / f"cluster_summary_max{self.seq_len}.csv"
        df = pd.read_csv(fpath)

        cluster_hemi_dict = {}
        for cluster in range(1, self.num_clusters + 1):
            hemi = df.loc[df["Cluster Number"] == cluster, "Hemi"].iloc[0]
            cluster_hemi_dict[cluster] = hemi

        self.cluster_hemis = cluster_hemi_dict

    def _update_source_parcels(
        self, dist=45, use_weighted=True, use_best=True, cutoff=0.5
    ):
        """Update self.valid_sources_all with a set of every possible source
        for each cluster. Update self.valid_sources_one with a single source
        for each cluster (choosing the one that is closest to the most frequent
        lead electrode or has highest proportion explained).

        Args:
            dist (int, optional): geodesic search distance. Defaults to 45.
            use_weighted (bool, optional): For narrowing down sources from all
                to one, weight all lead electrodes rather than taking the most
                frequent. Defaults to True.
            use_best (bool, optional): For narrowing down sources from all
                to one, only compare the parcels tied for highest proportion
                explained (don't consider the top 5%). Defaults to True.
            cutoff (float, optional): the minimum proportion of sequences
                explained for a cluster to be considered localized. Defaults to
                0.5.
        """

        valid_sources_all, valid_sources_one = {}, {}

        for cluster in self.valid_clusters:

            parc2prop_df = self.fetch_normalized_parc2prop_df(cluster, dist=dist)

            # get all top source parcels (within 5% of the top source)
            top_parc = parc2prop_df.sort_values(
                by=["propExplanatorySpikes"], ascending=False
            ).head(1)
            top_proportion = top_parc["propExplanatorySpikes"].iloc[0]

            if top_proportion > cutoff:
                valid_sources_all[cluster] = set(
                    parc2prop_df.loc[
                        parc2prop_df["propExplanatorySpikes"] > (top_proportion - 0.05)
                    ].index
                )
            else:
                valid_sources_all[cluster] = set()

            # as a tiebreaker, choose source parcel closest to most frequent
            # leading electrode
            if len(valid_sources_all[cluster]) > 1:

                if use_best:
                    # only compare sources with max proportion explained
                    sources_to_compare = set(
                        parc2prop_df.loc[
                            parc2prop_df.propExplanatorySpikes
                            == parc2prop_df.max().iloc[0]
                        ].index
                    )
                else:
                    # compare sources in top 5%
                    sources_to_compare = valid_sources_all[cluster]

                if use_weighted:
                    min_dist = np.Inf
                    for source in sources_to_compare:
                        d = self.compute_weighted_source2elec_dist(
                            cluster, source=source, lead_only=True, use_all_seqs=True
                        )
                        if min_dist > d:
                            min_dist = d
                            best_source = source

                    valid_sources_one[cluster] = set([best_source])
                else:
                    # compute most frequent lead electrode
                    seqs, _ = self.fetch_sequences(cluster)
                    lead = compute_top_lead_elec(seqs)

                    # get lead index
                    lead_idx = self.elec2index_dict[lead]

                    # find closest parcel to lead_idx
                    top_parc_idxs = [parc - 1 for parc in sources_to_compare]
                    min_loc = np.argmin(
                        self.parc_minEuclidean_byElec[top_parc_idxs, lead_idx]
                    )
                    valid_sources_one[cluster] = set([top_parc_idxs[min_loc] + 1])

            else:
                valid_sources_one[cluster] = valid_sources_all[cluster]

        self.valid_sources_all = valid_sources_all
        self.valid_sources_one = valid_sources_one

    def _update_engel_class(self):
        """Update self.engel_class and self.engel_months."""

        # load hemi df
        engel_fpath = data_directories["IED_ANALYSIS_DIR"] / "ied_subj_engelscores.csv"
        engel_df = pd.read_csv(engel_fpath)

        engel_dict = {
            "MoreThan24 Engel Class": "MoreThan24 Months",
            "Mo24 Engel Class": 24,
            "Mo12 Engel Class": 12,
        }

        # iterate through engel_df columns from longest time to shortest
        for col, time in engel_dict.items():
            # get class from df
            engel_class = engel_df[engel_df["Patient"] == self.subj][col].iloc[0]

            # set months
            if type(time) is str:
                engel_months = engel_df[engel_df["Patient"] == self.subj][time].iloc[0]
            elif engel_class in ["no_resection", "no_outcome", "deceased"]:
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

        # since none of the class columns were filled-in, set months to np.NaN
        engel_months = np.NaN

        self.engel_class = engel_class
        self.engel_months = engel_months

    def _compute_jaro_similarities(self, cluster):
        """Compute a Jaro-Winkler similarity matrix based on an array of
        electrode sequences. Saves out the matrix for future usage.

        Args:
            cluster (int): cluster number

        Returns:
            np.array: n_seq x n_seq similarity matrix
        """

        seqs, _ = self.fetch_sequences(cluster, all_elecs=True)
        n_sequences = seqs.shape[0]

        jaro_similarities = np.zeros((n_sequences, n_sequences))

        # retrieve similarities, ignoring 'nan' values
        for i in range(n_sequences):
            seq_i = [elec for elec in seqs[i, :] if elec != "nan"]
            for j in range(i, n_sequences):
                seq_j = [elec for elec in seqs[j, :] if elec != "nan"]
                similarity = jaro.jaro_winkler_metric(seq_i, seq_j)
                jaro_similarities[i, j] = similarity

        # symmetrize final matrix and convert to distance matrix
        jaro_similarities = np.fmax(jaro_similarities.T, jaro_similarities)

        if cluster != None:
            fpath = self.dirs["seqs"] / (
                f"cluster{cluster}_similarityMatrix_" f"max{self.seq_len}.csv"
            )
        else:
            fpath = self.dirs["seqs"] / (f"similarityMatrix_" f"max{self.seq_len}.csv")

        np.savetxt(fpath, X=jaro_similarities, fmt="%.3f", delimiter=",")

        return jaro_similarities

    def _compute_parc_dist_matrix(self):
        """Compute n_parcs x n_parcs matrix of Euclidean distances, when the
        file has not been saved previously.

        Returns:
            np.array: n_parcs x n_parcs distance array
        """

        fpath = (
            self.dirs["dti"]
            / "roi"
            / (
                f"indt_std.141.both.Schaefer2018_"
                f"{self.parcs}Parcels_"
                f"{self.networks}Networks_FINAL.ni"
                "i.gz"
            )
        )

        # run AFNI 3dCM command to find center of mass of parcels
        afni_cmd = shlex.split(f"3dCM -all_rois {fpath}")
        process = subprocess.run(
            afni_cmd, stdout=subprocess.PIPE, universal_newlines=True
        )

        # skip first row (zero parcel)
        parc_centers = np.genfromtxt(StringIO(process.stdout))[1:]

        parc_dists = distance_matrix(parc_centers, parc_centers)

        opath = self.dirs["sc"] / "parc_euc_dists.csv"
        np.savetxt(opath, X=parc_dists, fmt="%.3f", delimiter=",")

        return parc_dists

    def fetch_sequences(self, cluster=None, all_elecs=False):
        """Fetch electrode sequences and lag times once they have been saved
        as .csv files in self.dirs['seqs'].

        Args:
            cluster (int, optional): cluster of interest. Defaults to None.
            all_elecs (bool, optional): include elecs without parcels. Defaults
                to False

        Returns:
            seqs, delays: np.array of electrode names and np.array of delay
                times for every sequence
        """

        assert isinstance(cluster, (int, type(None)))

        suffix = ""
        if isinstance(cluster, int):
            suffix = f"_cluster{cluster}"

        noparc_str = "_withParc"
        if all_elecs:
            noparc_str = ""

        seqs_file = self.dirs["seqs"] / (
            f"elecSequences{noparc_str}_max" f"{self.seq_len}{suffix}.csv"
        )

        if not seqs_file.exists():
            # remove all electrodes without parcels from sequences
            self._remove_noparcel_elecs(cluster)

        seqs = np.loadtxt(seqs_file, dtype=str, delimiter=",")

        delays_file = self.dirs["seqs"] / (
            f"delaySequences{noparc_str}_max" f"{self.seq_len}{suffix}.csv"
        )
        delays = np.loadtxt(delays_file, dtype=float, delimiter=",")

        return seqs, delays

    def fetch_normalized_parc2prop_df(
        self, cluster, dist=45, only_gm=False, only_wm=False
    ):
        """Fetch df with conversion table of parcel number to proportion of
        sequences explained.

        Args:
            cluster (int): number of cluster
            dist (int, optional): Geodesic search distance in mm. Defaults to
                45.
            only_gm (bool, optional): Use gm only method. Defaults to
                False.
            only_wm (bool, optional): Use white matter only method. Defaults to
                False.

        Returns:
            pd.DataFrame: dataframe with index as parcel number and column as
                'propExplanatorySpikes'
        """

        # if not using combination method, only one input can be set to True
        assert not (only_gm and only_wm)

        method = ""
        if only_gm:
            method = "_geodesic"
        elif only_wm:
            method = "_whiteMatter"

        fname = (
            f"*{method}_normalizedCounts_within{dist}_max{self.seq_len}"
            f"_cluster{cluster}.csv"
        )
        fpath_lst = glob(str(self.dirs["source_loc"] / fname))

        if not (only_gm or only_wm):
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

    def fetch_geodesic_travel_times(self):
        """Fetch geodesic travel times based on geodesic velocities and
        distances from electrodes to nodes on the std.141 mesh.

        Returns:
            tuple: minGeo_maxSpeed_time, minGeo_minSpeed_time,
                maxGeo_minSpeed_time, maxGeo_maxSpeed_time (four np.arrays
                with shape (n_nodes, n_elecs))
        """

        # load and compute estimated lag times based on Geodesic distances

        fdir = self.dirs["sc"]
        temp_minGeo = pd.read_csv((fdir / "node_minGeo_byElec.csv"), header=None)
        temp_maxGeo = pd.read_csv((fdir / "node_maxGeo_byElec.csv"), header=None)

        minGeo = temp_minGeo.to_numpy(copy=True)
        maxGeo = temp_maxGeo.to_numpy(copy=True)

        # set all minimum values > dist to np.NaN and all max values to dist that
        # have a min value less than dist
        minGeo[minGeo > self.dist] = np.NaN
        maxGeo[np.isnan(minGeo)] = np.NaN
        maxGeo[maxGeo > self.dist] = self.dist

        minGeo_maxSpeed_time = minGeo / MAX_GM_VEL
        minGeo_minSpeed_time = minGeo / MIN_GM_VEL
        maxGeo_minSpeed_time = maxGeo / MIN_GM_VEL
        maxGeo_maxSpeed_time = maxGeo / MAX_GM_VEL

        return (
            minGeo_maxSpeed_time,
            minGeo_minSpeed_time,
            maxGeo_minSpeed_time,
            maxGeo_maxSpeed_time,
        )

    def fetch_wm_travel_times(self):
        """Fetch white matter min/max BL times.

        Returns:
            tuple: minBL_time, maxBL_time (np.arrays of shape
                (n_parcs, n_parcs))
        """

        # load and compute estimated lag times based on WM bundle lengths
        fdir = self.dirs["sc"]
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
        lead_elecs = elec_seqs[:, 0]

        # use normalized parc2prop file as template for lead_elec_df
        lead_elec_df = self.fetch_normalized_parc2prop_df(cluster)
        lead_elec_df["propExplanatorySpikes"] = 0

        parc_counts = {}
        for elec in lead_elecs:
            # get electrode parcel indices
            parc_idxs = [parc - 1 for parc in self.elec2parc_dict[elec]]

            # update count for each parcel
            for parc_idx in parc_idxs:
                parc_counts.setdefault(parc_idx, 0)
                parc_counts[parc_idx] += 1

        # normalize based on total number of sequences
        n_seqs = elec_seqs.shape[0]
        normalized_counts = {
            parc: (count / n_seqs) for parc, count in parc_counts.items()
        }

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

            # get hemi_specific df
            hemi_node2rsxn = self.node2rsxn_df_dict[hemi]

            # mask out nodes for parcel and find proportion of nodes resected
            parc_nodes = self.parc2node_dict[parcel]
            resected_prop = hemi_node2rsxn.iloc[parc_nodes].mean()["is_resected"]

            # add to list
            resected_props.append(resected_prop)

        return resected_props

    def compute_rsxn_source_arr(
        self,
        n_cluster,
        source=None,
        all_sources=False,
        rsxn_only=False,
        source_only=False,
    ):
        """Create n_node array with a gray resection zone and green TN/TP
        regions or red FN/FP parcels.

        Args:
            n_cluster (int): cluster number
            source (int): source parcel in range(1, n_parcs+1). Defaults to
                None which selects the combination source.
            all_sources (bool, optional): plot every possible source
                (not just the single best source). Defaults to False.
            rsxn_only (bool, optional): plot only the resection zone. Defaults
                to False.
            source_only (bool, optional): plot only the source. Defaults to 
                False.

        Returns:
            tuple: np.array of values for creating niml.dset, str hemisphere
        """

        if source is None:
            if all_sources:
                sources = self.valid_sources_all[n_cluster]
            else:
                sources = self.valid_sources_one[n_cluster]
        else:
            assert source in range(1, self.parcs + 1)
            sources = [source]

        hemi = get_parcel_hemi(list(sources)[0], self.parcs)

        # retrieve array of 0s (non-resected) and 1s (resected)
        rsxn_arr = (
            self.node2rsxn_df_dict[hemi.upper()]["is_resected"]
            .to_numpy(dtype=float)
            .copy()
        )

        if rsxn_only:
            return rsxn_arr, hemi
        elif source_only:
            rsxn_arr[:] = 0

        # get resection props
        rsxn_props = self.compute_resected_prop(sources)

        for source, rsxn_prop in zip(sources, rsxn_props):

            mask = self.parc2node_dict[source]

            accuracy = get_prediction_accuracy(self.engel_class, rsxn_prop)

            # set values depending on concordance
            if accuracy in ["TP", "FP"]:
                rsxn_arr[mask] = 2.6
            elif accuracy in ["TN", "FN"]:
                rsxn_arr[mask] = 4

        return rsxn_arr, hemi

    def compute_localizing_seq_idxs(
        self, cluster, source, only_gm=False, only_wm=False
    ):
        """Return an array of indices for which a source successfully localizes
        the sequences of a given cluster.

        Args:
            cluster (int): cluster number
            source (int): source parcel
            only_gm (bool, optional): Use gm only localization method.
                Defaults to False.
            only_wm (bool, optional): Use white matter only localization
                method. Defaults to False.

        Returns:
            np.array: array of indices for which the sequences localize to a
                given source (find the sequences with seqs[seq_indices])
        """

        method = ""
        if only_gm:
            method = "_geodesic"
        elif only_wm:
            method = "_whiteMatter"

        fname = (
            f"*{method}_sourceParcels_within{self.dist}_max{self.seq_len}"
            f"_cluster{cluster}.csv"
        )
        fpath_lst = glob(str(self.dirs["source_loc"] / fname))

        if not (only_gm or only_wm):
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
        return np.squeeze(np.argwhere(np.sum(seq_sources == source, axis=1)), axis=1)

    def get_cluster_lobe(self, cluster):
        """Return the majority lobe for all spikes in a given cluster.

        Args:
            cluster (int): cluster number

        Returns:
            str: lobe name (frontal, temporal, parietal, occipital, insula,
                or multilobar)
        """

        seqs, delays = self.fetch_sequences(cluster)
        elec_count_df = retrieve_lead_counts(self.elec2index_dict.keys(), seqs, delays)
        elec_count_dict = elec_count_df["Within 100ms"].to_dict()

        lobe_count_dict = {}
        for elec, count in elec_count_dict.items():
            lobes = self.elec2lobe_dict[elec]
            for lobe in lobes:
                # ignore electrodes without a parcel assigned
                if lobe == "":
                    continue
                lobe_count_dict.setdefault(lobe, 0)
                lobe_count_dict[lobe] += count

        total_counts = sum(lobe_count_dict.values())
        top_lobe = max(lobe_count_dict, key=lobe_count_dict.get)

        # require top lobe to include majority of all spikes
        if lobe_count_dict[top_lobe] >= (total_counts / 2):
            return top_lobe
        else:
            return "multilobar"

    def compute_weighted_source2elec_dist(
        self, cluster, source=None, lead_only=False, use_geo=False, use_all_seqs=False
    ):
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
            assert source in range(1, self.parcs + 1)

        seqs, delays = self.fetch_sequences(cluster=cluster)

        if use_all_seqs:
            count_df = retrieve_lead_counts(self.elec2index_dict.keys(), seqs, delays)
        else:
            source_idxs = self.compute_localizing_seq_idxs(
                cluster=cluster, source=source
            )
            count_df = retrieve_lead_counts(
                self.elec2index_dict.keys(),
                seqs[source_idxs, :],
                delays[source_idxs, :],
            )

        if lead_only:
            col = "Leader"
        else:
            col = "Within 100ms"

        count_dict = count_df[col].to_dict()

        if use_geo:
            temp_minGeo = pd.read_csv(
                (self.dirs["sc"] / "node_minGeo_byElec.csv"), header=None
            )
            minGeo = temp_minGeo.to_numpy(copy=True)

        dist_sum, count_total = 0, 0

        for elec, count in count_dict.items():
            if count == 0:
                continue
            elec_idx = self.elec2index_dict[elec]
            if use_geo:
                dist = compute_elec2parc_geo(
                    self.parc2node_dict, minGeo, elec_idx, source
                )
            else:
                dist = compute_elec2parc_euc(
                    self.parc_minEuclidean_byElec, elec_idx, source
                )

            dist_sum += dist * count
            count_total += count

        return dist_sum / count_total

    def compute_farthest_elec_dists(
        self, cluster, seq_indices=None, source=None, use_geo=True
    ):
        """Return an array of distances to the farthest electrode in each
        sequence. Hypothesis is that sequences requiring white matter will have
        a higher proportion of distant electrodes.

        Args:
            cluster (int): cluster number
            seq_indices (np.array): array of sequence indices. Defaults to None.
            source (int, optional): source parcel number. Defaults to None.
            use_geo (bool, optional): Use geodesic distance, otherwise
                Euclidean is used. Defaults to True.

        Returns:
            np.array: array of maximum distances from parcel to farthest
                electrode
        """

        if source is None:
            source = list(self.valid_sources_one[cluster])[0]
        else:
            assert source in range(1, self.parcs + 1)

        seqs, _ = self.fetch_sequences(cluster=cluster)

        if seq_indices is None:
            pass
        elif seq_indices.size == 0:
            return np.array(())
        else:
            seqs = seqs[seq_indices]

        if use_geo:
            temp_minGeo = pd.read_csv(
                (self.dirs["sc"] / "node_minGeo_byElec.csv"), header=None
            )
            minGeo = temp_minGeo.to_numpy(copy=True)

        max_dists = np.zeros(seqs.shape[0])

        for i in range(seqs.shape[0]):
            row = seqs[i, :]
            elecs = [elec for elec in row if elec != "nan"]

            max_dist = 0
            for elec in elecs:
                elec_idx = self.elec2index_dict[elec]
                if use_geo:
                    dist = compute_elec2parc_geo(
                        self.parc2node_dict, minGeo, elec_idx, source
                    )
                else:
                    dist = compute_elec2parc_euc(
                        self.parc_minEuclidean_byElec, elec_idx, source
                    )

                if dist > max_dist:
                    max_dist = dist

            max_dists[i] = max_dist

        return max_dists

    def retrieve_jaro_similarities(self, cluster):
        """Retrieve a Jaro similarity matrix for a given cluster

        Args:
            cluster (int): cluster number (None if all sequences)

        Returns:
            np.array: Jaro-Winkler similarity matrix
        """

        if cluster != None:
            fpath = self.dirs["seqs"] / (
                f"cluster{cluster}_similarityMatrix_" f"max{self.seq_len}.csv"
            )
        else:
            fpath = self.dirs["seqs"] / (f"similarityMatrix_" f"max{self.seq_len}.csv")

        if fpath.exists():
            similarities_arr = np.loadtxt(fpath, dtype=float, delimiter=",")
        else:
            similarities_arr = self._compute_jaro_similarities(cluster)

        return similarities_arr

    def compute_proportion_neighbors(self, cluster, neighbor_thresh=12.0):
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
            seq = [elec for elec in seqs[i, :] if elec != "nan"]
            elec_idxs = [self.elec2index_dict[elec] for elec in seq]

            # Get the list of all neighbor indices within neighbor threshold
            neighbors = set(
                np.argwhere(self.elec_euc_arr[elec_idxs, :] < neighbor_thresh)[:, 1]
            )

            # find the number of intersecting electrodes (neighbors in sequence)
            intersection = neighbors.intersection(set(elec_idxs))
            intersection_total += len(intersection)
            neighbors_total += len(neighbors)

        return intersection_total / neighbors_total

    def compute_closest_elec_metrics(
        self, cluster, seq_indices=None, source=None, use_geo=True
    ):
        """Return arrays filled with the position, distance, and lag time of
        the closest electrode to the source for every sequence. Hypothesis is
        that sequences requiring white matter will allow for electrodes firing
        later to be closer.

        Args:
            cluster (int): cluster number
            seq_indices (np.array): array of sequence indices. Defaults to
                None.
            source (int, optional): source parcel number. Defaults to None.
            use_geo (bool, optional): Use geodesic distance, otherwise
                Euclidean is used. Defaults to True.

        Returns:
            tuple: np.array of positions of closest electrodes,
                   np.array of distances to closest electrodes,
                   np.array of lag times of closest electrodes
        """

        if source == None:
            source = list(self.valid_sources_one[cluster])[0]
        else:
            assert source in range(1, self.parcs + 1)

        seqs, delays = self.fetch_sequences(cluster=cluster)

        if seq_indices is None:
            pass  # use all sequences
        elif seq_indices.size == 0:
            return [np.array(())] * 3  # no sequences
        else:
            seqs = seqs[seq_indices]
            delays = delays[seq_indices]

        if use_geo:
            temp_minGeo = pd.read_csv(
                (self.dirs["sc"] / "node_minGeo_byElec.csv"), header=None
            )
            minGeo = temp_minGeo.to_numpy(copy=True)

        closest_elec_positions = np.zeros(seqs.shape[0])
        closest_elec_dists = np.zeros(seqs.shape[0])
        closest_elec_delays = np.zeros(seqs.shape[0])

        for i in range(seqs.shape[0]):
            row = seqs[i, :]
            elecs = [elec for elec in row if elec != "nan"]

            min_dist = np.Inf
            closest_position = -1
            for j in range(len(elecs)):
                elec_idx = self.elec2index_dict[elecs[j]]
                if use_geo:
                    dist = compute_elec2parc_geo(
                        self.parc2node_dict, minGeo, elec_idx, source
                    )
                else:
                    dist = compute_elec2parc_euc(
                        self.parc_minEuclidean_byElec, elec_idx, source
                    )

                if dist < min_dist:
                    min_dist = dist
                    closest_position = j + 1  # add 1 so that the 1st elec is 1

            closest_elec_positions[i] = closest_position
            closest_elec_dists[i] = min_dist
            closest_elec_delays[i] = delays[i, closest_position - 1]

        return closest_elec_positions, closest_elec_dists, closest_elec_delays

    def compute_all_elec_dists(
        self, cluster, seq_indices=None, source=None, use_geo=True
    ):
        """Return an array of distances to all electrodes in each sequence.

        Args:
            cluster (int): cluster number
            seq_indices (np.array): array of sequence indices. Defaults to all
                sequences.
            source (int, optional): source parcel number. Defaults to cluster
                valid source one.
            use_geo (bool, optional): Use geodesic distance, otherwise
                Euclidean is used. Defaults to True.

        Returns:
            np.array: array of maximum distances from parcel to farthest
                electrode
        """

        if source is None:
            source = list(self.valid_sources_one[cluster])[0]
        else:
            assert source in range(1, self.parcs + 1)

        seqs, _ = self.fetch_sequences(cluster=cluster)

        if seq_indices is None:
            pass
        elif seq_indices.size == 0:
            return np.array(())
        else:
            seqs = seqs[seq_indices]

        if use_geo:
            temp_minGeo = pd.read_csv(
                (self.dirs["sc"] / "node_minGeo_byElec.csv"), header=None
            )
            minGeo = temp_minGeo.to_numpy(copy=True)

        all_dists = np.array(())

        for i in range(seqs.shape[0]):
            row = seqs[i, :]
            elecs = [elec for elec in row if elec != "nan"]
            seq_dists = np.zeros(len(elecs))

            for j, elec in enumerate(elecs):
                elec_idx = self.elec2index_dict[elec]
                if use_geo:
                    dist = compute_elec2parc_geo(
                        self.parc2node_dict, minGeo, elec_idx, source
                    )
                else:
                    dist = compute_elec2parc_euc(
                        self.parc_minEuclidean_byElec, elec_idx, source
                    )

                seq_dists[j] = dist

            all_dists = np.hstack((all_dists, seq_dists))

        return all_dists

    def fetch_minmax_distances(self):
        """Fetch min/max white matter and geodesic distances.

        Returns:
            tuple: minBL, maxBL, minGeo, maxGeo (np.arrays)
        """

        # load min/max WM bundle lengths

        BL = np.loadtxt((self.dirs["sc"] / "BL.csv"), delimiter=",", dtype=float)
        sBL = np.loadtxt((self.dirs["sc"] / "sBL.csv"), delimiter=",", dtype=float)

        BL[BL == 0] = np.NaN
        sBL[sBL == 0] = np.NaN

        maxBL = BL + sBL
        minBL = BL - sBL
        minBL[minBL < 0] = 0
        minBL[np.diag_indices_from(minBL)] = 0
        maxBL[np.diag_indices_from(maxBL)] = 10

        # load min/max geodesic distances
        temp_minGeo = pd.read_csv(
            (self.dirs["sc"] / "node_minGeo_byElec.csv"), header=None
        )
        temp_maxGeo = pd.read_csv(
            (self.dirs["sc"] / "node_maxGeo_byElec.csv"), header=None
        )
        minGeo = temp_minGeo.to_numpy(copy=True)
        maxGeo = temp_maxGeo.to_numpy(copy=True)

        # set all minimum values > dist to np.NaN and all max values to dist
        # that have a min value less than dist
        minGeo[minGeo > self.dist] = np.NaN
        maxGeo[np.isnan(minGeo)] = np.NaN
        maxGeo[maxGeo > self.dist] = self.dist

        return minBL, maxBL, minGeo, maxGeo

    def fetch_pial_surface_dict(self):
        """Fetch hemi dictionary of pial surface mesh objects.

        Returns:
            dict: keys = ["LH","RH"],
                  values = nilearn.surface.surface.mesh objects
        """

        surf_dict = {}
        for hemi in "lh", "rh":
            surf_path = self.dirs["surf"] / f"std.141.{hemi}.pial.gii"
            surf_dict[hemi.upper()] = surface.load_surf_mesh(str(surf_path))

        return surf_dict

    def retrieve_parc_dist_matrix(self):
        """Retrieve n_parcs x n_parcs matrix of Euclidean distances between
        ROI center of masses.

        Returns:
            np.array: n_parcs x n_parcs distance array
        """

        fpath = self.dirs["sc"] / "parc_euc_dists.csv"

        if fpath.exists():
            parc_dists = np.loadtxt(fpath, dtype=float, delimiter=",")
        else:
            parc_dists = self._compute_parc_dist_matrix()

        return parc_dists

    def compute_farthest_intrasequence_euc(self, cluster, seq_indices=None):
        """Return an array of distances to all electrodes in each sequence.

        Args:
            cluster (int): cluster number
            seq_indices (np.array): array of sequence indices. Defaults to all
                sequences.

        Returns:
            np.array: array of maximum distance between a pair of electrodes 
                for each sequence
        """

        seqs, _ = self.fetch_sequences(cluster=cluster)

        if seq_indices is None:
            pass
        elif seq_indices.size == 0:
            return np.array(())
        else:
            seqs = seqs[seq_indices]

        all_dists = np.full(seqs.shape[0], np.NaN)

        for i in range(seqs.shape[0]):
            row = seqs[i, :]
            elec_idxs = [self.elec2index_dict[elec] for elec in row if elec != "nan"]

            idx_combos = combinations(elec_idxs, r=2)

            max_euc_dist = 0

            for idx_combo in idx_combos:

                working_euc_dist = self.elec_euc_arr[idx_combo]

                if working_euc_dist > max_euc_dist:
                    max_euc_dist = working_euc_dist

            all_dists[i] = max_euc_dist

        return all_dists
