"""Subject class"""

from glob import glob
from math import isnan

import numpy as np
import pandas as pd

from .constants import *
from .helpers import *

class Subject:
    
    def __init__(self, subj, n_parcs=600, n_networks=17, max_length=10, 
                 dist=45):
        
        # set values
        self.subj = subj
        self.parcs = n_parcs
        self.networks = n_networks
        self.seq_len = max_length
        
        # set subject-specific directories
        self.dirs = {
            k.lower()[:-4]: (v / self.subj) for k,v in data_directories.items()
        }
        self.update_ied_subdirs()
        self.update_mri_subdirs()
        
        # get arrays and dataframes that are constant
        self.elec_labels_df = self.fetch_elec_labels_df()
        self.elec2parc_df = self.fetch_elec2parc_df()
        self.parc_minEuclidean_byElec = self.fetch_parc_minEuclidean_byElec()
        self.node2parc_df_dict = self.fetch_node2parc_df_dict()
        
        # update attributes
        self.update_num_clusters()
        self.update_cluster_types()
        self.update_source_parcels(dist=dist)
        self.update_engel_class()
        
        if self.engel_class not in ("no_resection","no_outcome","deceased"):
            self.node2rsxn_df_dict = self.fetch_node2rsxn_df_dict()
    
    def update_ied_subdirs(self):
        """add ied subdirectories"""
        
        dir_name = f"Schaefer_{self.parcs}P_{self.networks}N"
        
        for val, dir in [('seqs', 'sequence_classification'),
                         ('raw_fc', 'raw_fc'),
                         ('sc', f'sc/{dir_name}'),
                         ('source_loc', f'source_localization/{dir_name}')]:
            
            self.dirs[val] = self.dirs['ied'] / dir
    
    def update_mri_subdirs(self):
        """add mri subdirectories"""
        
        for val, dir in [('surf', 'surf/xhemi/std141/orig'),
                         ('general', 'surf/xhemi/std141/orig/general'),
                         ('align_elec_alt', f'icEEG/align_elec_alt')]:
            
            self.dirs[val] = self.dirs['mri'] / dir
    
    def fetch_sequences(self, cluster=None):
        """Fetch electrode sequences and lag times once they have been saved
        as .csv files in self.dirs['seqs']

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
    
    def fetch_parc_minEuclidean_byElec(self):
        """Fetch array of minimum Euclidean distances between parcels and 
        electrodes

        Returns:
            np.Array: array Euclidean distances with shape (n_parcs, n_elecs)
        """
        
        fpath = self.dirs['sc'] / "parc_minEuclidean_byElec.csv"
        return np.loadtxt(fpath, delimiter=',', dtype=float)
    
    def fetch_node2parc_df_dict(self):
        """Fetch node to parcel look-up table for given hemisphere

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
    
    def fetch_elec2parc_df(self):
        """Fetch elec to parcel look-up table
        
        Returns:
            pd.DataFrame: dataframe with "elecLabel" and "parcNumber" columns
        """
    
        return pd.read_csv((self.dirs['sc'] / "elec_to_parc.csv"))
    
    def fetch_elec_labels_df(self):
        """Fetch df of all electrode names for conversion to index
        
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
    
    def fetch_node2rsxn_df_dict(self):
        """Create dictionary of node to resection lookup tables for each hemisphere

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
    
    def update_num_clusters(self):
        """update self.num_clusters value"""

        # load cluster summary df
        in_file = f"cluster_summary_max{self.seq_len}.csv"
        try:
            df = pd.read_csv((self.dirs['seqs'] / in_file))
            self.num_clusters = max(set(df['Cluster Number']))
        except FileNotFoundError:
            self.num_clusters = "Clustering has not been run"
            
    def update_cluster_types(self):
        """update self.valid_clusters and self.soz_clusters"""
        
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
            self.valid_clusters = [
                i for i in range(1,self.num_clusters+1) if i not in bad_clusters
            ]    
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
                self.soz_clusters = [int(clust) for clust in soz_clusters]  
        except IndexError:
            self.soz_clusters = 'Subject not found in ied_soz_clusters.csv'
            
    def update_source_parcels(self, dist=45):
        """Update self.valid_sources_all with a set of every possible source 
        for each cluster. Update self.valid_sources_one with a single source 
        for each cluster (choosing the one that is closest to the most frequent
        lead electrode)

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
                
                # compute most frequent lead electrode
                seqs, _ = self.fetch_sequences(cluster)
                lead = compute_top_lead_elec(seqs)
                
                # get lead index
                lead_idx = self.elec_labels_df[
                    self.elec_labels_df['chanName'] == lead
                ].index[0]
                
                # find closest parcel to lead_idx
                top_parc_idxs = [parc - 1 for parc in valid_sources_all[cluster]]
                min_loc = np.argmin(self.parc_minEuclidean_byElec[top_parc_idxs, lead_idx])
                valid_sources_one[cluster] = set([top_parc_idxs[min_loc] + 1])
                
            else:
                valid_sources_one[cluster] = valid_sources_all[cluster]
        
        self.valid_sources_all = valid_sources_all
        self.valid_sources_one = valid_sources_one
    
    def update_engel_class(self):
        """Update self.engel_class and self.engel_months"""
        
        
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
    
    def compute_lead_elec_parc2prop_df(self, cluster):
        """Compute a lookup table of parcel to proportion explained for leading
        electrodes

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
        """Retrieve the resected proportion of a parcel for a given hemisphere

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