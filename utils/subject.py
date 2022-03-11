"""Subject class"""

import numpy as np
import pandas as pd

from .constants import *

class Subject:
    
    def __init__(self, subj, n_parcs=600, n_networks=17, max_length=10):
        self.subj = subj
        self.parcs = n_parcs
        self.networks = n_networks
        self.seq_len = max_length
        
        # subject-specific directories
        self.dirs = {
            k.lower()[:-4]: (v / self.subj) for k,v in data_directories.items()
        }
        self.update_ied_subdirs()
        
        # update cluster attributes
        self.update_num_clusters()
        self.update_cluster_types()
    
    def update_ied_subdirs(self):
        """add ied subdirectories"""
        
        dir_name = f"Schaefer_{self.parcs}P_{self.networks}N"
        
        for val, dir in [('seqs', 'sequence_classification'),
                         ('raw_fc', 'raw_fc'),
                         ('sc', f'sc/{dir_name}'),
                         ('source_loc', f'source_localization/{dir_name}')]:
            
            self.dirs[val] = self.dirs['ied'] / dir
    
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
        
        seqs_file = self.dirs['seqs'] / f"elecSequences_max{self.seq_len}{suffix}.csv"
        seqs = np.loadtxt(seqs_file, dtype=str, delimiter=",")
        
        delays_file = self.dirs['seqs'] / f"delaySequences_max{self.seq_len}{suffix}.csv"
        delays = np.loadtxt(delays_file, dtype=float,delimiter=",")
        
        return seqs, delays
    
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
        
        
            
            
        