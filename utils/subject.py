"""Data fetchers"""

import numpy as np

from .constants import *

class Subject:
    
    def __init__(self, subj, n_parcs=600, n_networks=17, max_length=10):
        self.parcs = n_parcs
        self.networks = n_networks
        self.seq_len = max_length
        
        # subject-specific directories
        self.dirs = {
            k.lower()[:-4]: (v / subj) for k,v in data_directories.items()
        }
        
        # add specific ied dirs
        self.update_ied_subdirs()
    
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
            
            
        