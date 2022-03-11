import numpy as np

from itertools import combinations


def get_frequent_seqs(seqs, n_members=3, n_top=1, ordered=True):
    """Retrieves the {n_top} number of most frequent {n_member} sequences 
    in a cluster.

    Args:
        seqs (np.array): Electrode sequences
        n_members (int, optional): Number of members in a sequence. Defaults
            to 3
        n_top (int, optional): Number of highest ranking sequences. 
            Defaults to 1.
        ordered (bool, optional): Order of electrodes matters. Defaults 
            to True.

    Returns:
        dict: Dictionary of most common sequences and number of occurences.
    """
    
    assert n_top >= 1
    assert n_members >= 2
    assert isinstance(ordered, bool)
    
    seq_groups_dict = {}

    # iterate through sequences
    for i in range(seqs.shape[0]):
        row = seqs[i,:]
        elecs = row[row != "nan"]
        combs_iter = combinations(elecs,n_members)
        
        for group in combs_iter:
            
            if ordered:
                # use lists to store
                seq_groups_dict.setdefault(group,0)
                seq_groups_dict[group] += 1
            else:
                # use sets to store
                seq_groups_dict.setdefault(frozenset(group),0)
                seq_groups_dict[frozenset(group)] += 1
    
    descending_keys = sorted(seq_groups_dict, key=seq_groups_dict.get, 
                             reverse=True)
    
    top_seqs = {k:seq_groups_dict[k] for k in descending_keys[:n_top]}
    
    return top_seqs