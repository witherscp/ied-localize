from itertools import combinations
from os import path, remove
import shlex
import subprocess
import tempfile

import numpy as np

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

def array_to_niml(array, odir, fname):
    """Create a niml.dset file at {odir / fname} using array of shape 
    (n_nodes,1)

    Args:
        array (np.array): values at each node; must have size n_nodes
        odir (pathlib.PosixPath): out directory
        fname (str): fname (excluding niml.dset)
    """
    
    # array must contain the same number of values as std.141 mesh
    assert np.size(array) == 198812
    
    # ensure that array is a column vector
    if array.ndim == 1:
        array = array[:,np.newaxis]
    
    full_path = odir / f"{fname}.niml.dset"
    
    # if file exists already, overwrite with updated version
    if path.exists(full_path):
        remove(full_path)
    
    # set-up temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # save out_1D file
        temp_file = path.join(tempdir, f"temp.1D")
        np.savetxt(temp_file, X=array, fmt='%f')

        # run AFNI ConvertDset command to create niml.dset file
        convert_cmd = shlex.split(f'ConvertDset -o_niml -input {temp_file} '
                                  f'-add_node_index -prefix {full_path}')
        subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, 
                       stderr=subprocess.STDOUT)
        
def compute_top_lead_elec(seqs):
    """Given an array of electrode sequences, return the most frequent lead 
    electrode.

    Args:
        seqs (np.array): electrode sequences of shape n_seq x n_elec

    Returns:
        str: most frequent leading electrode
    """
    
    elecs, counts = np.unique(seqs[:,0], return_counts=True)
    return elecs[np.argmax(counts)]

def convert_elec_to_parc(elec2parc_df, elec, no_index=False):
    """Get a list of parcels (or parcel indices) associated with a given 
    electrode

    Args:
        elec2parc_df (pd.DataFrame): electrode to parcel lookup table
        elec (str): electrode name
        no_index (bool, optional): retrieve the parcel numbers instead of 
            parcel indices. Defaults to False.

    Returns:
        list: list of parcel indices (or numbers)
    """
    
    parcs = elec2parc_df[elec2parc_df['elecLabel'] == elec]['parcNumber']
    parcs = parcs.iloc[0].strip('][').split(', ')
    return map_func(parcs, no_index=no_index)
    
def map_func(lst_str, no_index=False):
    """Map a list of strings to a list of integers

    Args:
        lst_str (list): list of strings
        no_index (bool, optional): if set to true, 1 will not be subtracted 
            from each integer. Defaults to False.

    Returns:
        list: list of integers
    """
    
    lst_int = []
    
    for val in lst_str:
        if val != '':
            if no_index:
                lst_int.append(int(val))
            else:
                # convert to int and subtract one to make an index
                lst_int.append(int(val) - 1)
    
    return lst_int

def get_parcel_hemi(parcel, n_parcs):
    """For a given parcel number, return the hemisphere of that parcel

    Args:
        parcel (int): parcel number in range(1,n_parcs+1)
        n_parcs (int): Schaefer parcellation

    Returns:
        str: hemisphere ("LH" or "RH")
    """
    
    assert parcel in range(1, n_parcs + 1)
    
    if parcel <= (n_parcs / 2):
        return "LH"
    else:
        return "RH"
    
def get_prediction_accuracy(engel_class, 
                            resected_prop,
                            resected_threshold = 0.5,
                            sz_free=['1a']):
    
    if resected_prop >= resected_threshold:
        resected = True
    else:
        resected = False

    # engel class values that would exclude patient
    no_outcome = ['no_outcome', 'deceased', 'no_resection']
    
    if engel_class in no_outcome:
        return "N/A"
    elif engel_class in sz_free:
        if resected:
            return "TP"
        else:
            return "FN"
    else:
        if resected:
            return "FP"
        else:
            return "TN"