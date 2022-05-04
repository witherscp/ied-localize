from collections import Counter
from itertools import combinations
from os import path, remove
import shlex
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .constants import NUM_MAP

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
    
    # if directory does not exist, make directory and all parents
    odir.mkdir(parents=True, exist_ok=True)
    
    # set-up temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # save out_1D file
        temp_file = path.join(tempdir, f"temp.1D")
        np.savetxt(temp_file, X=array, fmt='%f')

        # run AFNI ConvertDset command to create niml.dset file
        convert_cmd = shlex.split(f'ConvertDset -o_niml -input {temp_file} '
                                  f'-add_node_index -prefix {full_path}')
        # subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, 
        #                stderr=subprocess.STDOUT)
        subprocess.run(convert_cmd)
        
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

def convert_elec_to_lobe(elec2lobe_df, elec):
    """Get a list of lobes associated with a given electrode

    Args:
        elec2lobe_df (pd.DataFrame): electrode to lobe lookup table
        elec (str): electrode name

    Returns:
        list: list of lobes
    """
    
    lobes = elec2lobe_df[elec2lobe_df['elecLabel'] == elec]['Lobe']
    return lobes.iloc[0].replace("'",'').replace(' ','').strip('[]').split(',')
    
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
    """Based on the Engel class and proportion of a parcel resected, return
    the prediction accuracy (TP,TN,FP,FN).

    Args:
        engel_class (str): Engel class or (no_outcome, deceased, no_resection)
        resected_prop (float): proportion of parcel resected
        resected_threshold (float, optional): min proportion of parcel resected
            to consider as fully resected. Defaults to 0.5.
        sz_free (list, optional): Engel classes considered as seizure free. 
            Defaults to ['1a'].

    Returns:
        str: accuracy (TN, TP, FP, FN)
    """
    
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

def compute_elec2parc_euc(elec2parc_euc_arr, elec_idx, parc):
    """Return the Euclidean distance between an electrode index and parcel.

    Args:
        elec2parc_euc_arr (np.array): array of elec2parc Euclidean distances 
            (use: self.parc_minEuclidean_byElec)
        elec_idx (int): electrode index (use: self.get_elec_idx(elec))
        parc (int): parcel number

    Returns:
        float: Euclidean distance between electrode and parcel
    """
    
    return elec2parc_euc_arr[parc-1, elec_idx]

def compute_elec2parc_geo(node2parc_df_dict, elec2node_geo_arr, elec_idx, parc):
    """Return the geodesic distance between an electrode index and parcel.

    Args:
        node2parc_df_dict (dict): keys: "LH","RH"; values: node to parcel df
        elec2node_geo_arr (np.array): array of electrode to node geodesic 
            distances
        elec_idx (int): electrode index (use: self.get_elec_idx(elec))
        parc (int): parcel number

    Returns:
        float: minimum geodesic distance between electrode and parcel
    """ 
    
    n_parcs = int(node2parc_df_dict['LH']['parcel'].max() * 2)
    
    hemi = get_parcel_hemi(parc, n_parcs)
    
    node2parc_df = node2parc_df_dict[hemi]
    
    if hemi == "RH":
        mask = (node2parc_df['parcel'] == (parc - int(n_parcs/2)))
    else:
        mask = (node2parc_df['parcel'] == parc)
    
    parc_nodes = node2parc_df[mask]['node'].to_numpy(dtype=int)
    
    return np.min(elec2node_geo_arr[parc_nodes,elec_idx])

def num2roman(num):
    """Convert a number to Roman numeral

    Args:
        num (int): number

    Returns:
        str: Roman numeral
    """

    roman = ''

    while num > 0:
        for i, r in NUM_MAP:
            while num >= i:
                roman += r
                num -= i

    return roman

def roman2num(num):
    """Convert a Roman numeral to number.

    Args:
        num (str): Roman numeral

    Returns:
        int: number
    """
    
    roman_numerals = {'I':1, 'V':5, 'X':10}
    result = 0
    for i,c in enumerate(num):
        if (i+1) == len(num) or roman_numerals[c] >= roman_numerals[num[i+1]]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    return result

def compute_node2prop_arr(parc2prop_df, node2parc_df_dict, hemi=None):
    """Return an array of proportions explained by individual nodes for plotting purposes.

    Args:
        parc2prop_df (pd.DataFrame): parcel to proportion dataframe
        node2parc_df_dict (dict): dictionary of node to parcel dataframes
        hemi (str, optional): hemisphere; if set to None, will choose the 
            hemisphere of maximal proportion. Defaults to None.

    Returns:
        tuple: np.array of proportions at every node, hemisphere string
    """
    
    if hemi == None:
        # find hemisphere of maximal parcel
        if parc2prop_df['propExplanatorySpikes'].idxmax() < (len(parc2prop_df) // 2):
            hemi = "LH"
        else:
            hemi = "RH"
    
    # load node2parc_df and set proportion default to zero
    node2parc_df = node2parc_df_dict[hemi.upper()]
    node2parc_df['proportion'] = 0
    
    n_parcs = int(node2parc_df['parcel'].max() * 2)

    # get list of possible parcel numbers based on n_parcs and hemi
    possible_parcs = range(1,(int(n_parcs/2) + 1))
    if hemi.lower() == "rh":
        possible_parcs = [(n + int(n_parcs/2)) for n in possible_parcs]

    # update node2parc proportion at each parcel
    for parc in possible_parcs:
        if hemi.lower() == "rh":
            mask = (node2parc_df['parcel'] == (parc - int(n_parcs/2)))
        else:
            mask = (node2parc_df['parcel'] == parc)
        
        # get proportion for particular parcel
        parc_slice = parc2prop_df[parc2prop_df.index == parc]
        proportion = parc_slice['propExplanatorySpikes'].iloc[0]
        
        # set all nodes of particular parcel to correct proportion
        node2parc_df.loc[mask,'proportion'] = proportion
    
    return node2parc_df.proportion.to_numpy(), hemi.upper()

def retrieve_lead_counts(elec_df, seqs, delays, lead_times=[100]):
    """Create a dataframe containing the frequency for which each electrode
    occurs first in sequence. Optionally, use lead_times array to also create 
    columns for the number of times each electrode occurs within the 
    first xx ms. 

    Args:
        elec_df (pd.DataFrame): df of electrode names
        seqs (np.array): n_seqs x n_elecs array with names of electrodes firing
            within each sequence. 'nan' is used to fill blank positions
        delays (np.array): n_seqs x n_elecs array with lag times; np.NaN is 
            used to fill blank positions
        lead_times (list, optional): list of lag times of interest; ex: [20] 
            will create a column in which elecs within the first 20 ms are 
            counted. Defaults to [100].
    """
    
    # set default based on number of output values
    default_val = [0 for _ in range(len(lead_times)+1)]
    
    # fill dictionary with default values
    out_dict = {}
    for elec in elec_df.chanName:
        out_dict.setdefault(elec, default_val.copy())
    
    # iterate through sequences
    for i in range(seqs.shape[0]):
        
        # select all filled positions
        row = seqs[i,:]
        seq_elecs = row[row != "nan"]
        
        # get lead electrode
        leader = seq_elecs[0]
        
        # increment lead electrode frequency in out_dict
        out_dict[leader][0] += 1
        
        n_elecs = seq_elecs.size
        
        for j in range(n_elecs):
            
            elec = seq_elecs[j]
            
            #increment electrode frequencies for within xx ms columns
            lag = delays[i,j]
            for idx, time in enumerate(lead_times):
                if lag < time:
                    out_dict[elec][idx+1] += 1
    
    # initialize names of out columns
    out_cols = ['Leader']
    for time in lead_times:
        out_cols.append(f'Within {time}ms')
    
    out_df = pd.DataFrame.from_dict(out_dict, orient='index', columns=out_cols)
    out_df.sort_values(out_cols, ascending=False, inplace=True)
    
    return out_df

def compute_mean_seq_length(seqs):
    """Compute the mean sequence length

    Args:
        seqs (np.array): sequence of electrodes (n_seqs x n_elecs)

    Returns:
        float: mean sequence length
    """
    
    return seqs[seqs != 'nan'].size / seqs.shape[0]

def compute_mean_similarity(similarity_arr):
    """Compute mean sequence similarity

    Returns:
        float: mean sequence similarity for a given cluster
    """

    return np.mean(similarity_arr)

def compute_weighted_similarity_length(similarity_arr, seqs):
    """Compute metric combining similarity of sequences and mean length. My
    hypothesis is that a combination of greater length and greater similarity 
    makes it more likely that sequences are closer to the epileptogenic zone.

    Args:
        similarity_arr (np.array): square matrix of Jaro-Winkler similarities
        seqs (np.array): sequences of electrodes (n_seqs x n_elecs)

    Returns:
        float: mean similarity weighted by mean length
    """
    
    mean_similarity = compute_mean_similarity(similarity_arr)
    mean_length = compute_mean_seq_length(seqs)
    
    return mean_similarity * mean_length
    