from itertools import combinations, repeat
from time import time

import numpy as np
import pandas as pd

from .colors import Colors
from .constants import NUM_MAP


def _map_func(lst_str, no_index=False):
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
        if val != "":
            if no_index:
                lst_int.append(int(val))
            else:
                # convert to int and subtract one to make an index
                lst_int.append(int(val) - 1)

    return lst_int


def convert_parcs(parc_str):
    """Convert a string of a list of parcels to a list of integers.

    Args:
        parc_str (str): str of list of parcel numbers

    Returns:
        list: list of parcel numbers (not indices)
    """

    lst_str = parc_str.strip("][").split(", ")
    return _map_func(lst_str, no_index=True)


def convert_lobes(lobe_str):
    """Convert a string of a list of lobes to a list of strings

    Args:
        lobe_str (str): str of list of lobe names

    Returns:
        list: list of string lobe names
    """

    return lobe_str.replace("'", "").replace(" ", "").strip("[]").split(",")


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
        row = seqs[i, :]
        elecs = row[row != "nan"]
        combs_iter = combinations(elecs, n_members)

        for group in combs_iter:

            if ordered:
                # use lists to store
                seq_groups_dict.setdefault(group, 0)
                seq_groups_dict[group] += 1
            else:
                # use sets to store
                seq_groups_dict.setdefault(frozenset(group), 0)
                seq_groups_dict[frozenset(group)] += 1

    descending_keys = sorted(seq_groups_dict, key=seq_groups_dict.get, reverse=True)

    top_seqs = {k: seq_groups_dict[k] for k in descending_keys[:n_top]}

    return top_seqs


def compute_top_lead_elec(seqs):
    """Given an array of electrode sequences, return the most frequent lead
    electrode.

    Args:
        seqs (np.array): electrode sequences of shape n_seq x n_elec

    Returns:
        str: most frequent leading electrode
    """

    elecs, counts = np.unique(seqs[:, 0], return_counts=True)
    return elecs[np.argmax(counts)]


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


def get_prediction_accuracy(
    engel_class, resected_prop, resected_threshold=0.5, sz_free=["1a", "1b", "1c"]
):
    """Based on the Engel class and proportion of a parcel resected, return
    the prediction accuracy (TP,TN,FP,FN).

    Args:
        engel_class (str): Engel class or (no_outcome, deceased, no_resection)
        resected_prop (float): proportion of parcel resected
        resected_threshold (float, optional): min proportion of parcel resected
            to consider as fully resected. Defaults to 0.5.
        sz_free (list, optional): Engel classes considered as seizure free.
            Defaults to ['1a','1b','1c'].

    Returns:
        str: accuracy (TN, TP, FP, FN)
    """

    if resected_prop >= resected_threshold:
        resected = True
    else:
        resected = False

    # engel class values that would exclude patient
    no_outcome = ["no_outcome", "deceased", "no_resection"]

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

    return elec2parc_euc_arr[parc - 1, elec_idx]


def compute_elec2parc_geo(
    parc2node_dict, elec2node_geo_arr, elec_idx, parc, func=np.min
):
    """Return the geodesic distance between an electrode index and parcel.

    Args:
        parc2node_dict (dict): keys: parcel number; values: node list
        elec2node_geo_arr (np.array): array of electrode to node geodesic
            distances
        elec_idx (int): electrode index (use: self.get_elec_idx(elec))
        parc (int): parcel number
        func (function): function applied to array; should be np.min for
            minimum distances and np.max for maximum distances; defaults to
            np.min

    Returns:
        float: minimum geodesic distance between electrode and parcel
    """

    parc_nodes = parc2node_dict[parc]

    return func(elec2node_geo_arr[parc_nodes, elec_idx])


def num2roman(num):
    """Convert a number to Roman numeral

    Args:
        num (int): number

    Returns:
        str: Roman numeral
    """

    roman = ""

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

    roman_numerals = {"I": 1, "V": 5, "X": 10}
    result = 0
    for i, c in enumerate(num):
        if (i + 1) == len(num) or roman_numerals[c] >= roman_numerals[num[i + 1]]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    return result


def retrieve_lead_counts(elec_names, seqs, delays, lead_times=[100]):
    """Create a dataframe containing the frequency for which each electrode
    occurs first in sequence. Optionally, use lead_times array to also create
    columns for the number of times each electrode occurs within the
    first xx ms.

    Args:
        elec_names (list): list of electrode names
        seqs (np.array): n_seqs x n_elecs array with names of electrodes firing
            within each sequence. 'nan' is used to fill blank positions
        delays (np.array): n_seqs x n_elecs array with lag times; np.NaN is
            used to fill blank positions
        lead_times (list, optional): list of lag times of interest; ex: [20]
            will create a column in which elecs within the first 20 ms are
            counted. Defaults to [100].
    """

    # set default based on number of output values
    default_val = [0 for _ in range(len(lead_times) + 1)]

    # fill dictionary with default values
    out_dict = {}
    for elec in elec_names:
        out_dict.setdefault(elec, default_val.copy())

    # iterate through sequences
    for i in range(seqs.shape[0]):

        # select all filled positions
        row = seqs[i, :]
        seq_elecs = row[row != "nan"]

        # get lead electrode
        leader = seq_elecs[0]

        # increment lead electrode frequency in out_dict
        out_dict[leader][0] += 1

        n_elecs = seq_elecs.size

        for j in range(n_elecs):

            elec = seq_elecs[j]

            # increment electrode frequencies for within xx ms columns
            lag = delays[i, j]
            for idx, time in enumerate(lead_times):
                if lag < time:
                    out_dict[elec][idx + 1] += 1

    # initialize names of out columns
    out_cols = ["Leader"]
    for time in lead_times:
        out_cols.append(f"Within {time}ms")

    out_df = pd.DataFrame.from_dict(out_dict, orient="index", columns=out_cols)
    out_df.sort_values(out_cols, ascending=False, inplace=True)

    return out_df


def compute_mean_seq_length(seqs):
    """Compute the mean sequence length.

    Args:
        seqs (np.array): sequence of electrodes (n_seqs x n_elecs)

    Returns:
        float: mean sequence length
    """

    return seqs[seqs != "nan"].size / seqs.shape[0]


def compute_mean_similarity(similarity_arr):
    """Compute mean sequence similarity.

    Args:
        similarity_arr (np.array): array of Jaro similarities (n_seqs, n_seqs)

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


def retrieve_delays(delays, seq_idxs, include_zero=False):
    """Return array of delay times for sequence indices of interest. Hypothesis
    is that white matter sequences will have faster lag times than gray matter
    only.

    Args:
        delays (np.array): array of lag times
            (use: _, delays = s.fetch_sequences(cluster))
        seq_idxs (np.array): array of sequence indices
        include_zero (bool): include all of the 0ms lag times. Defaults to
            False.

    Returns:
        np.array: array of lag times for indexed sequences
    """

    if seq_idxs.size == 0:
        return np.array(())

    indexed_delays = delays[seq_idxs]

    if include_zero:
        mask = indexed_delays >= 0
    else:
        mask = indexed_delays > 0

    return indexed_delays[mask]


def output_lst_of_lsts(lst_of_lsts, my_dtype=float):
    """Convert a list of lists into numpy array.

    Args:
        lst_of_lsts (list): list of lists
        my_dtype (_type_, optional): type of output array (typically float or
            object). Defaults to float.

    Returns:
        np.array: final array with np.NaN filled in empty spaces
    """

    max_length = 1  # initialize max number of sources for any sequence
    for lst in lst_of_lsts:
        if len(lst) > max_length:
            max_length = len(lst)

    out_array = np.full((len(lst_of_lsts), max_length), np.NaN, dtype=my_dtype)
    for i, lst in enumerate(lst_of_lsts):
        out_array[i, 0 : len(lst)] = lst

    return out_array


def reorient_coord(coord_arr, in_orient, out_orient):
    """
    Change the orientation of a set of XYZ coordinates.

    Args:
        coord_arr (np.array): XYZ coordinate array of shape (n_coords,3) with
            columns corresponding to X, Y and Z axes
        in_orient (str): 3-letter code indicating orientation of input
            coordinates
        out_orient (str): 3-letter code indicating desired orientation of
            output coordinates

    Returns:
        np.array: XYZ coordinate array (same shape as coord_arr) with new
        orientation specified in "out_orient"
    """
    # define axis codes
    axis_codes_dict = {0: ["L", "R"], 1: ["A", "P"], 2: ["I", "S"]}

    change_axis_dict = dict(zip(range(3), repeat(True)))
    for idx in range(3):
        for code in axis_codes_dict[idx]:
            if code in in_orient.upper() and code in out_orient.upper():
                change_axis_dict[idx] = False
                break

    # change axes that need to be changed
    for axis, change_bool in change_axis_dict.items():
        if change_bool:
            coord_arr[:, axis] = -coord_arr[:, axis]

    return coord_arr


def convert_list_to_dict(lst):
    """Convert a list of source parcel lists to a dictionary with the
    proportion of sequences explained by each parcel.

    Args:
        lst (list): list of source lists (in do_localize_source.py)

    Returns:
        dict: dictionary with proportion of sequences explained by each source
            parcel
    """

    out_dict = {}

    for parcel_list in lst:
        for parcel in parcel_list:
            out_dict.setdefault(parcel, 0)
            out_dict[parcel] += 1

    return out_dict


def output_normalized_counts(num_counts, hemi_dict, n_parcs):
    """Convert source count dictionary to df for saving as csv.

    Args:
        num_counts (int): total number of sequences
        hemi_dict (dict): dictionary of normalized source proportions
        n_parcs (int): number of parcs total

    Returns:
        pd.DataFrame: dataframe index = parcNumber,
            column = propExplanatorySpikes
    """

    normalized_lst = [(count / num_counts) for count in hemi_dict.values()]
    output_df = pd.DataFrame(
        {"parcNumber": hemi_dict.keys(), "propExplanatorySpikes": normalized_lst}
    )
    output_df.set_index("parcNumber", inplace=True)
    output_df.sort_index(inplace=True)
    output_df = output_df.reindex(list(range(1, n_parcs + 1)), fill_value=0)

    return output_df


def convert_geo_arrays(Subj, minGeo, maxGeo):
    """Convert n_node x n_elec geodesic arrays to n_parc x n_elec.

    Args:
        Subj (Subject): instance of Subject class
        minGeo (np.array): n_node x n_elec array of min geodesic distances
        maxGeo (np.array): n_node x n_elec array of max geodesic distances

    Returns:
        tuple: parc_minGeo, parc_maxGeo (np.arrays with shape n_parc x n_elec)
    """

    parc_minGeo = np.full((Subj.parcs, minGeo.shape[1]), np.NaN)
    parc_maxGeo = np.full((Subj.parcs, minGeo.shape[1]), np.NaN)

    for parc_idx in range(Subj.parcs):
        parc_minGeo[parc_idx, :] = np.nanmin(
            minGeo[Subj.parc2node_dict[parc_idx + 1]], axis=0
        )
        parc_maxGeo[parc_idx, :] = np.nanmax(
            maxGeo[Subj.parc2node_dict[parc_idx + 1]], axis=0
        )

    hemis = np.array(list(Subj.elec2hemi_dict.values()))

    # remove values for electrodes whose nodes were in other hemisphere
    parc_minGeo[Subj.parcs // 2 :, hemis == "LH"] = np.NaN
    parc_minGeo[: Subj.parcs // 2, hemis == "RH"] = np.NaN

    return parc_minGeo, parc_maxGeo


def hms(seconds):
    """Convert seconds to hours, minutes, and seconds.

    Args:
        seconds (int): number of seconds

    Returns:
        str: {hours}h:{minutes}m:seconds{s}
    """

    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


def print_progress_update(i, start_time, n_seqs):
    """Print out progress update on source localization.

    Args:
        i (int): current working sequence
        start_time (time.Time): time when localization started
        n_seqs (int): number of sequences total
    """

    # print out progress update for user to track
    time_now = time()
    time_elapsed = time_now - start_time
    str_time_elapsed = hms(round(time_elapsed))
    time_remaining = (time_elapsed / i) * (n_seqs - i)
    str_time_left = hms(round(time_remaining))
    print(Colors.YELLOW, f"#{i}/{n_seqs} Sequences Complete:", Colors.END)
    print(
        Colors.BLUE,
        (
            f"Time Elapsed = {str_time_elapsed} ... "
            f"Estimated Time Remaining = {str_time_left}"
        ),
        Colors.END,
    )


def extend_lst_of_lsts(lst_of_lsts):
    """Uniformize the length of each list in a list_of_lists by repeatedly
    appending the last term to each list, until lengths match. Ex:
    [[0,1], [2,4,5], [1]] --> [[0,1,1], [2,4,5], [1,1,1]]. This function is
    used in lead_gm() within localize.py.

    Args:
        lst_of_lsts (list): list of lists

    Returns:
        list: list of lists with uniform lengths
    """

    max_len = max([len(lst) for lst in lst_of_lsts])
    out_lst = []

    for lst in lst_of_lsts:

        while len(lst) < max_len:
            lst.append(lst[-1])

        out_lst.append(lst)

    return out_lst


def get_geodesic_dist(source_nodes, target_nodes, pial):
    """Return the geodesic distances between source nodes and target nodes.

    Args:
        source_nodes (np.array): array of one or more source nodes
        target_nodes (np.array): array of one or more target nodes
        pial (nilearn.surface.surface.mesh): surface for given hemisphere,
            created using s.fetch_pial_surface_dict()[hemi]

    Returns:
        np.array: array of geodesic distances
    """
    import pygeodesic.geodesic as geodesic

    # run pygeodesic to get geodesic distance between node and all other nodes
    geoalg = geodesic.PyGeodesicAlgorithmExact(pial.coordinates, pial.faces)
    distances, _ = geoalg.geodesicDistances(source_nodes, target_nodes)

    return distances
