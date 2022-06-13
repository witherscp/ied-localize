from itertools import product
from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

from .constants import *


def lead_geodesic(Subj, elec_idxs, parc_idxs, lags, hemis, cluster_hemi, 
                  minGeo, maxGeo, minBL, maxBL, only_geo=False, only_wm=False,
                  n_steps=20):
    """Localize the sources of a sequence based on the assumption that the lead
    electrode receives signal via geodesic spread. Returns a list of unique 
    parcel numbers.

    Args:
        Subj (Subject): instance of Subject class
        elec_idxs (np.array): array of electrode indices (for indexing min/max 
            geodesic arrays)
        parc_idxs (list): list of parcel index lists associated with elecs
        lags (np.array): array of follower electrode lag times
        hemis (np.array): array of electrode hemispheres
        cluster_hemi (str): hemisphere associated with cluster (a geodesic 
            source can only come from this cluster)
        minGeo (np.array): minimum geodesic distances (n_nodes, n_elecs)
        maxGeo (np.array): maximum geodesic distances (n_nodes, n_elecs)
        minBL (np.array): minimum bundle lengths (n_parcs, n_parcs)
        maxBL (np.array): maximum bundle lengths (n_parcs, n_parcs)
        only_geo (bool, optional): require followers to use geodesic 
            propagation. Defaults to False.
        only_wm (bool, optional): require all white matter propagation 
            (immediately returns []). Defaults to False.
        n_steps (int, optional): number of steps used in linspace; a larger 
            number will increase max/min range but slow down algorithm. 
            Defaults to 20.

    Returns:
        list: list of unique source parcel numbers (if [], no parcels explain 
            lag times)
    """
    
    # only_wm means do not attempt localization with geodesic
    if only_wm:
        return []

    l_hemi = hemis[0]

    # lead must always be in the same hemisphere as the cluster
    if l_hemi != cluster_hemi:
        return []
    
    # if only geodesic allowed, check that all electrodes are in same
    # hemisphere as cluster
    if only_geo:
        if (np.any(hemis != cluster_hemi)): return []

    f_hemis = hemis[1:]
    l_elec_idx = elec_idxs[0]
    f_elec_idxs = elec_idxs[1:]
    f_parc_idxs = parc_idxs[1:]
    n_followers = len(f_elec_idxs)
    aveGeo = (minGeo + maxGeo) / 2

    # compute min/max velocity geo using equation from methods section
    min_v_geo = (minGeo[:,f_elec_idxs] - aveGeo[:,l_elec_idx][:,np.newaxis]) / lags
    max_v_geo = (maxGeo[:,f_elec_idxs] - aveGeo[:,l_elec_idx][:,np.newaxis]) / lags
    min_v_geo, max_v_geo = constrain_velocities(min_v_geo, max_v_geo, 
                                                is_geo=True)

    # check possibilities where followers are only geodesic
    if only_geo:
      
        # check for existence of overlapping range
        largest_min_v = np.max(min_v_geo,axis=1)
        smallest_max_v = np.min(max_v_geo,axis=1)
        source_indices = np.where(smallest_max_v > largest_min_v)[0]
    
    # check possibilities where followers are geodesic or WM
    else:
        
        # for elecs in different hemisphere, convert all values to np.NaN
        other_hemi_mask = (f_hemis != cluster_hemi)
        min_v_geo[:, other_hemi_mask] = np.NaN
        max_v_geo[:, other_hemi_mask] = np.NaN
        
        # iterate through parcel combos
        for f_parc_idxs_combo in product(*f_parc_idxs):
        
            # fill arrays
            parc_min_bl = minBL[:,f_parc_idxs_combo]
            parc_max_bl = maxBL[:,f_parc_idxs_combo]
            
            # find min/max of denominator (dist/vel - lags) on interval [0,inf]
            parc_min_denom, parc_max_denom = find_denom_range(parc_min_bl,
                                                              parc_max_bl,
                                                              MIN_WM_VEL,
                                                              MAX_WM_VEL,
                                                              lags,
                                                              n_steps=n_steps)
            
            # initialize n_node x n_elec arrays to store denominators
            min_denom = np.full((N_NODES,n_followers), np.NaN)
            max_denom = np.full((N_NODES,n_followers), np.NaN)
          
            # convert (n_parcs x n_elecs) to (n_nodes x n_elecs)
            for parc_idx in np.unique(np.where(~np.isnan(parc_min_denom))[0]):
                nodes = Subj.parc2node_dict[parc_idx+1]
                min_denom[nodes,:] = parc_min_denom[parc_idx,:]
                max_denom[nodes,:] = parc_max_denom[parc_idx,:]
      
            # compute min/max velocity geo (combo method) using equations from 
            # methods section
            combo_min_v_geo = aveGeo[:,l_elec_idx][:,np.newaxis] / max_denom
            combo_max_v_geo = aveGeo[:,l_elec_idx][:,np.newaxis] / min_denom
          
            (combo_min_v_geo, 
            combo_max_v_geo) = constrain_velocities(combo_min_v_geo, 
                                                    combo_max_v_geo, 
                                                    is_geo=True)
          
            # create matrix of shape (N_NODES,n_elecs,v/combo_v,min/max)
            master_v = np.stack((np.stack((min_v_geo,combo_min_v_geo), 
                                           axis=-1), 
                                np.stack((max_v_geo,combo_max_v_geo), 
                                          axis=-1)), 
                                axis=-1)
          
            # shuffle combinations of v/combo_v
            # shuffled array has shape (N_NODES,2**n_elecs,n_elecs,min/max)
            # each index in the second dimension is a different combination
            # of v/combo_v
            shuffled = master_v[:,
                                range(n_followers),
                                list(product(range(2), repeat=n_followers)),
                                :]
          
            # find overlapping interval along axis=2 (electrodes)
            # largest_min and smallest_max have shape (N_NODES,2**n_elecs)
            largest_min_v = np.max(shuffled[:,:,:,0],axis=2)
            smallest_max_v = np.min(shuffled[:,:,:,1],axis=2)
          
            # find any combination that had an overlapping interval for a node
            overlapping = np.any((smallest_max_v > largest_min_v), axis=1)
            source_indices = np.where(overlapping)[0]
        
    # convert indices to parcels
    if source_indices.size == 0:
        return []
    else:
        convert_node2parc = np.vectorize(
            Subj.node2parc_hemi_dict[cluster_hemi].get
        )
        source_parcs = set(convert_node2parc(source_indices))
        return [source for source in source_parcs if source != 0]
    
def lead_wm(elec_idxs, parc_idxs, lags, parc_minGeo, parc_maxGeo, minBL, 
            maxBL, only_geo=False, only_wm=False, n_steps=10):
    """Localize the sources of a sequence based on the assumption that the lead
    electrode receives signal via geodesic spread. Returns a list of unique 
    parcel numbers.

    Args:
        Subj (Subject): instance of Subject class
        elec_idxs (np.array): array of electrode indices (for indexing min/max 
            parc geodesic arrays)
        parc_idxs (list): list of parcel index lists associated with elecs
        lags (np.array): array of follower electrode lag times
        parc_minGeo (np.array): minimum geodesic distances (n_parcs, n_elecs)
        parc_maxGeo (np.array): maximum geodesic distances (n_parcs, n_elecs)
        minBL (np.array): minimum bundle lengths (n_parcs, n_parcs)
        maxBL (np.array): maximum bundle lengths (n_parcs, n_parcs)
        only_geo (bool, optional): require all geodesic propagation 
            (immediately returns []). Defaults to False.
        only_wm (bool, optional): require followers to use white matter 
            propagation. Defaults to False.
        n_steps (int, optional): number of steps used in linspace; a larger 
            number will increase max/min range but slow down algorithm. 
            Defaults to 10.

    Returns:
        list: list of unique source parcel numbers (if [], then no parcels 
            explain lag times)
    """
    
    # only_geo means do not attempt localization with wm
    if only_geo:
        return []
    
    f_elec_idxs = elec_idxs[1:]
    l_parc_idxs = parc_idxs[0]
    f_parc_idxs = parc_idxs[1:]
    n_followers = len(f_elec_idxs)
    aveBL = (minBL+maxBL) / 2

    source_parcs = set()
    
    if not only_wm:
        
        # find min/max of denominator (dist/vel - lags) on interval [0,inf]
        min_denom, max_denom = find_denom_range(parc_minGeo[:,f_elec_idxs],
                                                parc_maxGeo[:,f_elec_idxs],
                                                MIN_GEO_VEL,
                                                MAX_GEO_VEL,
                                                lags,
                                                fixed_velocity=True,
                                                n_steps=n_steps)
        
        # num of steps used for testing velocities and distances
        n_steps = min_denom.shape[0]
    
    # iterate over parcel combinations
    for l_parc_idx in l_parc_idxs:
        
        if not only_wm:
            # find all possible combo_v_wm given a range of constant v_geo
            # these have shape (n_steps, n_parcs, n_followers)
            combo_min_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / max_denom
            combo_max_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / min_denom
            
            (combo_min_v_wm, 
             combo_max_v_wm) = constrain_velocities(combo_min_v_wm, 
                                                    combo_max_v_wm, 
                                                    is_wm=True)
        
        for f_parc_idxs_combo in product(*f_parc_idxs):
            
            # fill arrays
            parc_min_bl = minBL[:,f_parc_idxs_combo]
            parc_max_bl = maxBL[:,f_parc_idxs_combo]
            
            # find min/max of denominator (dist/vel - lags) on interval [0,inf]
            parc_min_denom, parc_max_denom = find_denom_range(parc_min_bl,
                                                              parc_max_bl,
                                                              MIN_WM_VEL,
                                                              MAX_WM_VEL,
                                                              lags,
                                                              n_steps=n_steps)
            
            # compute min/max velocity wm using equation from methods section
            min_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / parc_max_denom
            max_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / parc_min_denom
            min_v_wm, max_v_wm = constrain_velocities(min_v_wm, 
                                                      max_v_wm, 
                                                      is_wm=True)

            # check possibilities where followers are only WM
            if only_wm:
            
                # check for existence of overlapping range
                largest_min_v = np.max(min_v_wm,axis=1)
                smallest_max_v = np.min(max_v_wm,axis=1)
                source_parcs = source_parcs.union(
                    set(np.where(smallest_max_v > largest_min_v)[0])
                )
            
            # check possibilities where followers are WM or geodesic
            else:
                
                # broadcast min/max_v_wm arrays to stack with combo arrays
                # these will have shape (n_steps, n_parcs, n_followers)
                min_v_wm = np.broadcast_to(min_v_wm,(n_steps,*min_v_wm.shape))
                max_v_wm = np.broadcast_to(max_v_wm,(n_steps,*max_v_wm.shape))
                
                # stack arrays to find all combinations, creates array of shape
                # (n_steps, n_parcs, n_followers, v/combo_v, min/max)
                master_v = np.stack((np.stack((min_v_wm,combo_min_v_wm),axis=-1),
                                     np.stack((max_v_wm,combo_max_v_wm),axis=-1)),
                                    axis=-1)
                
                # shuffle combinations of v/combo_v, creating array with shape
                # (n_steps, n_parcs, 2**n_followers, n_followers, min/max)
                # each index in the second dimension is a different combination
                # of v/combo_v
                shuffled = master_v[:,
                                    :,
                                    range(n_followers),
                                    list(product(range(2), repeat=n_followers)),
                                    :]
                
                # find overlapping interval along axis=3 (electrodes)
                # largest_min and smallest_max have shape 
                # (n_steps, n_parcs, 2**n_elecs)
                largest_min_v = np.max(shuffled[:,:,:,:,0],axis=3)
                smallest_max_v = np.min(shuffled[:,:,:,:,1],axis=3)
            
                # find any combination that had an overlapping interval for a 
                # parcel
                source_parcs = source_parcs.union(
                    set(np.where(smallest_max_v > largest_min_v)[1])
                )
    
    # return list of source parcels
    if len(source_parcs) == 0:
        return []
    else:
        return [parc_idx+1 for parc_idx in source_parcs]
    
def find_denom_range(min_dist, max_dist, min_vel, max_vel, lags, 
                     fixed_velocity=False, n_steps=50):
    """Find min/max range of localization denominator.

    Args:
        min_dist (np.array): minimum distance from source to followers
        max_dist (np.array): maximum distance from source to followers
        min_vel (float): minimum velocity (geodesic or wm)
        max_vel (float): maximum velocity (geodesic or wm)
        lags (np.array): array of lag times to followers
        fixed_velocity (bool, optional): followers must receive signal at the 
            same velocity (used for lead wm/ follower geodesic). Defaults to 
            False.
        n_steps (int, optional): number of steps used in linspace; a larger 
            number will increase max/min range but slow down algorithm. 
            Defaults to 50.

    Returns:
        tuple: minimum denominator (np.array), maximum denominator (np.array)
    """
    
    # shape (n_steps,n_regions,n_elecs)
    dist_range = np.linspace(min_dist, max_dist, n_steps)
    
    # shape (n_steps)
    vel_range = np.linspace(min_vel, max_vel, n_steps)
    
    # calculate denominator with equation (dist/vel - lags)
    # shape (n_steps,n_steps,n_regions,n_elecs)
    denom = (dist_range[np.newaxis,:] / vel_range.reshape((-1,1,1,1))) - lags
    
    # remove nonzero values
    denom[denom < 0] = np.NaN
    
    if fixed_velocity:
        axes = 1
    else:
        axes = (0,1)
    
    # minimize and maximize denominator over different distances and (
    # optionally) velocities
    min_denom = np.nanmin(denom, axis=axes)
    max_denom = np.nanmax(denom, axis=axes)
    
    return min_denom, max_denom

def constrain_velocities(min_arr, max_arr, is_geo=False, is_wm=False):
    """Given min and maximum velocity intervals, constrain them based on 
    physiological conduction velocities.

    Args:
        min_arr (np.array): array of minimum velocities
        max_arr (np.array): array of maximum velocities
        is_geo (bool, optional): Use geodesic constraints. Defaults to False.
        is_wm (bool, optional): Use white matter constraints. Defaults to 
            False.

    Returns:
        tuple: min_arr, max_arr (after undergoing constraints)
    """
    
    assert is_geo or is_wm
    
    if is_geo:
        min_arr = np.maximum(min_arr, MIN_GEO_VEL)
        max_arr = np.minimum(max_arr, MAX_GEO_VEL)
    else:
        min_arr = np.maximum(min_arr, MIN_WM_VEL)
        max_arr = np.minimum(max_arr, MAX_WM_VEL)
    
    return min_arr, max_arr