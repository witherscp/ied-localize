from itertools import product
from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from .constants import *
from .helpers import constrain_velocities

def lead_geodesic(Subj, elec_idxs, parc_idxs, lags, hemis, cluster_hemi, 
                  minGeo, maxGeo, minBL, maxBL, only_geo=False, only_wm=False):
    
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
    min_v_geo, max_v_geo = constrain_velocities(min_v_geo, max_v_geo, is_geo=True)

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
            
            # initialize n_node x n_elec arrays to store wm dists
            f_min_wm = np.full((N_NODES,n_followers), np.NaN)
            f_max_wm = np.full((N_NODES,n_followers), np.NaN)
          
            # convert (n_parcs x n_elecs) to (n_nodes x n_elecs)
            for parc_idx in np.unique(np.where(~np.isnan(parc_min_bl))[0]):
                nodes = Subj.parc2node_dict[parc_idx+1]
                f_min_wm[nodes,:] = parc_min_bl[parc_idx,:]
                f_max_wm[nodes,:] = parc_max_bl[parc_idx,:]
      
            # compute min/max velocity geo (combo method) using equations from methods section
            combo_min_v_geo = aveGeo[:,l_elec_idx][:,np.newaxis] / ((f_max_wm / MIN_WM_VEL) - lags)
            combo_max_v_geo = aveGeo[:,l_elec_idx][:,np.newaxis] / ((f_min_wm / MAX_WM_VEL) - lags)
          
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
          
            # shuffle combinations of v/combo_v #######
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
        convert_node2parc = np.vectorize(Subj.node2parc_hemi_dict[cluster_hemi].get)
        source_parcs = set(convert_node2parc(source_indices))
        return [source for source in source_parcs if source != 0]
    
def lead_wm(Subj, elec_idxs, parc_idxs, lags, hemis, cluster_hemi, 
            minGeo, maxGeo, minBL, maxBL, only_geo=False, only_wm=False):
    
    # only_geo means do not attempt localization with wm
    if only_geo:
        return []
    
    l_elec_idx = elec_idxs[0]
    f_elec_idxs = elec_idxs[1:]
    l_hemi = hemis[0]
    f_hemis = hemis[1:]
    l_parc_idxs = parc_idxs[0]
    f_parc_idxs = parc_idxs[1:]
    n_followers = len(f_elec_idxs)
    aveBL = (minBL+maxBL) / 2

    source_parcs = set()
    
    # iterate over parcel combinations
    for l_parc_idx in l_parc_idxs:
        
        for f_parc_idxs_combo in product(*f_parc_idxs):
            
            # compute min/max velocity wm using equation from methods section
            min_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / ((maxBL[:,f_parc_idxs_combo]/MIN_WM_VEL)-lags)
            max_v_wm = aveBL[:,l_parc_idx][:,np.newaxis] / ((minBL[:,f_parc_idxs_combo]/MAX_WM_VEL)-lags)
            min_v_wm, max_v_wm = constrain_velocities(min_v_wm, max_v_wm, is_wm=True)

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
                
                # broadcast min_v_wm and max_v_wm arrays to n_node x n_elec
                # initialize n_node x n_elec arrays to store wm dists
                nodes_min_v_wm = np.full((N_NODES,n_followers), np.NaN)
                nodes_max_v_wm = np.full((N_NODES,n_followers), np.NaN)
            
                for parc_idx in np.unique(np.where(~np.isnan(min_v_wm))[0]):
                    nodes = Subj.parc2node_dict[parc_idx+1]
                    nodes_min_v_wm[nodes,:] = min_v_wm[parc_idx,:]
                    nodes_max_v_wm[nodes,:] = max_v_wm[parc_idx,:]
                
                # find all possible v_wm given a range of constant v_geo
                ## IN PROGRESS ##
                pass
    
    # return list of source parcels
    if len(source_parcs) == 0:
        return []
    else:
        return [source for source in source_parcs if source != 0]