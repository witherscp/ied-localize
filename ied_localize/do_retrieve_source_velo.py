from argparse import ArgumentParser
from time import time
from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from ied_localize.utils.colors import Colors
from ied_localize.utils.helpers import (
    convert_geo_arrays,
    print_progress_update,
    get_parcel_hemi,
)
from ied_localize.utils.localize import lead_gm_velocities, lead_wm_velocities
from ied_localize.utils.subject import Subject

if __name__ == "__main__":

    # parse arguments
    purpose = ""
    parser = ArgumentParser(description=purpose)
    parser.add_argument("subj", help="subject code")
    parser.add_argument("--cluster", type=int, help="cluster")
    parser.add_argument(
        "-p", "--parcs", default=600, help=("Schaefer parcellation; defaults to 600"),
    )
    parser.add_argument(
        "-n", "--networks", default=17, help="Yeo network {7 or 17}; defaults to 17"
    )
    parser.add_argument(
        "--dist", default=45, help="GM max search distance; defaults to 45 mm"
    )
    parser.add_argument(
        "-l", "--max_length", default=10, help="maximum allowable length of a sequence"
    )
    parser.add_argument(
        "--only_gm", action="store_true", help="use gray matter localization only"
    )
    parser.add_argument(
        "--only_wm", action="store_true", help="use white matter localization only"
    )

    args = parser.parse_args()
    subj = args.subj
    cluster = args.cluster

    n_parcs = int(args.parcs)
    n_networks = int(args.networks)
    dist = int(args.dist)
    max_length = int(args.max_length)
    only_gm = args.only_gm
    only_wm = args.only_wm

    s = Subject(
        subj, dist=dist, n_parcs=n_parcs, n_networks=n_networks, in_progress=True
    )

    cluster_hemi = s.cluster_hemis[cluster]

    source_parc = (
        s.fetch_normalized_parc2prop_df(cluster, only_gm=only_gm, only_wm=only_wm)
        .idxmax()
        .iloc[0]
    )
    source_parc_nodes = s.parc2node_dict[source_parc]
    source_hemi = get_parcel_hemi(source_parc, n_parcs)

    minBL, maxBL, minGeo, maxGeo = s.fetch_minmax_distances()

    source_minGeo = minGeo[source_parc_nodes, :]
    source_maxGeo = maxGeo[source_parc_nodes, :]

    parc_minGeo, parc_maxGeo = convert_geo_arrays(s, minGeo, maxGeo)

    out_file = s.dirs["source_loc"] / f"lead_velocities_cluster{cluster}.csv"

    if out_file.exists():
        print(
            Colors.YELLOW,
            f"++ Skipping subject {subj} - cluster #{cluster} ++",
            Colors.END,
        )
    else:
        print(
            Colors.PURPLE,
            f"++ Retrieving electrode classifications for subject {subj} - cluster #{cluster} ++",
            Colors.END,
        )

    seqs, delays = s.fetch_sequences(cluster=cluster)
    source_seq_idxs = s.compute_localizing_seq_idxs(
        cluster=cluster, source=source_parc, only_gm=only_gm, only_wm=only_wm
    )

    # get start time for estimating time left to completion
    start_time = time()

    dict_for_df = {}

    # iterate through sequences
    for iter, i in enumerate(source_seq_idxs):

        # print progress report every 50 sequences
        if ((iter % 100) == 0) & (iter != 0):
            print_progress_update(iter, start_time, source_seq_idxs.size)

        # list type
        elecs = [elec for elec in seqs[i, :] if elec != "nan"]
        parcs = [s.elec2parc_dict[elec] for elec in elecs]
        parc_idxs = [[parc - 1 for parc in parc_lst] for parc_lst in parcs]

        # np array
        lags = delays[i, 1 : len(elecs)]
        elec_idxs = np.array([s.elec2index_dict[elec] for elec in elecs])
        hemis = np.array([s.elec2hemi_dict[elec] for elec in elecs])

        min_gm_vel, max_gm_vel = lead_gm_velocities(
            Subj=s,
            source=source_parc,
            elec_idxs=elec_idxs,
            parc_idxs=parc_idxs,
            lags=lags,
            hemis=hemis,
            cluster_hemi=source_hemi,
            minGeo=minGeo,
            maxGeo=maxGeo,
            minBL=minBL,
            maxBL=maxBL,
        )

        min_wm_vel, max_wm_vel = lead_wm_velocities(
            source=source_parc,
            elec_idxs=elec_idxs,
            parc_idxs=parc_idxs,
            lags=lags,
            parc_minGeo=parc_minGeo,
            parc_maxGeo=parc_maxGeo,
            minBL=minBL,
            maxBL=maxBL,
        )

        dict_for_df.setdefault("seq_idx", []).append(i)
        dict_for_df.setdefault("lead_elec", []).append(elecs[0])
        dict_for_df.setdefault("min_gm_vel", []).append(min_gm_vel)
        dict_for_df.setdefault("max_gm_vel", []).append(max_gm_vel)
        dict_for_df.setdefault("min_wm_vel", []).append(min_wm_vel)
        dict_for_df.setdefault("max_wm_vel", []).append(max_wm_vel)

    df = pd.DataFrame.from_dict(dict_for_df)

    df.to_csv(
        s.dirs["source_loc"] / f"lead_velocities_cluster{cluster}.csv", index=False
    )
