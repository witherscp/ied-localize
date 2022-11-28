from argparse import ArgumentParser
from time import time
from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

from ied_localize.utils.colors import Colors
from ied_localize.utils.helpers import (
    convert_geo_arrays,
    convert_list_to_dict,
    output_lst_of_lsts,
    output_normalized_counts,
    print_progress_update,
)
from ied_localize.utils.localize import lead_gm, lead_wm
from ied_localize.utils.subject import Subject

if __name__ == "__main__":

    # parse arguments
    purpose = (
        "to localize the putative source of interictal spike sequences "
        "using GM, WM, and spike timings"
    )
    parser = ArgumentParser(description=purpose)
    parser.add_argument("subj", help="subject code")
    parser.add_argument(
        "--only_gm", action="store_true", help="use gray matter localization only"
    )
    parser.add_argument(
        "--only_wm", action="store_true", help="use white matter localization only"
    )
    parser.add_argument(
        "--fixed_gm",
        action="store_true",
        help="fix GM velocity in all directions; defaults to variable velocity",
    )
    parser.add_argument(
        "-p",
        "--parcs",
        default=600,
        help=("Schaefer parcellation; defaults to 600"),
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

    args = parser.parse_args()
    subj = args.subj
    n_parcs = int(args.parcs)
    n_networks = int(args.networks)
    dist = int(args.dist)
    max_length = int(args.max_length)
    only_gm = args.only_gm
    only_wm = args.only_wm
    fixed_gm = args.fixed_gm

    # either run only gm, only white matter, or combination method
    assert not (only_gm and only_wm)

    if only_gm:
        file_str = "_geodesic"
        print_str = "gray matter"
    elif only_wm:
        file_str = "_whiteMatter"
        print_str = "white matter"
    else:
        file_str = ""
        print_str = "combination"

    s = Subject(subj, n_parcs=n_parcs, n_networks=n_networks, in_progress=True)
    odir = s.dirs["source_loc"]
    odir.mkdir(parents=True, exist_ok=True)

    minBL, maxBL, minGeo, maxGeo = s.fetch_minmax_distances()

    parc_minGeo, parc_maxGeo = convert_geo_arrays(s, minGeo, maxGeo)

    # iterate through clusters
    for cluster in range(1, s.num_clusters + 1):

        cluster_hemi = s.cluster_hemis[cluster]

        print(
            Colors.PURPLE,
            (
                f"++ Retrieving {print_str} source parcels for cluster "
                f"#{cluster}/{s.num_clusters} - {cluster_hemi} ++"
            ),
            Colors.END,
        )

        seqs, delays = s.fetch_sequences(cluster=cluster)
        all_source_parcs = []

        # get start time for estimating time left to completion
        start_time = time()
        n_seqs = seqs.shape[0]

        # iterate through sequences
        for i in range(seqs.shape[0]):

            # print progress report every 50 sequences
            if ((i % 100) == 0) & (i != 0):
                print_progress_update(i, start_time, n_seqs)

            # list type
            elecs = [elec for elec in seqs[i, :] if elec != "nan"]
            parcs = [s.elec2parc_dict[elec] for elec in elecs]
            parc_idxs = [[parc - 1 for parc in parc_lst] for parc_lst in parcs]

            # numpy array type
            lags = delays[i, 1 : len(elecs)]
            elec_idxs = np.array([s.elec2index_dict[elec] for elec in elecs])
            hemis = np.array([s.elec2hemi_dict[elec] for elec in elecs])

            if cluster_hemi == "Both":

                seq_sources = []

                # check both hemispheres for sources
                for c_hemi in ("LH", "RH"):
                    gm_sources = lead_gm(
                        s,
                        elec_idxs,
                        parc_idxs,
                        lags,
                        hemis,
                        c_hemi,
                        minGeo,
                        maxGeo,
                        minBL,
                        maxBL,
                        only_gm=only_gm,
                        only_wm=only_wm,
                        fixed_gm=fixed_gm,
                    )
                    seq_sources = seq_sources + gm_sources

                wm_sources = lead_wm(
                    elec_idxs,
                    parc_idxs,
                    lags,
                    parc_minGeo,
                    parc_maxGeo,
                    minBL,
                    maxBL,
                    only_gm=only_gm,
                    only_wm=only_wm,
                    fixed_gm=fixed_gm,
                )
                seq_sources = seq_sources + wm_sources

            else:

                gm_sources = lead_gm(
                    s,
                    elec_idxs,
                    parc_idxs,
                    lags,
                    hemis,
                    cluster_hemi,
                    minGeo,
                    maxGeo,
                    minBL,
                    maxBL,
                    only_gm=only_gm,
                    only_wm=only_wm,
                    fixed_gm=fixed_gm,
                )

                wm_sources = lead_wm(
                    elec_idxs,
                    parc_idxs,
                    lags,
                    parc_minGeo,
                    parc_maxGeo,
                    minBL,
                    maxBL,
                    only_gm=only_gm,
                    only_wm=only_wm,
                    fixed_gm=fixed_gm,
                )

                seq_sources = gm_sources + wm_sources

            all_source_parcs.append(list(set(seq_sources)))

        # output source parcels
        opath = odir / (
            f"{cluster_hemi}{file_str}_sourceParcels_within{s.dist}"
            f"_max{s.seq_len}_cluster{cluster}.csv"
        )
        out_array = output_lst_of_lsts(all_source_parcs)
        np.savetxt(opath, X=out_array, delimiter=",", fmt="%f")
        print(Colors.GREEN, f"Source parcel file created at {opath}", Colors.END)

        # output normalized counts
        opath = odir / (
            f"{cluster_hemi}{file_str}_normalizedCounts_within"
            f"{s.dist}_max{s.seq_len}_cluster{cluster}.csv"
        )
        sources_dict = convert_list_to_dict(all_source_parcs)
        output_df = output_normalized_counts(seqs.shape[0], sources_dict, s.parcs)
        output_df.to_csv(opath)
        print(Colors.GREEN, f"Normalized counts file created at {opath}", Colors.END)
