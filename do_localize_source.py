from argparse import ArgumentParser
from warnings import filterwarnings

from utils.constants import INTRAPARCEL_DISTS

filterwarnings("ignore", category=RuntimeWarning)

from ied_repo.analysis import *
from ied_repo.utils import print_progress_update, lead_geodesic, lead_wm

if __name__ == "__main__":

    # parse arguments

    purpose = (
        "to localize the putative source of interictal spike sequences"
        "using GM, WM, and spike timings"
    )
    parser = ArgumentParser(description=purpose)
    parser.add_argument("subj", help="subject code")
    parser.add_argument(
        "--only_geo", action="store_true", help="use geodesic localization only"
    )
    parser.add_argument(
        "--only_wm", action="store_true", help="use white matter localization only"
    )
    parser.add_argument(
        "--fixed_geo",
        action="store_true",
        help="fix geodesic velocity in all directions",
    )
    parser.add_argument(
        "-p",
        "--parcs",
        default=600,
        help=("number of parcels to use to generate FC matrix;" " defaults to 600"),
    )
    parser.add_argument(
        "-n", "--networks", default=17, help="Yeo network {7 or 17}; defaults to 17"
    )
    parser.add_argument(
        "--dist", default=45, help="geodesic max search distance; defaults to 45"
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
    only_geo = args.only_geo
    only_wm = args.only_wm
    fixed_geo = args.fixed_geo

    # either run only geodesic, only white matter, or combination method
    assert not (only_geo and only_wm)

    if only_geo:
        file_str = "_geodesic"
        print_str = "geodesic"
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
    maxBL[np.diag_indices_from(maxBL)] = INTRAPARCEL_DISTS[n_parcs]

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
                    geo_sources = lead_geodesic(
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
                        only_geo=only_geo,
                        only_wm=only_wm,
                        fixed_geo=fixed_geo,
                    )
                    seq_sources = seq_sources + geo_sources

                wm_sources = lead_wm(
                    elec_idxs,
                    parc_idxs,
                    lags,
                    parc_minGeo,
                    parc_maxGeo,
                    minBL,
                    maxBL,
                    only_geo=only_geo,
                    only_wm=only_wm,
                    fixed_geo=fixed_geo,
                )
                seq_sources = seq_sources + wm_sources

            else:

                geo_sources = lead_geodesic(
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
                    only_geo=only_geo,
                    only_wm=only_wm,
                    fixed_geo=fixed_geo,
                )

                wm_sources = lead_wm(
                    elec_idxs,
                    parc_idxs,
                    lags,
                    parc_minGeo,
                    parc_maxGeo,
                    minBL,
                    maxBL,
                    only_geo=only_geo,
                    only_wm=only_wm,
                    fixed_geo=fixed_geo,
                )

                seq_sources = geo_sources + wm_sources

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
