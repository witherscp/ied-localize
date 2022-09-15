from os import path
import shlex
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from nilearn import plotting
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from ied_localize.utils.constants import N_NODES
from ied_localize.utils.helpers import reorient_coord, get_parcel_hemi
from ied_localize.utils.subject import Subject

rc("font", family="TimesNewRoman")


def compute_node2prop_arr(parc2prop_df, parc2node_dict, n_parcs, hemi=None):
    """Return an array of proportions explained by individual nodes for
    plotting purposes.

    Args:
        parc2prop_df (pd.DataFrame): parcel to proportion dataframe
        parc2node_dict (dict): keys = parcel number,
            values = list of nodes in given parcel
        n_parcs (int): number of parcels (use s.parcs)
        hemi (str, optional): hemisphere; if set to None, will choose the
            hemisphere of maximal proportion. Defaults to None.

    Returns:
        tuple: np.array of proportions at every node, hemisphere string
    """

    if hemi == None:
        # find hemisphere of maximal parcel
        if parc2prop_df["propExplanatorySpikes"].idxmax() < (len(parc2prop_df) // 2):
            hemi = "LH"
        else:
            hemi = "RH"

    assert hemi in ["LH", "RH"]

    if hemi == "LH":
        parc_range = range(1, (n_parcs // 2) + 1)
    elif hemi == "RH":
        parc_range = range((n_parcs // 2) + 1, n_parcs + 1)

    # initialize empty proportion array
    node2prop_arr = np.zeros(N_NODES)

    # update node2parc proportion at each parcel
    for parc in parc_range:

        # get proportion for particular parcel
        parc_slice = parc2prop_df[parc2prop_df.index == parc]
        proportion = parc_slice["propExplanatorySpikes"].iloc[0]

        # set all nodes of particular parcel to correct proportion
        node2prop_arr[parc2node_dict[parc]] = proportion

    return node2prop_arr, hemi.upper()


def plot_prop_explained_histogram(parc2prop, hemi, n_parcs=600):
    """Return a Figure object displaying the proportion of sequences explained
    by each parcel, in histogram format.

    Args:
        parc2prop (pd.DataFrame): proportion of sequences explained for each
            parcel
        hemi (str): hemisphere; "LH" or "RH"
        n_parcs (int, optional): number of parcels. Defaults to 600.

    Returns:
        matplotlib.Figure: histogram with results; figure can be saved using
            fig.savefig("file_name.png", bbox_inches="tight", dpi=300)
    """

    # change plot style
    sns.set_theme(
        font_scale=0.8,
        rc={
            "xtick.major.size": 4,
            "xtick.major.width": 0.5,
            "xtick.bottom": True,
            "xtick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "xtick.minor.bottom": True,
        },
    )

    # create figure and get axis
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # plot data using seaborn
    sns.barplot(data=parc2prop, x=parc2prop.index, y="propExplanatorySpikes", ax=ax)

    # modify figure
    ax.set_title("Proportion of Sequences Explained by Schaefer Parcel\n")
    ax.set_xlabel("Parcel Number")
    ax.set_ylabel("Proportion of Sequences Explained")
    ax.set_ylim((0, 1))
    ax.xaxis.set_major_locator(
        MultipleLocator(5),
    )
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.setp(ax.get_xticklabels(), fontsize=6)

    if hemi.upper() == "RH":
        plt.xlim((int(n_parcs / 2) - 1, n_parcs + 1))
    else:
        plt.xlim((-1, int(n_parcs / 2) + 1))

    plt.close(fig)

    return fig


def plot_source_localization(
    Subj,
    surfs=["pial", "inf_200"],
    cmap="Spectral_r",
    dist=45,
    only_geo=False,
    only_wm=False,
    lead_elec=False,
):
    """Run nilearn.plotting.view_surf on a parcel to value df.

    Args:
        Subj (utils.subject.Subject): instance of Subject class
        surfs (list, optional): surfaces of interest. Defaults to ['pial','inf_200'].
        cmap (str, optional): colormap used to display results. Defaults to 'Spectral_r'.
        dist (int, optional): Geodesic search distance in mm. Defaults to
            45.
        only_geo (bool, optional): Use geodesic only method. Defaults to
            False.
        only_wm (bool, optional): Use white matter only method. Defaults to
            False.
        lead_elec (bool, optional): Plot lead electrode parcels instead of
            localization results. Defaults to False.

    Returns:
        dict: dictionary with keys = surf names and values = plotting.view_surf
            views; surface can be saved using views['view'].save_as_html(out_path)
            or opened in a browser using views['view'].open_in_browser()
    """

    # check arguments
    assert type(surfs) == list
    assert cmap in plt.colormaps()
    assert isinstance(Subj, Subject)
    assert (only_geo + only_wm + lead_elec) <= 1

    # load directory names
    surf_dir = Subj.dirs["surf"]
    general_dir = Subj.dirs["general"]

    master_views = {}

    for n_cluster in Subj.valid_clusters:

        if lead_elec:
            parc2prop = Subj.compute_lead_elec_parc2prop_df(cluster=n_cluster)
        else:
            parc2prop = Subj.fetch_normalized_parc2prop_df(
                cluster=n_cluster, dist=dist, only_geo=only_geo, only_wm=only_wm
            )

        node2prop_arr, hemi = compute_node2prop_arr(
            parc2prop, Subj.parc2node_dict, Subj.parcs
        )

        # load array of sulci contours
        sulc_file = general_dir / f"std.141.{hemi.lower()}.sulc.1D.dset"
        sulc_array = np.loadtxt(sulc_file, comments="#")

        # initialize dictionary to store views
        views = {}

        # iterate through surfs
        for surf in surfs:

            surf_file = surf_dir / f"std.141.{hemi.lower()}.{surf}.gii"

            # plot results
            view = plotting.view_surf(
                str(surf_file),
                node2prop_arr,
                cmap=cmap,
                symmetric_cmap=False,
                bg_map=sulc_array,
                threshold=0.001,
                vmax=1,
            )

            # add to dictionary
            views[surf] = view

        master_views[n_cluster] = views

    return master_views


def barplot_annotate_brackets(
    num1, num2, data, center, height, yerr=None, dh=0.05, barh=0.05, fs=16, maxasterix=3
):
    """
    Annotate barplot with p-values.

    Args:
        num1: number of left bar to put bracket over
        num2: number of right bar to put bracket over
        data: string to write or number for generating asterixes
        center: centers of all bars (like plt.bar() input)
        height: heights of all bars (like plt.bar() input)
        yerr: yerrs of all bars (like plt.bar() input)
        dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        barh: bar height in axes coordinates (0 to 1)
        fs: font size
        maxasterix: maximum number of asterixes to write (for very small
            p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.01
        # *** is p < 0.001
        # etc.
        text = ""
        p = 0.05

        while data < p:
            text += "*"
            if p == 0.05:
                p = 0.01
            elif p == 0.01:
                p = 0.001

            if maxasterix and (len(text) == maxasterix):
                break

        if len(text) == 0:
            text = "n. s."

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c="black")

    kwargs = dict(ha="center", va="bottom")
    if fs is not None:
        kwargs["fontsize"] = fs

    plt.text(*mid, text, **kwargs)


def get_electrode_node_arr(
    Subj, elecs=[], lags=[], func=(lambda x: ((100 - x) / 100 * 2))
):
    """Create array that can be saved out as .node file.

    Args:
        Subj (Subject instance): an instance of Subject
        elecs (list, optional): electrodes to plot. Defaults to [].
        lags (list, optional): lag times to influence size and color of
            electrodes. Defaults to [].
        func (func, optional): function to convert lag time to radius.
            Defaults to (lambda x: ((100 - x)/100 * 2)).

    Returns:
        np.array: array with shape (n_elecs, 5); first 3 columns are x,y,z
            coordinates; last 2 columns are color intensity and radius,
            respectively
    """

    if len(elecs) > 0:
        is_sequence = True
    else:
        is_sequence = False

    rai_coords_file = Subj.dirs["align_elec_alt"] / "electrodes.COORDS.ALT.RAI.1D"
    all_rai_coords = np.loadtxt(rai_coords_file)[:, :-1]

    if is_sequence:
        elec_idxs = np.array([Subj.elec2index_dict[elec] for elec in elecs])
        rai_coords = all_rai_coords[elec_idxs, :]
    else:
        # get all electrodes in gray matter
        gm_elec_file = Subj.dirs["align_elec_alt"] / "electrodes.LABELS.GM.1D"
        gm_elec_nums = np.loadtxt(gm_elec_file, usecols=1, dtype=int)[1:] - 1
        rai_coords = all_rai_coords[gm_elec_nums]

    # reorient to LPI orientation
    lpi_coords = reorient_coord(rai_coords, "RAI", "LPI")

    # fill intensity and radius parameters to new columns
    if is_sequence:
        intensity = np.array(lags)[:, np.newaxis] + 1

        # convert radii using provided function; default is range 0 to 2
        # normalized based on lag time
        radii_lst = [func(lag) for lag in lags]
        radii = np.array(radii_lst)[:, np.newaxis]

        new_cols = np.hstack((intensity, radii))
    else:
        new_cols = np.tile(np.array([1, 0.5]), (lpi_coords.shape[0], 1))

    # concatenate to electrode coordinates
    node_arr = np.concatenate([lpi_coords, new_cols], axis=1)

    return node_arr


def array_to_niml(array, odir, fname):
    """Create a niml.dset file at {odir / fname} using array of shape
    (n_nodes,1)

    Args:
        array (np.array): values at each node; must have size n_nodes
        odir (pathlib.PosixPath): out directory
        fname (str): fname (excluding niml.dset)
    """

    # array must contain the same number of values as std.141 mesh
    assert np.size(array) == N_NODES

    # ensure that array is a column vector
    if array.ndim == 1:
        array = array[:, np.newaxis]

    full_path = odir / f"{fname}.niml.dset"

    # if file exists already, overwrite with updated version
    full_path.unlink(missing_ok=True)

    # if directory does not exist, make directory and all parents
    odir.mkdir(parents=True, exist_ok=True)

    # set-up temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # save out_1D file
        temp_file = path.join(tempdir, f"temp.1D")
        np.savetxt(temp_file, X=array, fmt="%f")

        # run AFNI ConvertDset command to create niml.dset file
        convert_cmd = shlex.split(
            f"ConvertDset -o_niml -input {temp_file} "
            f"-add_node_index -prefix {full_path}"
        )
        subprocess.run(convert_cmd)


def create_topo_arr(Subj, hemi):
    """Create an array of nodes for all triangle vertices in std.141.mesh of
    specified hemisphere.

    Args:
        Subj (Subject): instance of Subject class
        hemi (str): hemisphere of interest

    Returns:
        np.array: array of shape (n_nodes, 3) where each row represents one
            triangle and each column represents a node at one vertex.
    """

    surf_path = Subj.dirs["surf"] / f"std.141.{hemi.lower()}.pial.gii"

    # get names of out files
    coord_path = Subj.dirs["surf"] / "pial.1D.coord"
    topo_path = Subj.dirs["surf"] / "pial.1D.topo"

    # run AFNI ConvertSurface to get topography
    shell_cmd = shlex.split(
        f"ConvertSurface -i_gii {surf_path} -o_vec " f"{coord_path} {topo_path}"
    )
    subprocess.run(shell_cmd, stdout=subprocess.DEVNULL)

    coord_path.unlink()  # delete coordinates file that is unused
    topo_arr = np.loadtxt(topo_path, delimiter=" ", dtype=int)

    return topo_arr


def get_border_nodes(Subj, parcel):
    """Create (n_nodes,1) array with value 1 for every node on the border of
    given parcel.

    Args:
        Subj (Subject): instance of Subject class
        parcel (int): parcel number in range (1,n_parcs+1)

    Returns:
        np.array: array with 1's for every node on border of parcel,
            shape (n_nodes,1). Use array_to_niml() to convert to niml.dset
    """

    hemi = get_parcel_hemi(parcel, Subj.parcs)
    topo_path = Subj.dirs["surf"] / "pial.1D.topo"

    if not topo_path.exists():
        topo_node_arr = create_topo_arr(Subj, hemi)
    else:
        topo_node_arr = np.loadtxt(topo_path, delimiter=" ", dtype=int)

    # convert nodes to parcels
    topo_parc_arr = np.vectorize(Subj.node2parc_hemi_dict[hemi].get)(topo_node_arr)

    # get all triangles that have at least 2 unique parcels
    border_idx = ~np.logical_and(
        topo_parc_arr[:, 0] == topo_parc_arr[:, 1],
        topo_parc_arr[:, 0] == topo_parc_arr[:, 2],
    )

    # get rows of topo_arrs that are on the border
    border_topo_parc_arr = topo_parc_arr[border_idx, :]
    border_topo_node_arr = topo_node_arr[border_idx, :]

    # retrieve all nodes of given parcel in border rows
    nodes = np.unique(border_topo_node_arr[np.isin(border_topo_parc_arr, parcel)])

    nodes_arr = np.zeros(N_NODES)
    nodes_arr[nodes] = 1

    return nodes_arr
