import warnings

from utils.helpers import *
warnings.filterwarnings("ignore", category=FutureWarning)

from nilearn import plotting
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from utils.subject import Subject

rc('font', family='TimesNewRoman')

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
    sns.set_theme(font_scale=0.8, rc={'xtick.major.size':4,
                                      'xtick.major.width':0.5,
                                      'xtick.bottom':True,
                                      'xtick.minor.size':2,
                                      'xtick.minor.width':0.5,
                                      'xtick.minor.bottom':True})

    # create figure and get axis
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    
    # plot data using seaborn
    sns.barplot(data=parc2prop, x=parc2prop.index, 
                y="propExplanatorySpikes", ax=ax)
    
    # modify figure
    ax.set_title("Proportion of Sequences Explained by Schaefer Parcel\n")
    ax.set_xlabel("Parcel Number")
    ax.set_ylabel("Proportion of Sequences Explained")
    ax.set_ylim((0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(5), )
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.setp(ax.get_xticklabels(), fontsize=6)

    if hemi.upper() == "RH":
        plt.xlim((int(n_parcs/2)-1,n_parcs+1))
    else:
        plt.xlim((-1,int(n_parcs/2)+1))
    
    plt.close(fig)
        
    return fig

def plot_source_localization(Subj, 
                             surfs=['pial','inf_200'], 
                             n_parcs=600, 
                             cmap='Spectral_r',
                             dist=45,
                             only_geo=False,
                             only_wm=False,
                             lead_elec=False
                            ):
    """Run nilearn.plotting.view_surf on a parcel to value df.

    Args:
        Subj (utils.subject.Subject): instance of Subject class
        surfs (list, optional): surfaces of interest. Defaults to ['pial','inf_200'].
        n_parcs (int, optional): number of Schaefer parcels. Defaults to 600.
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
    surf_dir = Subj.dirs['surf']
    general_dir = Subj.dirs['general']

    master_views = {}

    for n_cluster in Subj.valid_clusters:

        if lead_elec:
            parc2prop = Subj.compute_lead_elec_parc2prop_df(cluster=n_cluster)
        else:
            parc2prop = Subj.fetch_normalized_parc2prop_df(cluster=n_cluster,
                                                           dist=dist, 
                                                           only_geo=only_geo,
                                                           only_wm=only_wm
                                                        )

        node2prop_arr, hemi = compute_node2prop_arr(parc2prop, 
                                                    Subj.node2parc_df_dict)

        # load array of sulci contours
        sulc_file = general_dir / f"std.141.{hemi.lower()}.sulc.1D.dset"
        sulc_array = np.loadtxt(sulc_file, comments="#")

        # initialize dictionary to store views
        views = {}
        
        # iterate through surfs
        for surf in surfs:

            surf_file = surf_dir / f"std.141.{hemi.lower()}.{surf}.gii"

            # plot results
            view = plotting.view_surf(str(surf_file), 
                                    node2prop_arr,
                                    cmap=cmap, symmetric_cmap=False,
                                    bg_map=sulc_array, 
                                    threshold=0.001, vmax=1)
            
            # add to dictionary
            views[surf] = view
            
        master_views[n_cluster] = views
        
    return master_views

