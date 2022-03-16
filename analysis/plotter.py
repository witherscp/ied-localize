import warnings
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

def plot_source_localization(parc2prop, hemi, subj, surfs=['pial','inf_200'],
                             n_parcs=600, cmap='Spectral_r'):
    """Run nilearn.plotting.view_surf on a parcel to value df.

    Args:
        parc2prop (pd.DataFrame): dataframe to plot with parcel numbers as 
            indices and 'propExplanatorySpikes' as the column containing values 
            to plot
        hemi (str): hemisphere of interest
        subj (utils.subject.Subject): instance of Subject class
        surfs (list, optional): surfaces of interest. Defaults to ['pial','inf_200'].
        n_parcs (int, optional): number of Schaefer parcels. Defaults to 600.
        cmap (str, optional): colormap used to display results. Defaults to 'Spectral_r'.

    Returns:
        dict: dictionary with keys = surf names and values = plotting.view_surf
            views; surface can be saved using views['view'].save_as_html(out_path)
            or opened in a browswer using views['view'].open_in_browser()
    """

    # check arguments
    assert type(surfs) == list
    assert cmap in plt.colormaps()
    assert isinstance(subj, Subject)
    
    # load directory names
    surf_dir = subj.dirs['surf']
    general_dir = subj.dirs['general']
    
    # load node2parc_df and set proportion default to zero
    node2parc_df = subj.fetch_node2parc_df()[hemi]
    node2parc_df['proportion'] = 0

    # get list of possible parcel numbers based on n_parcs and hemi
    possible_parcs = range(1,(int(n_parcs/2) + 1))
    if hemi.lower() == "rh":
        possible_parcs = [(n + int(n_parcs/2)) for n in possible_parcs]

    # update node2parc proportion at each parcel
    for parc in possible_parcs:
        if hemi.lower() == "rh":
            mask = (node2parc_df[f'{hemi.upper()}parcel'] == (parc - int(n_parcs/2)))
        else:
            mask = (node2parc_df[f'{hemi.upper()}parcel'] == parc)
        
        # get proportion for particular parcel
        parc_slice = parc2prop[parc2prop.index == parc]
        proportion = parc_slice['propExplanatorySpikes'].iloc[0]
        
        # set all nodes of particular parcel to correct proportion
        node2parc_df.loc[mask,'proportion'] = proportion

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
                                  node2parc_df.proportion.to_numpy(),
                                  cmap=cmap, symmetric_cmap=False,
                                  bg_map=sulc_array, 
                                  threshold=0.001, vmax=1)
        
        # add to dictionary
        views[surf] = view
        
    return views

