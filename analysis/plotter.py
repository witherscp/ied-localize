import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from nilearn import plotting
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

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
        matplotlib.Figure: histogram with results; can be saved out using
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

    if hemi == "RH":
        plt.xlim((int(n_parcs/2)-1,n_parcs+1))
    else:
        plt.xlim((-1,int(n_parcs/2)+1))
    
    plt.close(fig)
        
    return fig
