import numpy as np
import matplotlib.pyplot as plt
import os 
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


def plot2d_spectra(image,region=None):
    """
    Plots a 2D spectra image with optional region highlighting.

    Parameters:
    ----------
    image : array-like
        The 2D spectra image data to be plotted.
        
    region : float or None, optional, default=None
        A float value representing the region to highlight, as a fraction of the image height. 
        If provided, a horizontal line and shaded area will be added to the plot.

    Returns:
    -------
    None
    """
    image_to_plot = np.nan_to_num((image/image.max(axis=0)),0)
    size = image_to_plot.shape
    fig, ax = plt.subplots(figsize=(20, 9), dpi=80)
    img =ax.imshow(image_to_plot,aspect='auto', origin = 'lower', vmin = 0, vmax = 1)
    if isinstance(region,float):
        threshold = int(region * size[0])
        ax.axhline(threshold, color='green', lw=2, alpha=0.7)
        ax.fill_between(np.arange(size[1]), threshold, size[0],
                        color='green', alpha=0.5, transform=ax.get_yaxis_transform())
    colorbar =fig.colorbar(img, ax=ax, label='Norm spectra')
    colorbar.ax.yaxis.label.set_size(20)
    ax.set_xlabel('dispersion axis', fontsize=20)  # Add label to x axis
    ax.set_ylabel('spacial axis', fontsize=20)  # Add label to y axis
    ax.tick_params(axis=("both"), labelsize=20) 
    ax.set_title(f'2d spectra {image_to_plot.shape[1]} dispersion x {image_to_plot.shape[0]} spacial', fontsize=20)
    ax.set_ylim(0,size[0]-1)
    plt.show()





#maybe pre render a kind of plots 
tableau_colors = list(mcolors.TABLEAU_COLORS.values())
__all__ = ("ploting_result","plot_spectra","plot_three_levels")
module_dir = os.path.dirname(os.path.abspath(__file__))

def plot_spectra(wavelenght,flux,title="?",save=None,show=True,xlim=None,ylim=None):
    plt.figure(figsize=(35, 15))
    line_name,wv = np.loadtxt(os.path.join(module_dir,"tabuled_values/linelist.txt"),dtype="str").T
    zs = 0
    plt.plot(wavelenght,flux,c="k",label=title)
    _,ymax = plt.gca().get_ylim()
    #print(plt.gca().get_xlim())
    plt.xlim(wavelenght[0],wavelenght[-1])
    if xlim:
         plt.xlim(*xlim)
    xmin,xmax=plt.gca().get_xlim()
    if ylim:
         plt.ylim(*ylim)
    ymin,ymax = plt.gca().get_ylim()
    for key,value in zip(line_name,wv):
        value = float(value)
        if xmin<value<xmax:
            if "Fe" in key or "H1" in key or "H9" in key or "H8" in key:
                continue
            #print(key,value)
            plt.axvline(float(value)*(1+zs),c="k",ls="--",alpha=0.2)
            plt.text(float(value)*(1+zs),ymax,key, rotation=90, verticalalignment='bottom', fontsize=20)
    plt.xlabel('Rest Wavelength', fontsize=20)
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.ylabel('Flux', fontsize=20)
    plt.legend(loc='upper center',  prop={'size': 24}, frameon=False, ncol=2)
    if save:
        plt.savefig(save)
     # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_spectra(result,obj=None,bands=None,add_lines=False,xlim=None,ylim=None,save=None,normalize=False):
    xlabel = "Observe wavelength"
    zs = result.zs.values[0]
    plt.figure(figsize=(30,10))
    m,M  = [],[]
    n = ""
    if bands == None:
        bands = result.band.unique()
    for band in bands:
        local_r=result[(result["band"]==band)][[i for i in result.columns if ("flux" in i) or ("wavelength" in i)]]
        if obj==None:
            obj = [i.replace("flux_","") for i in local_r.columns if "flux" in i]
        for i,o in enumerate(obj):
            try:
                flux = local_r[f"flux_{o}"]
                if normalize == True:
                    n = "normalized "
                    if i==0:
                        norm = np.sum(flux)
                    flux = flux/norm

                plt.plot(local_r.wavelength,flux,label=f"{o} {band}")
            except:
                print(f' the objs are {[i.replace("flux_","") for i in local_r.columns if "flux" in i]}')
                plt.close()
                return 
        m.append(local_r.wavelength.values[0]), M.append(local_r.wavelength.values[-1])
    
    if m==[]:
        plt.close()
        return print(f' the bans are {result.band.unique()}')
    
    plt.xlim(np.min(m),np.max(M))
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(n+'Flux', fontsize=20)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if add_lines:
        #maybe pre render a kind of plots 
        import os
        #tableau_colors = list(mcolors.TABLEAU_COLORS.values())
        __all__ = ("ploting_result","plot_spectra","plot_three_levels")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        xmin,xmax=plt.gca().get_xlim()
        _,ymax = plt.gca().get_ylim()
        line_name,wv = np.loadtxt(os.path.join(module_dir,"tabuled_values/linelist.txt"),dtype="str").T
        for key,value in zip(line_name,wv):
            value = float(value)*(1+zs)
            if xmin<value<xmax:
                #remove lines in masked zone
                if "Fe" in key or "H1" in key or "H9" in key or "H8" in key:
                    continue#print(key,value)
                plt.axvline(float(value),c="k",ls="--",alpha=0.2)
                plt.text(float(value),ymax,key, rotation=90, verticalalignment='bottom', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(loc='upper right', prop={'size': 24}, frameon=False, ncol=2)
    plt.tick_params(axis='both', which='major', labelsize=20)
    if save:
        plt.savefig(save)