import numpy as np
import matplotlib.pyplot as plt



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

