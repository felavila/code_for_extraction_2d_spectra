import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from .utils import find_signal


class spectra_2d:
    "Little class to handle 2D image of spectra"
    
    def __init__(self,object,center_cut = None,size_cut=None,verbose=False):
        """
        Initialize the spectra_2d class.

        Parameters:
        ----------
        object : str or array-like
            The input data for the spectra. If a string ending with 'fits', it is treated as a filepath to a FITS file.
            
        center_cut : int or None, optional, default=None
            The center position for cutting the 2D image. If None, the center will be estimated.
            
        size_cut : int or None, optional, default=None
            The size of the cut-out region. If None, a default size of 70 will be used.
            
        verbose : bool, optional, default=False
            If True, print additional information during processing.
        """
        
        self.center_cut = center_cut
        self.size_cut = size_cut
        self.verbose = verbose
        self.header= None 
        self.relevant_keywords_header = None 
        if isinstance(object,str) and  object.endswith("fits"):
            print(object)
            self.fits_image = fits.open(object,center_cut=None,size_cut=None)
            if len(self.fits_image)>1:
                print("Fits image has a len bigger than 1 be aware of in what layer is the image")
            self.header = self.fits_image[0].header
            self.full_data2d = self.fits_image[0].data
            if len(self.fits_image[0].data.shape)==3:
                 self.full_data2d = self.fits_image[0].data[0]
                 self.full_data2d = np.nan_to_num(self.full_data2d,0)
        else:
            raise Exception("#####Check if is a fits file###")
            
        # along the code in 2D array we will asume a [0] axis for spacial, and [1] for dispersion axis 
        
        if self.full_data2d.shape[1] < self.full_data2d.shape[0]:
            self.full_data2d = self.full_data2d.T
        
        data_only_sky = self.full_data2d[int(0.9*self.full_data2d.shape[0]):,:]  # NOTE: here a single one, but better to take as many corners as possible
        # estimate the standard deviation of background noise using MAD (https://en.wikipedia.org/wiki/Median_absolute_deviation)
        self.mad = np.median(np.abs(data_only_sky - np.median(data_only_sky)))
        print("mad =",self.mad)
        self.full_data2d = self.full_data2d #- self.mad
        self.shape_2d_image = self.full_data2d.shape
        self.norm_full_data2d = self.full_data2d.max(axis=0)
        self.data2d = spectra_2d.cut_2d_image(self.full_data2d,center=self.center_cut,size=size_cut,verbose=self.verbose)
        self.stacked_median = np.nanmedian(self.data2d,axis=1)
        if isinstance(self.header,astropy.io.fits.header.Header):
            self.relevant_keywords_header = {i:self.header[i] for i in ["ORIGIN","INSTRUME","OBJECT","NAXIS1","CRVAL1","CD1_1","CUNIT1"] if i in list(self.header.keys()) }
    
    @staticmethod
    def cut_2d_image(image,center=None,size=None,verbose=False):
        """
        Cut a 2D image to the specified region.

        Parameters:
        ----------
        image : array-like
            The input 2D image to be cut.
            
        center : int or None, optional, default=None
            The center position for cutting the 2D image. If None, the center will be estimated.
            
        size : int or None, optional, default=None
            The size of the cut-out region. If None, a default size of 70 will be used.
            
        verbose : bool, optional, default=False
            If True, print additional information during processing.

        Returns:
        -------
        array-like
            The cut-out 2D image.
        """
        
        if center==None:
            center = int(np.nanmedian(np.array([find_signal(i) for i in image.T])))
        if size==None:
            size = 70 # should be fine as initial value
        if verbose:
            print(f"cut center {center} and cut size {size}")
        return image[int(center-size//2):int(center+size//2),:]
    
    def plot_cut_out(self):
        norm_image = self.data2d/self.data2d.max(axis=0)
        fig,axs = plt.subplots(1, 2, figsize=(18, 5))
        # Plot data on the first subplot
        im = axs[0].imshow(norm_image,aspect="auto",vmin=0,vmax=1)
        axs[0].set_title('2d cut')
        axs[0].set_xlabel('X-pixel')
        axs[0].set_ylabel('Y-pixel')
        
        plt.colorbar(im, ax=axs[0], label="normalized intensity")
        #axs[0].legend()

        # Plot data on the second subplot
        axs[1].plot(np.nanmedian(norm_image,axis=1), color='orange')
        axs[1].set_title('stacked median')
        axs[1].set_xlabel('intensity')
        axs[1].set_ylabel('y-pixels')
        #axs[1].invert_xaxis()
        #axs[1].legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()