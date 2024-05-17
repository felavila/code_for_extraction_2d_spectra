import astropy
from astropy.io import fits
import numpy as np
from .utils import find_signal


class spectra_2d:
    "little class to handle with 2d image of spectra"
    def __init__(self,object,center_cut = None,size_cut=None,verbose=False):
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
        if center==None:
            center = int(np.nanmedian(np.array([find_signal(i) for i in image.T])))
        if size==None:
            size = 100 # should be fine as initial value
        if verbose:
            print(f"cut center {center} and cut size {size}")
        return image[int(center-size//2):int(center+size//2),:]