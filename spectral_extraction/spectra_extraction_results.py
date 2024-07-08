from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from .utils import gaussian,moffat
from copy import deepcopy
import numpy as np
import pandas as pd
import itertools
from scipy.interpolate import interp1d
import astropy.io
from astropy.stats import sigma_clip

class spectral_extraction_results_handler:
    def __init__(self,spectral_extraction_results,conditions={"rsquared":0.7},header=None,names=None):
        """
        Initialize the spectral_extraction_results_handler class.

        Parameters:
        ----------
        spectral_extraction_results : dict
            Dictionary containing results from the spectral extraction process.
        
        conditions : dict, optional, default={"rsquared": 0.7}
            Conditions for cleaning the pandas DataFrame. The default condition is to have rsquared >= 0.7.
        
        header : FITS header or None, optional
            Header information, typically from a FITS file.
        """
        
        self.spectral_extraction_results = spectral_extraction_results
        self.conditions = conditions
        self.image,self.full_fit,self.normalization_array,self.source_number,self.distribution,self.mask,self.original_image = list(self.spectral_extraction_results.values())
        self.distribution_function = gaussian if self.distribution=="gaussian" else moffat
        self.parameter_number = 3 if self.distribution=="gaussian" else 4
        self.columns_distribtuion = ["center","height","sigma"] if self.distribution=="gaussian" else ["center","height","alpha","sigma"]       
        self.pandas_results,self.image_model = self.array_to_pandas()
        self.cleaned_panda = spectral_extraction_results_handler.clean_pandas(deepcopy(self.pandas_results),conditions=self.conditions)
        self.residuals = self.image - self.image_model
        self.spectras1d_raw = {i:spectral_extraction_results_handler.interpolate_1d(self.pandas_results[i].values) for i in self.cleaned_panda.columns if "flux" in i}
        self.spectras1d = {i:spectral_extraction_results_handler.interpolate_1d(self.cleaned_panda[i].values) for i in self.cleaned_panda.columns if "flux" in i}
        self.cleaned_panda[[i for i in self.spectras1d.keys()]] = np.array(list(self.spectras1d.values())).T
        self.names = np.arange(1,self.source_number+1).astype("str")
        if header:
            self.header=header
            if isinstance(self.header,astropy.io.fits.header.Header):
                self.relevant_keywords_header = {i:self.header[i] for i in ["ORIGIN","INSTRUME","OBJECT","NAXIS1","CRVAL1","CD1_1","CUNIT1"] if i in list(self.header.keys()) }
                to_angs = 1
                if "CUNIT1" in self.relevant_keywords_header.keys():
                    if self.relevant_keywords_header["CUNIT1"]=="nm":
                        to_angs=10
                self.wavelength =  np.array([(self.relevant_keywords_header["CRVAL1"]+i*self.relevant_keywords_header["CD1_1"])*to_angs for i in self.cleaned_panda["n_pixel"].values])
                self.cleaned_panda["wavelength"] = self.wavelength
        else:
            print("your results will not have wavelength try to define a header")
        if names:
            self.set_names(names)

    def set_names(self,names):
        if len(names)==self.cleaned_panda.source_number.values[0]:
            self.renames = {i:i.replace(i[-1],names[int(i[-1])-1]) for i in self.cleaned_panda.columns if i[-1] in np.arange(1,self.source_number+1).astype("str")}
            self.cleaned_panda = self.cleaned_panda.rename(columns=self.renames)
            self.names = names  
        else:
            print("check well your defined names and your number of objects in the sleet")
    def unset_names(self):
        if self.renames != None:
            self.cleaned_panda = self.cleaned_panda.rename(columns={value:key for key,value in self.renames.items()})
            self.names = np.arange(1,self.source_number+1).astype("str")
        else:
            print("you cant rename")
    def array_to_pandas(self):
        """
        Convert array results to pandas DataFrame.

        Returns:
        -------
        tuple
            A tuple containing the pandas DataFrame of results and the 2D model image.
        """
        separation_as_parameter = False
        columns_model = [i for i in np.unique(deepcopy(self.full_fit[:,2*self.parameter_number*self.source_number:3*(self.parameter_number*self.source_number)]),axis=0) if "nan" not in i][0]
        columns_flux =[f"flux_{n}" for n in range(1,self.source_number+1)]
        columns_stats = ["chisqr","redchi","aic","bic","rsquared","n_pixel","x_num"]
        columns_model_init =["value_norm_"+i if "height" in i else "value_"+i for i in columns_model] # if separation exist should be here
        columns_std_init = ["std_norm_"+i if "height" in i else "std_"+i for i in columns_model]
        if any([bool("separation" in i) for i in columns_model]): 
            separation_as_parameter = True
            columns_model_final = ["value_"+i.replace("separation","center") if "separation" in i else "value_"+i for i in columns_model]
            columns_std_final = ["std_"+i.replace("separation","center") if "separation" in i else "std_"+i for i in columns_model]
        else:
            columns_model_final = ["value_"+i for i in columns_model]
            columns_std_final = ["std_"+i for i in columns_model]
        panda_columns = columns_flux+columns_model_final+columns_std_final+columns_stats+columns_model_init#+columns_std_init 
        model_parameters = deepcopy(self.full_fit[:,:self.parameter_number*self.source_number]).astype(float)
        std = deepcopy(self.full_fit[:,self.parameter_number*self.source_number:2*(self.parameter_number*self.source_number)]).astype(float)
        stats = deepcopy(self.full_fit[:,3*(self.parameter_number*self.source_number):]).astype(float)
        pre_values = deepcopy(model_parameters)
         # ############################################
        model_parameters[:,[i for i in range(1,self.parameter_number*self.source_number,self.parameter_number)]] = model_parameters[:,[i for i in range(1,self.parameter_number*self.source_number,self.parameter_number)]] * self.normalization_array
        if separation_as_parameter:
            model_parameters[:,[i for i in range(self.parameter_number,self.parameter_number*self.source_number,self.parameter_number)]] = (model_parameters[:,[i for i in range(self.parameter_number,self.parameter_number*self.source_number,self.parameter_number)]].T + model_parameters[:,0]).T
        multiple_dist = np.array([self.distribution_function(np.arange(stats[0][-1])[:, np.newaxis],*i.T) for i in model_parameters.reshape(len(model_parameters),self.source_number,self.parameter_number)])
        fluxes = multiple_dist.sum(axis=1)
        image_2d_model = multiple_dist.T.sum(axis=0)
        sumary_results = pd.DataFrame(np.hstack((fluxes,model_parameters,std,stats,pre_values)),columns=panda_columns)#,columns=panda_columns)
        sumary_results["distribution"] = [self.distribution] * len(sumary_results)
        sumary_results["source_number"] = [self.source_number] * len(sumary_results)
        return sumary_results.loc[:,~sumary_results.columns.duplicated()].copy(),image_2d_model#values,std,fluxes,stats,pre_v
    @staticmethod
    def interpolate_1d(flux):
        """
        Interpolate 1D flux data.

        Parameters:
        ----------
        flux : array-like
            1D array of flux values to be interpolated.

        Returns:
        -------
        array-like
            Interpolated 1D flux data.
        Notes:
        ------ 
            This require more analize given the posibility of what happend when we are working with a cuted 2d image,add table with parameters
        
        """
        clip_,lower,upper = sigma_clip(flux,sigma=10, cenfunc='median', return_bounds=True)
        flux[flux>upper] = np.nan
        if np.isnan(flux[0]):
            flux[0] = np.nanmedian(flux)
        if np.isnan(flux[-1]):
            flux[-1] = np.nanmedian(flux)
        x = np.arange(len(flux))
        mask_nan = np.isnan(flux)
        flux_1_no_nan = flux[~mask_nan]
        x_non_nan = x[~mask_nan]
        function_to_interpolate = interp1d(x_non_nan, flux_1_no_nan, kind='linear')
        return function_to_interpolate(x)
    @staticmethod
    def clean_pandas(pandas_no_clean,conditions={"min":{"rsquared":0.7}}):
        """
        Clean pandas DataFrame based on conditions.

        Parameters:
        ----------
        pandas_no_clean : DataFrame
            The pandas DataFrame to be cleaned.
        
        conditions : dict, optional, default={"min": {"rsquared": 0.7}}
            Conditions for cleaning the DataFrame. The default condition is to have rsquared >= 0.7.

        Returns:
        -------
        DataFrame
            Cleaned pandas DataFrame.
        Notes:
        ------ 
            Here will be a good idea add the posibility of decide over what "pandas" plot the column so is more clear where could be the problem 
        """
      
        for super_key,super_values in conditions.items():
            if super_key=="min":
                for key,values in super_values.items():
                    indices = pandas_no_clean.index[pandas_no_clean[key] < values]
                    flux_columns = [col for col in pandas_no_clean.columns if "flux" in col]
                    pandas_no_clean.loc[indices, flux_columns] = np.nan
            elif super_key=="max":
                for key,values in super_values.items():
                    indices = pandas_no_clean.index[pandas_no_clean[key] > values]
                    flux_columns = [col for col in pandas_no_clean.columns if "flux" in col]
                    pandas_no_clean.loc[indices, flux_columns] = np.nan      
        return pandas_no_clean
    
    def plot_2d_image_residuals(self,save=None):
        model_result = {"original_image":self.image/self.image.max(axis=0),"model_image":self.image_model/self.image_model.max(axis=0),"residuals original-model":self.residuals/np.max(np.abs(self.residuals),axis=0)}
        fig, axes = plt.subplots(1,3, figsize=(50, 10))
        for ax, (key, spectra2d) in zip(axes, model_result.items()):
            vmin,vmax,label=0,1,"normalize"
            ax.set_title(key, fontsize=30)
            if key=="residuals original-model":
                vmin,vmax,label=-1,1,"(image-model)/max"
            im = ax.imshow(spectra2d,aspect="auto",vmin=vmin,vmax=vmax, cmap='coolwarm')
            cbar = plt.colorbar(im, ax=ax, shrink=1)
            ax.set_xlabel("Pixel",fontsize=20)
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label(label, fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            #fig.colorbar(im, ax=ax, shrink=1,label=label, fontsize=14)

        plt.show()
    def plot_1d(self,n_pixel,save=None):
        parameters=self.cleaned_panda[self.cleaned_panda['n_pixel'].isin([n_pixel])][[f"value_{c}_{n}"  for n in self.names for c in  self.columns_distribtuion]]
        pixel_1d =self.image.T[n_pixel]
        x = np.linspace(0,len(pixel_1d),100)
        plt.plot(self.image.T[n_pixel],label="raw data")
        separated_sources = np.array([self.distribution_function(x,*i) for i in parameters.values[0].reshape(self.source_number,self.parameter_number)])
        plt.plot(x,np.sum(separated_sources,axis=0),color="k",label="added models")
        [plt.plot(x ,i, linestyle="--", linewidth=1.5,label=f"source {self.names[n]}") for n,i in  enumerate(separated_sources)]
        
        plt.plot(np.arange(len(pixel_1d)),pixel_1d-np.sum(np.array([self.distribution_function(np.arange(len(pixel_1d)),*i) for i in parameters.values[0].reshape(self.source_number,self.parameter_number)]),axis=0),label="residuals",alpha=0.5)
        plt.title(f"pixel {n_pixel}")
        plt.legend()
        plt.show()
    def plot_column(self,column_name):
       
        
        try:
            column = self.cleaned_panda[column_name].values
            print(f"mean value for {column_name} if {np.nanmedian(column)}")
            plt.plot(column)
            plt.axhline(np.nanmedian(column),zorder=10,c="k", linewidth=1.5)
            plt.title(f"column: {np.nanmedian(column)}")
            plt.show()
            return np.nanmedian(column)
        except:
            print(f"{column_name} is not a avalaible column try \n {list(self.cleaned_panda.columns)}")
    def plot_spectra(self,obj=None):
        wavelength = None
        try:
            wavelength = self.wavelength
            xlabel="wavelength"
        except:
            print("not wavelength in the class")
            wavelength = np.arange(len(self.clean_pandas))
            xlabel="pixel"
        plt.figure(figsize=(20,10))
        [plt.plot(wavelength,flux,label=key,linewidth=0.5) for key,flux in self.spectras1d.items()]
        plt.xlabel(xlabel)
        plt.ylabel("flux")
        plt.legend()