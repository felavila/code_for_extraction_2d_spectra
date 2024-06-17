from .function_maker import create_multigaussian_model, create_multimoffat_model
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import cpu_count
from parallelbar import progress_imap, progress_map, progress_imapu
from parallelbar.tools import cpu_bench, fibonacci
import numpy as np
from copy import deepcopy

def make_fit(ydata:np.array,num_source=2,initial_center=None,initial_separation=None
                 ,bound_sigma=None,fix_sep=None,fix_height=None,custom_expr=None
                 ,weights=None,param_limit=None,param_fix=None,param_value=None,verbose=False,\
                distribution="gaussian"):
    """
    Fits a model to the provided data.

    Parameters:
    ----------
    ydata : array-like
        The dependent data (observations or measurements) to be fitted by the model.
        
    num_source : int, optional, default=2
        Specifies the number of sources (or components) to include in the fitting model.
        
    initial_center : float or int or None, optional, default=None
        Initial estimates for the center (mean or peak position) of each source/component.
        
    initial_separation : list or None, optional, default=None
        Initial guess for the separation between the sources/components.
        
    bound_sigma : array-like or None, optional, default=None
        Bounds or constraints on the sigma (standard deviation) starting from second component.
        
    fix_sep : bool or None, optional, default=None
        Whether to fix the separation between sources/components (True/False).
        
    fix_height : bool or None, optional, default=None
        Whether to fix the height (amplitude) of the sources/components (True/False).
        
    custom_expr : str or None, optional, default=None
        Custom expression for the fitting function, allowing for user-defined models https://lmfit.github.io/lmfit-py/index.html.
        
    weights : array-like or None, optional, default=None
        Weights for each data point in ydata, useful for weighted fitting.
        
    param_limit : dict or None, optional, default=None
        Limits on parameters (e.g., min and max values) as a dictionary.
        
    param_fix : dict or None, optional, default=None
        Parameters to be fixed at specific values during fitting.
        
    param_value : dict or None, optional, default=None
        Initial values for parameters, provided as a dictionary.
        
    verbose : bool, optional, default=False
        If True, the function will print detailed information during execution.
        
    distribution : str, optional, default="gaussian"
        The type of distribution to use for fitting. Common choices might be "gaussian", "moffat".

    Returns:
    -------
    result : Fitting result
        The result of the fitting process, typically including fitted parameters and possibly fit statistics.
    
    """   
    if distribution=="moffat":
        model, params,xdata = create_multimoffat_model(num_source,ydata, initial_separation=initial_separation,initial_center=initial_center)
    elif distribution=="gaussian":
        model, params,xdata = create_multigaussian_model(num_source,ydata, initial_separation=initial_separation,initial_center=initial_center)
    else:
        raise ValueError(f"the distribution {distribution} not available, only can be use [gaussian, moffat]")
    if bound_sigma:
        #se podria definir a cual sigma atar
        for i in bound_sigma:
            if i>num_source:
                continue
            params["sigma_"+str(i)].expr=f'sigma_{1}'
    if fix_sep and initial_separation:
        for i in fix_sep:
            if i>num_source:
                continue
            params["separation_"+str(i)].vary = False
    if custom_expr:
        for param, expr in custom_expr.items():
            params[param].expr = expr
    if param_limit:
        for param, limit in param_limit.items():
            params[param].min, params[param].max = limit
    if param_value:
        for param, value in param_value.items():
            params[param].value = value
    if param_fix:
        for param in param_fix:
            params[param].vary = False
    result = model.fit(ydata, params, x=xdata, weights=weights)#,max_nfev=200 
    if verbose:
        print(f"Model parameters {params}")
    return result




def paralel_fit(image,num_source,pixel_limit=None,n_cpu=None,wavelenght=[],mask_list=[],**kwargs):
    """
    Perform parallel fitting on an image using multiple sources.

    Parameters:
    ----------
    image : 2D array-like
        The image data to be fitted.
        
    num_source : int
        Specifies the number of sources (or components) to include in the fitting model.
        
    pixel_limit : list or None, optional, default=None
        Limit on the number of pixels to process. If None, process all pixels.
        
    n_cpu : int or None, optional, default=None
        Number of CPU cores to use for parallel processing. If None, use all available cores.
        
    wavelength : Not available yet 
        
    mask_list : list, optional, default=[]
        List of masks to apply to the image data, used to exclude certain regions from fitting.
        
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the make_fit function. These can include:
        - initial_center
        - initial_separation
        - bound_sigma
        - fix_sep
        - fix_height
        - custom_expr
        - weights
        - param_limit
        - param_fix
        - param_value
        - verbose
        - distribution

    Returns:
    -------
    result : list
        List of fitting results for each processed pixel or region.
    
    """
    
    if n_cpu==None:
        n_cpu = cpu_count()
    if "distribution" not in kwargs.keys():
        kwargs["distribution"]= "gaussian"
    image_copy = deepcopy(image)
    mask =None
    if isinstance(pixel_limit,list) and len(pixel_limit)==2:
        pixel_limit = [pixel_limit[0],pixel_limit[1]]
    else:
        pixel_limit = [0,image.shape[1]]
    if isinstance(mask_list,list):
        mask = np.ones_like(image, dtype=bool)
        for i in mask_list:
            mask[:,range(*i)] = False
        image_copy = deepcopy(image_copy*mask)
    parameter_number = 3 if kwargs["distribution"]=="gaussian" else 4
    image_copy = deepcopy(image_copy.T[np.arange(*pixel_limit)])
    normalize_matrix = np.abs(image_copy.T).max(axis=0)
    x_num = len(image_copy.T)
    normalized_image =  np.nan_to_num(image_copy.T/normalize_matrix,0)
    global process_pixel
    def process_pixel(args):
        n_pixel, pixel = args
        #if i want to add a parameter for the stats part of the matrix always should be and the left of it
        if np.all(pixel== 0) or  np.all(pixel== np.nan):
            return list([np.nan]*parameter_number*num_source)+[1e15]*parameter_number*num_source+ list([np.nan]*parameter_number*num_source)+[1e15,1e15,1e15,1e15,1e15,n_pixel,x_num] 
        fiting =  make_fit(pixel, num_source=num_source,**kwargs)
        if np.all(fiting.covar) == None:
            return list([np.nan]*parameter_number*num_source)+[1e15]*parameter_number*num_source+ list(fiting.values.keys())+[1e15,1e15,1e15,1e15,1e15,n_pixel,x_num] 
        return list(np.array([[value.value,value.stderr] for key,value in fiting.params.items()]).T.reshape(parameter_number*num_source*2,))+ list(fiting.values.keys()) + [fiting.chisqr,fiting.redchi,fiting.aic,fiting.bic,fiting.rsquared,n_pixel,x_num] 
    print(f"The code will be executed in {n_cpu} core using {num_source} sources an a {kwargs['distribution']} distribution")
    args = [(n_pixel, pixel) for n_pixel, pixel in zip(np.arange(*pixel_limit) ,normalized_image.T)]
    normalize_matrix = normalize_matrix[:,np.newaxis]
    full_fit = np.array(progress_map(process_pixel, args, process_timeout=20, n_cpu=n_cpu,need_serialize=False))
    spectral_extraction_results = {"image":image_copy.T,"full_fit":full_fit,"normalize_matrix":normalize_matrix,"num_source":num_source,"distribution":kwargs["distribution"],"mask":mask,"original_image":image}
    return spectral_extraction_results

