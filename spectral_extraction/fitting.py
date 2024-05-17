from .function_maker import create_multigaussian_model, create_multimoffat_model
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import cpu_count
from parallelbar import progress_imap, progress_map, progress_imapu
from parallelbar.tools import cpu_bench, fibonacci
import numpy as np
from copy import deepcopy

def make_fit(ydata,num_source=2,initial_center=None,initial_separation=None
                 ,bound_sigma=None,fix_sep=None,fix_height=None,custom_expr=None
                 ,weights=None,param_limit=None,param_fix=None,param_value=None,verbose=False,\
                distribution="gaussian"):
    """ydata:1d array
    num_source: set two 2 because is the most usual case
    initial_center: float value to set the center of the fit
    initial_separatio: list like fulled with posite or negative values depending of the separation with the center
    bound_sigma: list fulled with the number of components of the fit that wants to be bounded to the central PSF
    fix_sep: list defined to fix the values in initial_sep
    distribution: str distribution to perform a fit only available gaussian or moffat
    custom_expr: custom expr. lmfit like avalaible just read this https://lmfit.github.io/lmfit-py/index.html to get how to add expr."""
    
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
    if n_cpu==None:
        n_cpu = cpu_count()
    if "distribution" not in kwargs.keys():
        kwargs["distribution"]= "gaussian"
    image_copy = deepcopy(image)
    mask =None
    if isinstance(pixel_limit,list) and len(pixel_limit)==2:
        print(pixel_limit)
        pixel_limit = [pixel_limit[0],pixel_limit[1]]
    else:
        pixel_limit = [0,image.shape[1]]
    if isinstance(mask_list,list) and len(mask_list) ==2:
        mask = np.ones_like(image, dtype=bool)
        mask[:, int(mask_list[0]):int(mask_list[1])] = False  # Mask the first 1000 rows
        image_copy = deepcopy(image_copy*mask)
    parameter_number = 3 if kwargs["distribution"]=="gaussian" else 4
    image_copy = deepcopy(image_copy.T[np.arange(*pixel_limit)])
    normalize_matrix = image_copy.T.max(axis=0)
    x_num = len(image_copy.T)
    normalized_image =  np.nan_to_num(image_copy.T/normalize_matrix,0)
    print(normalized_image.shape)
    global process_pixel
    def process_pixel(args):
        n_pixel, pixel = args
        #if i want to add a parameter for the stats part of the matrix always should be and the left of it
        if np.all(pixel== 0):
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

