from lmfit import Model, Parameters
import numpy as np


def create_multigaussian_model(num_image, ydata, initial_separation=None,initial_center=None):
    """""Recordar que cuando se ingresa una initial_separation, los parametetros del modelo seran
    center1,separation2,sepation...., en cambio cuando no hay sep se denominan
    center1,center2,center3"""
    # Start building the function definition as a string
    func_def = "def multigauss(x, "
    # Add the parameters to the function definition
    params = []
    for i in range(num_image):
        if i == 0 or not initial_separation:
            params.append(f"center_{i+1}")
        params.append(f"height_{i+1}, sigma_{i+1}")
        if initial_separation and i > 0:
            params.append(f"separation_{i+1}")
    func_def += ', '.join(params)
    # Add the function body
    func_def += "):\n    y = np.zeros_like(x)\n"
    for i in range(num_image):
        if i == 0:
            func_def += f"    y += height_{i+1} * np.exp(-(x - center_{i+1})**2 / (2 * sigma_{i+1}**2))\n"
        elif initial_separation:
            func_def += f"    y += height_{i+1} * np.exp(-(x - (center_1 + separation_{i+1}))**2 / (2 * sigma_{i+1}**2))\n"
        else:
            func_def += f"    y += height_{i+1} * np.exp(-(x - center_{i+1})**2 / (2 * sigma_{i+1}**2))\n"
    func_def += "    return y"
    exec(func_def, globals())

    # Create the lmfit model from the dynamically created function
    model = Model(multigauss)
    xdata = np.arange(len(ydata))
    # Create initial parameters based on data
    params = Parameters()
    centers = np.linspace(np.min(xdata), np.max(xdata), num_image+2)[1:-1]  # Evenly spread along x-axis, but not at the very ends
    centers = np.append([np.argmax(ydata)],centers)
    if initial_center:
        centers[0]= initial_center
    height = np.max(ydata)  # Max height of y-data
    sigma = 2  # Std dev of x-data
    for i in range(num_image):
        if initial_center:
            if i == 0 or not initial_separation:
            #params.add(f'center{i+1}', value=centers[i], min=centers[i]-3, max=centers[i]+3)
                params.add(f'center_{i+1}', value=centers[i], min=centers[i]-3, max=centers[i]+3)
            if i > 0 and initial_separation:
                #params.add(f'separation{i+1}', value=np.round(sep[i-1],3), min=sep[i-1]-2, max=sep[i-1]+2)
                params.add(f'separation_{i+1}', value=np.round(initial_separation[i-1],3), min=initial_separation[i-1]-10, max=initial_separation[i-1]+10)
        else:
            if i == 0 or not initial_separation:
                params.add(f'center_{i+1}', value=centers[i], min=np.min(xdata), max=np.max(xdata))
            if i > 0 and initial_separation:
                params.add(f'separation_{i+1}', value=np.round(initial_separation[i-1],3), min=initial_separation[i-1]-5, max=initial_separation[i-1]+5)
        
        
        params.add(f'height_{i+1}', value=height/(i+1), min=0)#np.min(ydata)  # Set the heights of the subsequent Gaussians to be decreasing

        params.add(f'sigma_{i+1}', value=sigma,min=0)
    return model, params,xdata


def create_multimoffat_model(num_image, ydata, initial_separation=None,initial_center=None):
    """""Recordar que cuando se ingresa una initial_separation, los parametetros del modelo seran
    center1,separation2,sepation...., en cambio cuando no hay initial_separation se denominan
    center1,center2,center3"""
    # Start building the function definition as a string
    func_def = "def multimoffat(x, "
    # Add the parameters to the function definition
    params = []
    for i in range(num_image):
        if i == 0 or not initial_separation:
            params.append(f"center_{i+1}")
        params.append(f"height_{i+1}, sigma_{i+1},alpha_{i+1}")
        if initial_separation and i > 0:
            params.append(f"separation_{i+1}")
    func_def += ', '.join(params)
    # Add the function body
    func_def += "):\n    y = np.zeros_like(x)\n"
    for i in range(num_image):
        if i == 0:
            func_def += f"    y += height_{i+1} * (1 + (x - center_{i+1})**2 / (sigma_{i+1}**2))**-alpha_{i+1}\n"
        elif initial_separation:
            func_def += f"    y += height_{i+1} * (1 + (x - (center_1 + separation_{i+1}))**2 / (sigma_{i+1}**2))**-alpha_{i+1}\n"
        else:
            func_def += f"    y += height_{i+1} * (1 + (x - center_{i+1})**2 / (sigma_{i+1}**2))**-alpha_{i+1}\n"
    func_def += "    return y"
    exec(func_def, globals())

    # Create the lmfit model from the dynamically created function
    model = Model(multimoffat)
    xdata = np.arange(len(ydata))
    # Create initial parameters based on data
    params = Parameters()
    centers = np.linspace(np.min(xdata), np.max(xdata), num_image+2)[1:-1]  # Evenly spread along x-axis, but not at the very ends
    centers = np.append([np.argmax(ydata)],centers)
    if initial_center:
        centers[0]= initial_center
    height = np.max(ydata)  # Max height of y-data
    sigma = 7  # Std dev of x-data
    alpha = 3
    for i in range(num_image):
        if initial_center:
            if i == 0 or not initial_separation:
            #params.add(f'center{i+1}', value=centers[i], min=centers[i]-3, max=centers[i]+3)
                params.add(f'center_{i+1}', value=centers[i], min=centers[i]-3, max=centers[i]+3)
            if i > 0 and initial_separation:
                #params.add(f'separation{i+1}', value=np.round(sep[i-1],3), min=sep[i-1]-2, max=sep[i-1]+2)
                params.add(f'separation_{i+1}', value=np.round(initial_separation[i-1],3), min=initial_separation[i-1]-10, max=initial_separation[i-1]+10)
        else:
            if i == 0 or not initial_separation:
                params.add(f'center_{i+1}', value=centers[i], min=np.min(xdata), max=np.max(xdata))
            if i > 0 and initial_separation:
                params.add(f'separation_{i+1}', value=np.round(initial_separation[i-1],3), min=initial_separation[i-1]-2, max=initial_separation[i-1]+2)
        
        
        params.add(f'height_{i+1}', value=height/(i+1), min=0)#np.min(ydata)  # Set the heights of the subsequent Gaussians to be decreasing
        params.add(f'alpha_{i+1}', value=alpha,min=0,max=10)
        params.add(f'sigma_{i+1}', value=sigma,min=0,max=10)
    return model, params,xdata
