from matplotlib.patches import FancyArrowPatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.coordinates import Angle
from regions import PixCoord, RectanglePixelRegion

def inclination_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return np.degrees(np.arctan2(dy, dx))
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
def project_point(x1, y1, x2, y2, x3, y3):
    dx = x2 - x1
    dy = y2 - y1
    dot = dx*(x3-x1) + dy*(y3-y1)
    length_squared = dx**2 + dy**2
    t = dot / length_squared
    xp = x1 + t*dx
    yp = y1 + t*dy
    return xp, yp
#TODO add xlim ylim to the code so can be use with system that have broad separation 
def get_distances(p,Obs=None,Cont=None,plot=None,system_name=None,omit=None,
                  cdd={"UVB":0.0199999999999818,"VIS":0.0199999999999818,"NIR":0.0599999999999454},save=None):    
    
    try:
        images={i:p.loc[(p.index.get_level_values('IS')=="ima") & (p["component"]==i)][["RA","DEC","dDEC","dRA"]].astype(float).values[0] for i in p.loc[p.index.get_level_values('IS')=="ima"]["component"]}
        lens= {i:p.loc[(p.index.get_level_values('IS')=="dif") & (p["component"]==i)][["RA","DEC","dDEC","dRA"]].astype(float).values[0] for i in p.loc[p.index.get_level_values('IS')=="dif"]["component"]}
    except:
        images={i:p[(p['IS']=="ima") & (p['component']==i)][["RA","DEC","dDEC","dRA"]].astype(float).values[0] \
         for i in p.loc[p['IS']=="ima"]["component"]}
        lens={i:p[(p['IS']=="lens") & (p['component']==i)][["RA","DEC","dDEC","dRA"]].astype(float).values[0] \
         for i in p.loc[p['IS']=="lens"]["component"]}
    if omit:
        lens = {i:lens[i] for i in lens.keys() if i != omit}
    Images = pd.DataFrame(images.values(),columns=["RA","DEC","dDEC","dRA"],index=images.keys())
    Lens = pd.DataFrame(lens.values(),columns=["RA","DEC","dDEC","dRA"],index=lens.keys())
    
    P,r_flux=[],[]
    distancias=None
    if Obs:
        p1,p2=Images.loc[Obs][["RA","DEC"]].values
        angle=inclination_angle(*p1,*p2)
        distancias={''.join(Obs):distance(*p1,*p2)}
    if Cont:
        for i in Cont:
            try:
                a,f=Images.loc[i][["RA","DEC"]].values,1
            except:
                a,f=Lens.loc[i][["RA","DEC"]].values,0.483
            P.append(a),r_flux.append(f)
            distancias[''.join(Obs)+f"_{i}"]=distance(*p1, *np.array(project_point(*p1, *p2, *a)))
    if plot:
        fig, ax = plt.subplots(1, 1,figsize=(10, 9))
        for i in Images.index:
            plt.scatter(*Images.loc[i][["RA","DEC"]].values,c="y",label="images",zorder=3)
            plt.text(*Images.loc[i][["RA","DEC"]].values+0.01,i,c="w",size=20)
        for i in Lens.index:
            plt.scatter(*Lens.loc[i][["RA","DEC"]].values,c="r",label="lens",zorder=3)
            plt.text(*Lens.loc[i][["RA","DEC"]].values+0.01,i,c="r",size=20)
         # Add arrows in the right corner
        arrow1 = FancyArrowPatch((4, -4.5), (4,-4), color='black', arrowstyle='->', mutation_scale=20,zorder=10)
        arrow2 = FancyArrowPatch((4, -4.5), (4.5, -4.5), color='black', arrowstyle='->', mutation_scale=20,zorder=10)
        plt.text(4,-4,"North",c="r",size=10)
        plt.text(4.5, -4.5+-0.4,"East",c="r",size=10)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        axis_limit= np.max(np.abs(np.array([*plt.gca().get_xlim(),*plt.gca().get_ylim()])))
    
        w=models.Gaussian2D()
        x = np.linspace(-axis_limit*1.1, axis_limit*1.1,100)
        y = np.linspace(-axis_limit*1.1, axis_limit*1.1,100)
        X, Y = np.meshgrid(x, y)
        sigma=2.53*cdd["UVB"]*10#multiplico x 10 para pasar a Angstroms
        
        if Obs:
            origin = 'lower'
            cmap = plt.cm.binary
            plt.contourf(x, y, w.evaluate(X, Y, 1, *p1, sigma, sigma, 0) + w.evaluate(X, Y, 1, *p2, sigma, sigma, 0) + sum(
                [w.evaluate(X, Y, r_flux[i], *P[i], sigma, sigma, 0) for i in range(len(P))]), cmap=cmap, origin=origin)

            width=11
            height=1.2
            angle=Angle(angle, 'deg')
            reg = RectanglePixelRegion(PixCoord(x=(p1[0]+p2[0])/2, y=(p1[1]+p2[1])/2), width=width,
                                    height=height, angle=angle)
            patch = reg.plot(ax=ax, facecolor='none', edgecolor='g', lw=2,
                            label=f'Slit:\nwidth={width} arcsec\nheight={height} arcsec\nangle={np.round(angle.value,2)}°')
            if Cont:
                for i in range(len(Cont)):
                    print(f"contaminante {Cont[i]}, pos = {P[i]}, proyectado= {project_point(*p1, *p2, *P[i])}")
                    #plt.scatter(*project_point(*p1, *p2, *P[i]),label=f"c_{Cont[i]}",zorder=3)
       
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=15)
        ax.set_ylabel(r"$\delta \gamma$",fontsize=20)
        ax.set_xlabel(r"$\delta \alpha$",fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        #if isinstance(cdd,dict):
        #    distancias = result = {f"{band}_{i}": value / v for band, v in ccd.items() for i, value in distancias.items()}
        if system_name:
            plt.title(system_name)
        ax.invert_xaxis()
        if save:
            plt.savefig(f"images/{save}_quasi_slit.jpg")
        plt.show()
    return distancias