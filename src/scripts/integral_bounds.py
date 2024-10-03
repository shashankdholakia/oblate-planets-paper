from eclipsoid.numpy_src import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Arc

import paths

def draw_oblate(b, xo, yo, ro, theta):
    # Set up the figure
    #theta in degrees
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(min(-1.2, xo - ro - 0.2), max(1.05, xo + ro + 0.2));
    ax.set_ylim(min(-1.2, yo - ro - 0.2), max(1.05, yo + ro + 0.2));
    ax.set_aspect(1);
    ax.axis('off');

    # Draw the star
    occulted = Circle((0, 0), 1, fill=False, color='k')
    occulted_fill = Circle((0, 0), 1, fill=True, color='k',alpha=0.03)
    ax.add_artist(occulted)
    ax.add_artist(occulted_fill)
    
    #Draw the line along the star's equator to reference xi and theta:
    ax.plot([0,1],[0,0],'k:',alpha=0.4,lw=1.)
    
    # Draw the planet, multiply semi major and semi minor axes by 2 to get major and minor axes
    occulter = Ellipse((xo, yo), ro*2,ro*b*2, fill=False, color='r')
    occulter_fill = Ellipse((xo, yo), ro*2,ro*b*2, fill=True, color='r',alpha=0.03)
    ax.add_artist(occulter_fill)
    ax.add_artist(occulter)
    
    ax.annotate("", xy=(xo, yo+ro), xytext=(xo, yo+ro*b), arrowprops=dict(arrowstyle="<->", color='r', alpha=0.7, shrinkA=1, shrinkB=1, linewidth=0.7))
    ax.annotate(r"$f r_{eq}$", xy=(xo, yo+ro*b/2+ro/2), xytext=(3, -1), textcoords="offset points", ha="left", va="center", color='r', fontsize=12, alpha=0.7)
    
    ax.annotate("", xy=(xo, yo), xytext=(xo-ro, yo), arrowprops=dict(arrowstyle="<->", color='r', alpha=0.7, shrinkA=1, shrinkB=1, linewidth=0.7))
    ax.annotate(r"$r_{eq}$", xy=(xo-ro/2, yo), xytext=(0, -3), textcoords="offset points", ha="center", va="top", color='r', fontsize=12, alpha=0.7)

    ax.annotate(r"$(x_o, y_o)$", xy=(xo, yo), xytext=(0, 6), textcoords="offset points", ha="right", va="bottom", color='r', fontsize=12, alpha=0.7)

    ax.annotate(r"occulter", xy=(xo, yo+ro*b+.19), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=14, color="r")
    ax.annotate(r"(rotated frame)", xy=(xo, yo+ro*b+.19),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="r")
    
    ax.annotate(r"occulted", xy=(0, -1.04), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=14, color="k")

    #dots for center of star and planet
    ax.plot(0, 0, 'ko')
    ax.plot(xo, yo, 'ro')
    ax.annotate(r"$b_o$", xy=(xo/2, yo/2), xytext=(5, 5), textcoords="offset points", ha="left", va="center", color='k', fontsize=14, alpha=0.7)

    
    #dotted line connecting star and planet to reference theta
    ax.plot([0,xo],[0,yo],'k:',alpha=0.4,lw=1., zorder=-1)
    
    #pre rotated reference frame
    prerot_center = (xo*np.cos(np.radians(theta))-yo*np.sin(np.radians(theta)), 
                      xo*np.sin(np.radians(theta))+yo*np.cos(np.radians(theta)))
    prerot = Ellipse(prerot_center, ro*2,ro*b*2, angle=theta, fill=False, color='r', alpha=0.3, zorder=-1)
    ax.plot(prerot_center[0], prerot_center[1], 'ro', alpha=0.3, zorder=1, mew='0', ms=7)
    
    ax.annotate(r"occulter", xy=(prerot_center[0], prerot_center[1]-ro*b-.18), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=14, color="r", alpha=0.3)
    ax.annotate(r"(original frame)", xy=(prerot_center[0], prerot_center[1]-ro*b-.18),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="r", alpha=0.3)
    
    ax.plot([0,prerot_center[0]],[0,prerot_center[1]],'k:',alpha=0.3,lw=1., zorder=-1)
    ax.add_artist(prerot)
    
    rot_angle = Arc((0, 0), 0.15, 0.15, angle=0, theta1=np.degrees(np.arctan2(yo, xo)), theta2=np.degrees(np.arctan2(yo, xo))+theta, color="r", alpha=0.4, ls="-", lw=1)
    ax.add_patch(rot_angle)
    ax.text(-0.15, -0.04, s=r"$\theta$", size=14, color='r',alpha=0.7)
    
    x_real, y_real = intersection_points(b, xo, yo, ro)
    if x_real.shape == (0,): #if there are no intersections
        xi = (0,0) #we want xi to be from 0 to 2 pi
    else: 
        xi = np.sort(np.arctan2(y_real,x_real))

    ax.plot(x_real,y_real, 'ko')

    #if xi0 to xi1 contains the unit vector alpha
    #we DONT want to perform that integral, instead switch to xi1 to xi0 
    if (xi[0]<np.arctan2(-yo,-xo)<xi[1]):
        xi_grid = np.linspace(xi[1],xi[0],1000)
    else:
        xi_grid = np.linspace(xi[0]+2*np.pi,xi[1],1000)
        
    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(xi_grid):
        x[i] = np.cos(v)
        y[i] = np.sin(v)
        
    for i in np.arange(0,len(x),len(x)//10)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="k"),
            size=20,
        )
    #bold the Q integral region
    ax.plot(x, y, color='k', lw=2,zorder=-1); 

    ax.plot([0,np.cos(xi[0])],[0,np.sin(xi[0])], 'k-', alpha=0.7, lw=1.0)
    ax.plot([0,np.cos(xi[1])],[0,np.sin(xi[1])], 'k-', alpha=0.7, lw=1.0)

    #add the circle bounding the planet to help parametrize the angle phi
    anomaly = Circle((xo, yo), ro, fill=False, color='r', alpha=0.3, ls=':')
    ax.add_artist(anomaly)

    #horizontal line along the major axis
    ax.plot([xo-ro,xo+ro],[yo,yo],'r:',alpha=0.3,lw=1.)

    #arctan of y *on the circle circumscribing the occcultor ellipse* to x on the ellipse (drops straight down)
    if x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)<1: #if there are no intersections and the planet is in star
        phi = (0,0) #phi from 0 to 2 pi (2 pi will be added later)
    elif x_real.shape == (0,) and np.sqrt(xo**2 + yo**2)>=1: #if no intersections and planet is outside star
        phi = (0,2*np.pi) #phi from 2pi to 2pi (we don't want to integrate boundary of planet)
    else:
        phi = np.sort(np.arctan2(np.sqrt(ro**2-(x_real-xo)**2),x_real-xo)*np.sign(np.arctan2(y_real-yo,x_real-xo)))

    #plot the phi angle (parametrized like eccentric anomaly)
    ax.plot([xo,xo+ro*np.cos(phi[0])],[yo,yo+ro*np.sin(phi[0])], 'r-', alpha=0.4, lw=1.0)
    ax.plot([xo,xo+ro*np.cos(phi[1])],[yo,yo+ro*np.sin(phi[1])], 'r-', alpha=0.4, lw=1.0)

    #plot the line down to the major axis
    ax.plot([xo+ro*np.cos(phi[0]), xo+ro*np.cos(phi[0])],[yo+ro*np.sin(phi[0]), yo], 'r:', alpha=0.3, lw=1.0)
    ax.plot([xo+ro*np.cos(phi[1]), xo+ro*np.cos(phi[1])],[yo+ro*np.sin(phi[1]), yo], 'r:', alpha=0.3, lw=1.0)

    phi_inters=np.arctan2(-yo,-xo)


    if phi[0] < phi_inters < phi[1]:
        phi_grid = np.linspace(phi[0],phi[1],1000)
    else:
        #reverse the order of integration so it is always performed counterclockwise
        phi_grid = np.linspace(phi[1],2*np.pi+phi[0],1000)

    x = np.zeros(1000)
    y = np.zeros(1000)
    for i, v in enumerate(phi_grid):
        x[i] = ro*np.cos(v) + xo
        y[i] = ro*b*np.sin(v) + yo
        
    #plot arrows to show the direction of the P integral
    for i in np.arange(0,len(x),len(x)//5)[1:]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="r"),
            size=20,
        )
    #bold the P integral region
    ax.plot(x, y, color='r', lw=2,zorder=-1);
    
    #angle arcs
    stellar_angle = Arc((0, 0), 0.15, 0.15, angle=0, theta1=0, theta2=np.degrees(xi[0]), color="k", alpha=0.7, ls="-", lw=1)
    ax.add_patch(stellar_angle)
    ax.text(0.08,0.04, s=r"$\xi$", size=14, color='k',alpha=0.7)
    
    planet_angle = Arc((xo, yo), 0.25, 0.25, angle=0, theta1=0, theta2=np.degrees(phi[1]), color="r", alpha=0.7, ls="-", lw=1)
    ax.add_patch(planet_angle)
    ax.text(xo+0.15, yo+0.02, s=r"$\phi$", size=13, color='r',alpha=0.7)
    return fig, ax


fig, ax = draw_oblate(0.7, -1.05, 0.6, 0.4, 52.)
fig.savefig(paths.figures / f"oblate_planet.pdf", dpi=300, bbox_inches='tight')