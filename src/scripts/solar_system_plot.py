import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import zodiax as zdx

from jaxoplanet import light_curves, orbits
from jaxoplanet.light_curves import limb_dark_light_curve
import arviz as az
import corner

from eclipsoid.light_curve import oblate_lightcurve, compute_bounds
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import matplotlib 

matplotlib.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'text.usetex':False,'font.weight':'ultralight',"font.sans-serif": ['Helvetica Neue']})

oblate_lightcurve = jit(oblate_lightcurve)

class OblateTransitModel(zdx.Base):
    period: jnp.ndarray
    t0: jnp.ndarray
    r_circ: jnp.ndarray
    u: jnp.ndarray
    f: jnp.ndarray
    bo: jnp.ndarray
    theta: jnp.ndarray
    duration: jnp.ndarray
    t: jnp.ndarray
    
    def __init__(self, period, t0, r_circ, u, f, bo, theta, duration, t):
        self.period = period
        self.r_circ = r_circ
        self.t0 = t0
        self.u = u
        self.f = f
        self.bo = bo
        self.theta = theta
        self.duration = duration
        self.t = t
        
    def model(self):
      #Compute a limb-darkened light curve using starry
      params = {'period':self.period,
                't0':self.t0,
                'radius':jnp.sqrt(self.r_circ**2/(1-self.f)),
                'u':self.u,
                'f':self.f,
                'bo':self.bo,
                'theta':self.theta,
                'duration': self.duration
        }
      lc = oblate_lightcurve(params,self.t)
      return lc
  
#Fiducial planet parameters:
params_saturn = {'period':365,
          'radius':0.0866,
          'u':jnp.array([0.3,0.2]),
          'f':0.098,
          'bo':0.7,
          'theta':jnp.radians(26.73),
          'duration': 0.4
}
params_jupiter = {'period':365,
          'radius':0.1028,
          'u':jnp.array([0.3,0.2]),
          'f':0.0649,
          'bo':0.7,
          'theta':jnp.radians(3.13),
          'duration': 0.4
}
params_uranus = {'period':365,
          'radius':0.0365,
          'u':jnp.array([0.3,0.2]),
          'f':0.0229,
          'bo':0.7,
          'theta':jnp.radians(97.77),
          'duration': 0.4
}
# The light curve calculation requires an orbit

# Compute a limb-darkened light curve using starry
fig, ax = plt.subplots(2,1, figsize=(5, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
fig.patch.set_facecolor('white')
fig.patch.set_alpha(0.3)

ax[0].patch.set_facecolor('white')
ax[0].patch.set_alpha(0.3)

ax[1].patch.set_facecolor('white')
ax[1].patch.set_alpha(0.3)

t = jnp.linspace(-0.3, 0.3, 1000)
saturn = OblateTransitModel(params_saturn['period'], 0.0, 
                            params_saturn['radius']*jnp.sqrt((1-params_saturn['f'])), 
                            params_saturn['u'], params_saturn['f'], params_saturn['bo'], 
                            params_saturn['theta'], params_saturn['duration'], t)

ax[0].plot(t, saturn.model(), color="C0", lw=2, label="Saturn", zorder=2)
ax[0].set_ylabel("relative flux")

circ = saturn.set("f",0.0)

ax[1].plot(t, 1e6*(saturn.model()-circ.model()), color="C0", lw=2, zorder=2)
ax[1].set_ylabel("deviation from circular [ppm]")
ax[1].set_xlabel("time [days]")
_ = ax[1].set_xlim(t.min(), t.max())
fig.align_ylabels(ax)

jupiter = OblateTransitModel(params_jupiter['period'], 0.0, 
                            params_jupiter['radius']*jnp.sqrt((1-params_jupiter['f'])), 
                            params_jupiter['u'], params_jupiter['f'], params_jupiter['bo'], 
                            params_jupiter['theta'], params_jupiter['duration'], t)

ax[0].plot(t, jupiter.model(), color="C1", lw=2, label="Jupiter", zorder=1)
circ = jupiter.set("f",0.0)
ax[1].plot(t, 1e6*(jupiter.model()-circ.model()), color="C1", lw=2, zorder=1)

uranus = OblateTransitModel(params_uranus['period'], 0.0, 
                            params_uranus['radius']*jnp.sqrt((1-params_uranus['f'])), 
                            params_uranus['u'], params_uranus['f'], params_uranus['bo'], 
                            params_uranus['theta'], params_uranus['duration'], t)

ax[0].plot(t, uranus.model(), color="C2", lw=2, label="Uranus", zorder=0)
circ = uranus.set("f",0.0)
ax[1].plot(t, 1e6*(uranus.model()-circ.model()), color="C2", lw=2, zorder=0)
fig.legend()
fig.subplots_adjust(hspace=0.1)
plt.savefig("oblate_lc.png", dpi=300, bbox_inches="tight")