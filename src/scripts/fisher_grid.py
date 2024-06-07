import jax.numpy as jnp
import jax
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)


import arviz as az
import paths
from chainconsumer import ChainConsumer


from eclipsoid.light_curve import legacy_oblate_lightcurve
legacy_oblate_lightcurve = jax.jit(legacy_oblate_lightcurve)

import zodiax as zdx
from eclipsoid.utils import zero_safe_arctan2

class OblateTransitModel(zdx.Base):
    period: jnp.ndarray
    t0: jnp.ndarray
    r_circ: jnp.ndarray
    u: jnp.ndarray
    bo: jnp.ndarray
    duration: jnp.ndarray
    t: jnp.ndarray
    f: jnp.ndarray
    theta: jnp.ndarray
    
    def __init__(self, period, t0, r_circ, u, f, bo, theta, duration, t):
        self.period = period
        self.r_circ = r_circ
        self.t0 = t0
        self.u = u
        self.bo = bo
        self.duration = duration
        self.t = t
        self.f = f
        self.theta = theta
        
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
      lc = legacy_oblate_lightcurve(params,self.t)
      return lc
  
t = jnp.linspace(0.9,1.1,1000)

true_params = {'period':7.704045333151538,
               't0':1.0,
          'radius':0.1446,
          'u':jnp.array([0.3,0.2]),
          'f':0.1,
          'bo':0.20,
          'theta':jnp.radians(35),
          'duration': 0.12
}
yerr = 10*1e-6

data = jnp.array(legacy_oblate_lightcurve(true_params, t-true_params['t0'])+yerr*np.random.normal(size=len(t)))

model = OblateTransitModel(true_params['period'], 
                           true_params['t0'], 
                           true_params['radius']*jnp.sqrt((1-true_params['f'])),
                           true_params['u'], true_params['f'], true_params['bo'],
                           true_params['theta'], true_params['duration'],
                           t-true_params['t0'])
def custom_loglike(model, data, noise):
    sigma2 = noise**2
    return 0.5*np.sum((data - model.model()) ** 2 / sigma2)

img = np.zeros((50, 20))
model = model.set('f', 0.1)
opt_params = ['f', 'r_circ', 'theta', 'bo', 'duration','u']
for i, theta in enumerate(jnp.linspace(0.0,jnp.pi/2, 50)):
    for j, yerr in enumerate(jnp.logspace(-6,-3, 20, base=10)):
        #print(theta, yerr)
        data = jnp.array(legacy_oblate_lightcurve(true_params, t-true_params['t0']))
        cov = zdx.covariance_matrix(model.set('theta',theta), opt_params, custom_loglike, data, noise=yerr)
        #print(cov[0,0])
        img[i,j] = cov[0,0]
        
sigma = 5

matplotlib.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'text.usetex':False,'font.weight':'ultralight',"font.sans-serif": ['Helvetica Neue']})
plt.imshow(true_params['f']-sigma*np.sqrt(np.abs(img)), origin='lower', extent=[-6,-3,jnp.degrees(0.0),jnp.degrees(jnp.pi/2)], aspect=((6-3)/img.shape[1])/(jnp.degrees((jnp.pi/2-0.0))/img.shape[0]), cmap='RdBu_r', vmin=0)
plt.xlabel('Noise (ppm)')
plt.ylabel('Projected obliquity angle (degrees)')
plt.xticks([-6,-5,-4,-3],labels=['1','10','100','1000'])
cb = plt.colorbar(pad=0.01)
cb.set_label(label=f'Oblateness {sigma}$\sigma$ lower bound', labelpad=10)
plt.suptitle(fr"Planetary Oblateness Detectability ", x=0.7)
plt.figtext(x=0.5, y=0.91, s=f"for a transit with {len(t)} data points")
plt.tight_layout()
plt.savefig(paths.figures / f"fisherinfo_theta_yerr_{sigma}sigma_{len(t)}_bo_{true_params['bo']}.png", dpi=300)