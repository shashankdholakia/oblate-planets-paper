import arviz as az
import paths
import sys
import os
sys.path.append("..")

import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap

from eclipsoid.light_curve import legacy_oblate_lightcurve, compute_bounds
import zodiax as zdx
from eclipsoid.utils import zero_safe_arctan2

from chainconsumer import ChainConsumer, Chain, PlotConfig, ChainConfig
from chainconsumer.plotting import plot_contour
import pandas as pd
import matplotlib.patches as mpatches

oblate_lightcurve = jit(legacy_oblate_lightcurve)
grad = jit(jax.jacrev(legacy_oblate_lightcurve))

class OblateTransitModel(zdx.Base):
    period: jnp.ndarray
    t0: jnp.ndarray
    r_circ: jnp.ndarray
    u: jnp.ndarray
    bo: jnp.ndarray
    logd: jnp.ndarray
    duration: jnp.ndarray
    t: jnp.ndarray
    h: jnp.ndarray
    k: jnp.ndarray
    f: jnp.ndarray
    theta: jnp.ndarray
    
    def __init__(self, period, t0, r_circ, u, h, bo, k, duration, t):
        self.period = period
        self.r_circ = r_circ
        self.t0 = t0
        self.u = u
        self.h = h
        self.bo = bo
        self.k = k
        self.logd = jnp.log(duration)
        self.t = t
        self.f = (self.h**2 + self.k**2)/2.
        self.theta = zero_safe_arctan2(self.k,self.h)/2.
        self.duration = jnp.exp(self.logd)
        
    def model(self):
      f = (self.h**2 + self.k**2)/2.
      theta = zero_safe_arctan2(self.k,self.h)/2.
      duration = jnp.exp(self.logd)
      #Compute a limb-darkened light curve using starry
      params = {'period':self.period,
                't0':self.t0,
                'radius':jnp.sqrt(self.r_circ**2/(1-f)),
                'u':self.u,
                'f':f,
                'bo':self.bo,
                'theta':theta,
                'duration': duration
        }
      lc = oblate_lightcurve(params,self.t)
      return lc
  
class OblateTransitModel_f(zdx.Base):
    period: jnp.ndarray
    t0: jnp.ndarray
    r_circ: jnp.ndarray
    u: jnp.ndarray
    bo: jnp.ndarray
    logd: jnp.ndarray
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
        self.logd = jnp.log(duration)
        self.t = t
        self.f = f
        self.theta = theta
        self.duration = jnp.exp(self.logd)
        
    def model(self):
      duration = jnp.exp(self.logd)
      #Compute a limb-darkened light curve using starry
      params = {'period':self.period,
                't0':self.t0,
                'radius':jnp.sqrt(self.r_circ**2/(1-self.f)),
                'u':self.u,
                'f':self.f,
                'bo':self.bo,
                'theta':self.theta,
                'duration': duration
        }
      lc = oblate_lightcurve(params,self.t)
      return lc
  

def custom_loglike(model, data, noise):
    sigma2 = noise**2
    return 0.5*jnp.nansum((data - model.model()) ** 2 / sigma2)

# CREATE FIGURE
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))


for i, ppm in enumerate([10, 100]):
# SIMULATE DATA

    np.random.seed(11)
    period_true = np.random.uniform(5, 20)
    t = np.linspace(0.9,1.1,1000)

    true_params = {'period':period_true,
                't0':1.0,
                    'radius':0.1446,
                    'bo':0.8,
                    'u':jnp.array([0.3, 0.2]),
                    'f':0.1,
                    'theta':np.radians(35),
                    'duration': 0.12
    } 

    print(true_params)
    yerr = ppm*1e-6
    # Compute a limb-darkened light curve using starry
    lc_true = oblate_lightcurve(true_params, t-true_params['t0'])
    print(np.any(np.isnan(lc_true)))
    lc = lc_true #+ yerr*np.random.normal(size=len(t))
    
    inf_data = az.from_netcdf(paths.data / f"average_radius_NUTS_{ppm}ppm_bo_{true_params['bo']}.h5")
    print(az.summary(inf_data, var_names=['r_circ', 'bo', 'u', 'f', 'theta', 'duration']))
    h_samples, k_samples = np.concatenate(inf_data.posterior.hk.to_numpy(), axis=0).T
    f_samples = np.concatenate(inf_data.posterior.f.to_numpy(), axis=0).T
    theta_samples = np.concatenate(inf_data.posterior.theta.to_numpy(), axis=0)
    
    r_circ_samples = np.concatenate(inf_data.posterior.r_circ.to_numpy(), axis=0)
    bo_samples = np.concatenate(inf_data.posterior.bo.to_numpy(), axis=0)
    logd_samples = np.concatenate(inf_data.posterior.log_d.to_numpy(), axis=0)
    u1_samples, u2_samples = np.concatenate(inf_data.posterior.u.to_numpy(), axis=0).T
    h = 2*jnp.sqrt(true_params['f']/2)*np.cos(2*true_params['theta'])
    k = 2*jnp.sqrt(true_params['f']/2.)*np.sin(2*true_params['theta'])
    model = OblateTransitModel(true_params['period'], true_params['t0'], true_params['radius']*jnp.sqrt((1-true_params['f'])), true_params['u'], h, true_params['bo'], k, true_params['duration'], t-true_params['t0'])
    opt_params = ['h','k', 'r_circ', 'bo', 'logd','u']
    print("Lnlike: " + str(custom_loglike(model, lc, yerr)))
    fim = zdx.fisher_matrix(model, opt_params, custom_loglike, lc, noise=yerr)
    print("FIM: " + str(fim))
    prior_fim = jnp.zeros((len(fim), len(fim)))
    prior_fim = prior_fim.at[4,4].set(1/0.01**2) #normal prior on logd
    cov_prior = -jnp.linalg.inv(fim+prior_fim)
    #c1 = ChainConsumer()
    X = np.array([model.h,model.k, model.r_circ, model.bo, model.logd, model.u[0],model.u[1]])
    print("Cov with prior: " + str(cov_prior))
    c1_cov = Chain.from_covariance(X[0:2], cov_prior[0:2,0:2], columns=['h','k'], name='Fisher info',
                                   color="#1f77b4", shade_alpha=0.3)
    nuts1 = Chain(samples=pd.DataFrame(np.array([h_samples,k_samples]).T, columns=['h','k']) ,name='NUTS'
                  ,color="#ff7f0e", shade_alpha=0.3)
    
    plot_contour(axes[0, i], chain=c1_cov, px='h',py='k', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))
    plot_contour(axes[0, i], chain=nuts1, px='h',py='k', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))

    axes[0, i].set_xlabel("h", fontsize=14)
    #axes[0, i].set_xlim([-1,1])
    axes[0, i].set_ylabel("k", fontsize=14)
    if ppm==100:
        axes[0, i].set_ylabel("k", fontsize=14) 
    
    #axes[0, i].set_ylim([-1,1])
    
    model = OblateTransitModel_f(true_params['period'], true_params['t0'], true_params['radius']*jnp.sqrt((1-true_params['f'])), true_params['u'], true_params['f'], true_params['bo'], true_params['theta'], true_params['duration'], t-true_params['t0'])
    opt_params = ['f','theta', 'r_circ', 'bo', 'logd','u']
    print("Lnlike: " + str(custom_loglike(model, lc, yerr)))
    fim = zdx.fisher_matrix(model, opt_params, custom_loglike, lc, noise=yerr)
    print("FIM: " + str(fim))
    prior_fim = jnp.zeros((len(fim), len(fim)))
    prior_fim = prior_fim.at[4,4].set(1/0.01**2) #normal prior on logd
    cov_prior = -jnp.linalg.inv(fim+prior_fim)
    print("Cov with prior: " + str(cov_prior))
    
    c2 = ChainConsumer()
    X = np.array([model.f,model.theta, model.r_circ, model.bo, model.logd, model.u[0],model.u[1]])
    c2_cov = Chain.from_covariance(X[0:2], cov_prior[0:2,0:2], columns=['f','theta'], name='Fisher info',
                                   color="#1f77b4", shade_alpha=0.3)
    nuts2 = Chain(samples=pd.DataFrame(np.array([f_samples,theta_samples]).T, columns=['f','theta']) ,name='NUTS',
                  color="#ff7f0e", shade_alpha=0.3)
    
    plot_contour(axes[1, i], chain=c2_cov, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))
    plot_contour(axes[1, i], chain=nuts2, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))

    axes[1, i].set_xlabel("f", fontsize=14)
    #axes[1, i].set_xlim([0,0.5])
    axes[1, i].set_ylabel(r"$\theta$", fontsize=14)
    if ppm==100:
        axes[1, i].set_ylabel(r"$\theta$", fontsize=14) 
    #axes[1, i].set_ylim([-np.pi/2,np.pi/2])

cols = ['10 ppm', '100 ppm']
pad = 5
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
blue_patch = mpatches.Patch(color='C0', alpha=0.3, label='Laplace Approximation')
orange_patch = mpatches.Patch(color='C1', alpha=0.3, label='NUTS')
fig.legend(handles=[blue_patch, orange_patch], loc='upper right')
fig.tight_layout(w_pad=1.0, h_pad=1.0, rect=[0, 0, .94, 0.94])
    

fig.savefig(paths.figures / f"corner_bo{true_params['bo']}.png", dpi=300)