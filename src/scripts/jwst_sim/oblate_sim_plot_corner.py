NUM_CORES = 1
import argparse
import pandas as pd

import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import numpyro
from numpyro import distributions as dist
from numpyro import infer

from numpyro_ext import distributions as distx
from numpyro_ext import info, optim

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(NUM_CORES)

from jaxoplanet import light_curves, orbits
from jaxoplanet.light_curves import limb_dark_light_curve
import arviz as az

from eclipsoid.legacy.light_curve import oblate_lightcurve_dict

oblate_lightcurve = jit(oblate_lightcurve_dict)

import zodiax as zdx
from eclipsoid.utils import zero_safe_arctan2

from chainconsumer import ChainConsumer


def calc_T14(inc, a, per, rp, rs):
    rs = rs*109.076
    a = a*23454.8
    b = jnp.cos(jnp.radians(inc))*a/(rs)
    #convert jupiter radius into earth radii
    rat = (rp*11.2089)/(rs)
    t14 = per/np.pi* rs/a * np.sqrt((1+rat)**2 - b**2)
    return t14
def bo_func(a,rs,inc):
    a = a*23454.8
    rs = rs*109.076
    return jnp.abs(a*jnp.cos(jnp.deg2rad(inc))/rs)
def inc_func(a,rs,bo):
    a = a*23454.8
    rs = rs*109.076
    return jnp.rad2deg(jnp.arccos(bo*rs/a))

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

def custom_loglike(model, data, noise):
    sigma2 = noise**2
    return 0.5*jnp.sum((data - model.model()) ** 2 / sigma2)

def main(planet_id, theta_id):
    #theta_id in the range 0-9
    true_theta = jnp.linspace(0, jnp.pi, 10)[theta_id]
    theta_deg = int(np.round(np.degrees(true_theta)))
    
    # Read the CSV file
    planet_pop = pd.read_csv("planet_params_full.csv")
    df_noise_pop = pd.read_csv("jexosim_niriss_soss_gr700xd_point_to_point_scatter.csv")


    # Get the row corresponding to the planet_id
    planet_row = planet_pop.iloc[planet_id]

    # Get the name of the planet from the first column
    planet_name = planet_row.iloc[0]
    i = df_noise_pop['Planet Name'][df_noise_pop['Planet Name'] == planet_name].index[0]
    
    print(planet_name)
    print("Planet ID:", planet_id)
    print("Theta ID", theta_id)
    print("Theta:", true_theta)

    if str(df_noise_pop['Planet Name'][i]) in list(planet_pop['name']):
        planet_name = df_noise_pop['Planet Name'][i]
        index = planet_pop.loc[planet_pop['name'] == planet_name].index[0]
        print(planet_pop['pl_orbsmax'][index],planet_pop['st_rad'][index],planet_pop['pl_orbincl'][index])
        
        # if not jnp.isnan(planet_pop['pl_imppar'][index]):
        #         bo = planet_pop['pl_imppar'][index]
        #         inc = inc_func(planet_pop['pl_orbsmax'][index],planet_pop['st_rad'][index],bo)
        #         dur = calc_T14(inc,planet_pop['pl_orbsmax'][index],planet_pop['pl_orbper'][index],planet_pop['pl_rade'][index],planet_pop['st_rad'][index])
        # else:
        bo = bo_func(planet_pop['pl_orbsmax'][index],planet_pop['st_rad'][index],planet_pop['pl_orbincl'][index])
        dur = calc_T14(planet_pop['pl_orbincl'][index],planet_pop['pl_orbsmax'][index],planet_pop['pl_orbper'][index],planet_pop['pl_radj'][index],planet_pop['st_rad'][index])

        print(planet_name)
        
        if not jnp.isnan(dur) and not jnp.isnan(bo):
            u1_true, u1_std = planet_pop['u1'][index], planet_pop['u1_std'][index]
            u2_true, u2_std = planet_pop['u2'][index], planet_pop['u2_std'][index]
            params = {'period':planet_pop['pl_orbper'][index],
            'r_circ':(0.102763*planet_pop['pl_radj'][index] / planet_pop['st_rad'][index]),
            'u':jnp.array([u1_true,u2_true]),
            't0':0,
            'f':0.1,
            'bo':bo,
            'theta':true_theta,
            'duration':dur
            }
                
            cadence = (df_noise_pop.loc[df_noise_pop['Planet Name'] == planet_name]['t_cycle'].iloc[0]/86400)        
            t = jnp.arange(-0.8*dur, 0.8*dur, cadence) #jnp.linspace(-0.8*dur, 0.8*dur,1000)
            h = 2*jnp.sqrt(params['f']/2)*np.cos(2*params['theta'])
            k = 2*jnp.sqrt(params['f']/2.)*np.sin(2*params['theta'])
            model = OblateTransitModel(params['period'], params['t0'], params['r_circ'], params['u'], h, params['bo'], k, params['duration'], t)
            lc = model.model()
            noise = df_noise_pop.loc[df_noise_pop['Planet Name'] == planet_name]["sigma_ppm"]
            noise = float(noise)
            noise = noise/1e6
            noise_arr = jax.random.normal(jax.random.PRNGKey(0), shape=(len(t),))*noise
            
            inf_data = az.from_netcdf(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
            try:
                axes = az.plot_trace(
                    inf_data,
                    var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'],
                    backend_kwargs={"constrained_layout": True},
                )
                
                fig = axes.ravel()[0].figure
                fig.savefig(f"mcmc_sim/chain_plots/chain_{planet_name.replace(" ", "")}_{theta_deg}.png", dpi=300)
            except:
                "No trace plot"
            

            plt.clf()
            
            h_samples, k_samples = np.concatenate(inf_data.posterior.hk.to_numpy(), axis=0).T
            r_circ_samples = np.concatenate(inf_data.posterior.r_circ.to_numpy(), axis=0)
            bo_samples = np.concatenate(inf_data.posterior.bo.to_numpy(), axis=0)
            logd_samples = np.concatenate(inf_data.posterior.log_d.to_numpy(), axis=0)
            u1_samples, u2_samples = np.concatenate(inf_data.posterior.u.to_numpy(), axis=0).T
            
            opt_params = ['h','k', 'r_circ', 'bo', 'logd','u']
            fim = zdx.fisher_matrix(model, opt_params, custom_loglike, lc, noise=noise)
            prior_fim = jnp.zeros((len(fim), len(fim)))
            prior_fim = prior_fim.at[4,4].set(1/0.01**2) #normal prior on logd
            #FIM for uniform from [0,theta] is 1/theta?
            #prior_fim = prior_fim.at[0,0].set(1) #h
            #prior_fim = prior_fim.at[1,1].set(1) #k
            #prior_fim = prior_fim.at[2,2].set(1/0.2) #r_circ
            #prior_fim = prior_fim.at[3,3].set(1.0) #bo
            prior_fim = prior_fim.at[5,5].set(1.0/u1_std**2) #u1
            prior_fim = prior_fim.at[6,6].set(1.0/u2_std**2) #u2
            
            
            cov_prior = -jnp.linalg.inv(fim+prior_fim)
            
            
            ## build a censoring function to remove samples that are in unphysical regions of parameter space
            def logprior(h,k, r_circ, bo, duration, u1, u2):
                if (r_circ < 0) or (r_circ > 0.2):
                    return jnp.nan
                if (bo < 0) or (bo > 1):
                    return jnp.nan
                # if (duration < 0):
                #     return jnp.nan
                if (u1 + u2 > 1):
                    return jnp.nan
                if (u1<0):
                    return jnp.nan
                if (u1+2*u2)<0.:
                    return jnp.nan
                if ((h**2 + k**2) > 1.0):
                    return jnp.nan
                if np.arctan2(k, h) < -np.pi or np.arctan2(k, h) > np.pi:
                    return jnp.nan
                return 0
            
            X = np.array([model.h,model.k, model.r_circ, model.bo, model.logd, model.u[0],model.u[1]])
            # generate samples from scipy.stats.multivariate_normal from X and cov
            samples = np.random.multivariate_normal(X, cov_prior, size=10**6)
            mask = []
            for i in range(samples.shape[0]):
                mask.append(np.isfinite(logprior(*samples[i])))
            samples = samples[mask,:]
            print(samples.shape)
            
            c = ChainConsumer()
            c.add_covariance(X, cov_prior, parameters=['h','k', 'r_circ', 'bo', 'logd','u1','u2'], name='Fisher info')
            c.add_chain(np.array([h_samples,k_samples,r_circ_samples,bo_samples,logd_samples,u1_samples, u2_samples]).T, parameters=['h','k', 'r_circ', 'bo', 'logd','u1','u2'],name='NUTS')
            c.add_chain(samples, parameters=['h','k', 'r_circ', 'bo', 'logd','u1','u2'],name='Fisher info (truncated)')

            #calculate the 95% percentiles for f and theta from the MCMC chains
            f_samples = np.concatenate(inf_data.posterior.f.to_numpy(), axis=0)
            theta_samples = np.concatenate(inf_data.posterior.theta.to_numpy(), axis=0)
            f_mcmc = np.percentile(f_samples, q=(5., 95.))
            theta_mcmc = np.percentile(theta_samples, q=(5., 95.))
            fisher_f_samples = (samples[:,0]**2+samples[:,1]**2)/2.0
            print(np.min(fisher_f_samples), np.max(fisher_f_samples))
            fisher_theta_samples = np.arctan2(samples[:,1], samples[:,0]) / 2.0
            f_fisher = np.percentile(fisher_f_samples, q=(5., 95.))
            theta_fisher = np.percentile(fisher_theta_samples, q=(5., 95.))
            
            percentiles = pd.DataFrame({'MCMC f':f_mcmc, 'MCMC theta': theta_mcmc, 
                                        'Fisher f':f_fisher, 'Fisher theta':theta_fisher})
            
            #save the 5th and 95th percentiles for f and theta
            percentiles.to_csv(f"fisher_sim/percentiles/percentile_{planet_name.replace(" ", "")}_{theta_deg}.csv")
            
            c.configure(serif=True, shade=True, bar_shade=True, shade_alpha=0.2, spacing=1., max_ticks=5)
            fig = c.plotter.plot()
            fig.savefig(f"fisher_sim/corner_plots/corner_{planet_name.replace(" ", "")}_{theta_deg}.png", dpi=300)
            
            ### PLOT LC predictive samples:
            
            N = 30
            sample_inds = np.random.choice(np.concatenate(inf_data.posterior.light_curve.values, axis=0).shape[0], N, replace=False)
            sample_lcs = np.concatenate(inf_data.posterior.light_curve.values, axis=0)[sample_inds]
            
            h = 2*jnp.sqrt(params['f']/2)*np.cos(2*params['theta'])
            k = 2*jnp.sqrt(params['f']/2.)*np.sin(2*params['theta'])
            r_circ = params['r_circ']
            logd = jnp.log(params['duration'])
            u = params['u']
            #params_draw =np.random.multivariate_normal(np.array([h,k,r_circ, params['bo'], logd, u[0], u[1]]), cov_prior, size=N, check_valid='warn', tol=1e-8)
            #draw a random index to 
            params_indices_draw = np.random.choice(range(len(samples)), size=N, replace=False)
            plt.clf()
            data = lc+noise_arr
            fig, (ax, ax1) = plt.subplots(2,1,figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})
            ax.plot(t, data, "k.", label="data")
            ax.plot(t, model.model(),c='r', zorder=3, label='NUTS')
            ax.plot(t, model.model(),c='c', zorder=2, label='Fisher')
            ax1.plot(t, data-model.model(), "k.")
            for lc_draw, index in zip(sample_lcs, params_indices_draw):
                ax.plot(t, lc_draw,c='r', alpha=0.2, lw=1)
                ax1.plot(t, lc_draw-model.model(),c='r', alpha=0.2, lw=1)
                
                f = (samples[index][0]**2 + samples[index][1]**2)/2.
                theta = jnp.arctan2(samples[index][1],samples[index][0])/2.
                r_eq = samples[index][2]/(jnp.sqrt(1.-f))
                duration = jnp.exp(samples[index][4])
                param_dict = {'period':params['period'],
                        't0':0.0,
                    'radius':r_eq,
                    'u':np.array([samples[index][5], samples[index][6]]),
                    'f':f,
                    'bo':samples[index][3],
                    'theta':theta,
                    'duration': duration
                }
                
                fim_lc = oblate_lightcurve(param_dict,t-params['t0'])
                ax.plot(t, fim_lc ,c='C9', alpha=0.5, lw=1)
                ax1.plot(t, fim_lc-model.model(),c='C9', alpha=0.5, lw=1)
            fig.legend()
            fig.savefig(f"fisher_sim/posterior_predictive/lc_{planet_name.replace(" ", "")}_{theta_deg}.png", dpi=300)
            
        else:
            raise ValueError("Either duration or impact parameter are nan and cannot be calculated")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create simulation scripts for given planet_id.')
    parser.add_argument('--planet_id', type=int, required=True, help='The ID of the planet.')
    parser.add_argument('--theta_id', type=int, required=True, help='The number in the theta grid')

    args = parser.parse_args()

    main(args.planet_id, args.theta_id)
