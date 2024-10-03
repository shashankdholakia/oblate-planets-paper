NUM_CORES = 4
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
import numpyro_ext.distributions, numpyro_ext.optim

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CORES}"

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(NUM_CORES)

from jaxoplanet import light_curves, orbits
from jaxoplanet.light_curves import limb_dark_light_curve
import arviz as az

from eclipsoid.legacy.light_curve import oblate_lightcurve_dict

oblate_lightcurve = jit(oblate_lightcurve_dict)

import zodiax as zdx
from eclipsoid.utils import zero_safe_arctan2

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
        self.duration = duration
        self.t = t
        self.f = (self.h**2 + self.k**2)/2.
        self.theta = zero_safe_arctan2(self.k,self.h)/2.
        
    def model(self):
      f = (self.h**2 + self.k**2)/2.
      theta = zero_safe_arctan2(self.k,self.h)/2.
      #Compute a limb-darkened light curve using starry
      params = {'period':self.period,
                't0':self.t0,
                'radius':jnp.sqrt(self.r_circ**2/(1-f)),
                'u':self.u,
                'f':f,
                'bo':self.bo,
                'theta':theta,
                'duration': self.duration
        }
      lc = oblate_lightcurve(params,self.t)
      return lc
  
def custom_loglike(model, data, noise):
    sigma2 = noise**2
    return 0.5*jnp.sum((data - model.model()) ** 2 / sigma2)

def main(planet_id, theta_id):
    #theta_id in the range 0-9
    true_theta = jnp.linspace(0, jnp.pi, 10)[theta_id]
    
    # Read the CSV file
    planet_pop = pd.read_csv("planet_params_full.csv")
    df_noise_pop = pd.read_csv("jexosim_niriss_soss_gr700xd_point_to_point_scatter.csv")


    # Get the row corresponding to the planet_id
    planet_row = planet_pop.iloc[planet_id]

    # Get the name of the planet from the first column
    planet_name = planet_row.iloc[0]
    print(planet_name)
    print("Planet ID:", planet_id)
    print("Theta ID", theta_id)
    print("Theta:", true_theta)
    i = df_noise_pop['Planet Name'][df_noise_pop['Planet Name'] == planet_name].index[0]

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

        
        if not jnp.isnan(dur) and not jnp.isnan(bo):
            

            params = {'period':planet_pop['pl_orbper'][index],
            'r_circ':(0.102763*planet_pop['pl_radj'][index] / planet_pop['st_rad'][index]),
            'u':jnp.array([planet_pop['u1'][index],planet_pop['u2'][index]]),
            't0':0,
            'f':0.1,
            'bo':bo,
            'theta':true_theta,
            'duration':dur
            }
            
            #fudge to avoid instability at theta=0
            if true_theta==0:
                params['theta'] = 0.001
                
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
            #comment out below to run MCMC on noise free data
            #lc += noise_arr

            period_true = planet_pop['pl_orbper'][index]
            duration_true = dur
            
            theta_deg = int(np.round(np.degrees(true_theta)))
            if os.path.exists(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5"):
                temp_inf_data = az.from_netcdf(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
                min_ess_bulk = az.summary(temp_inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['ess_bulk'].min()
                min_ess_tail = az.summary(temp_inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['ess_tail'].min()
                max_r_hat = az.summary(temp_inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['r_hat'].max()
                
                if min_ess_bulk < 100 or min_ess_tail < 100 or max_r_hat > 1.1 or bo < 0.05:
                    print("Rerunning MCMC")
                else:
                    raise ValueError("Already run a converged MCMC for this planet and theta")
                
                temp_inf_data.close()
                os.remove(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
                
            def model(t, yerr, y=None):
                # If we wanted to fit for all the parameters, we could use the following,
                # but we'll keep these fixed for simplicity.
                
                #log_duration = numpyro.sample("log_duration", dist.Uniform(jnp.log(0.08), jnp.log(0.2)))
                #b = numpyro.sample("b", dist.Uniform(0.0, 1.0))

                #log_jitter = numpyro.sample("log_jitter", dist.Normal(jnp.log(yerr), 1.0))
                u1_true, u1_std = planet_pop['u1'][index], planet_pop['u1_std'][index]
                u2_true, u2_std = planet_pop['u2'][index], planet_pop['u2_std'][index]
                r_circ = numpyro.sample("r_circ", dist.Uniform(0.01, 0.2))
                #u = numpyro.sample("u", distx.QuadLDParams())
                u = numpyro.sample("u", dist.Normal(jnp.array([u1_true, u2_true]), jnp.array([u1_std, u2_std])))
                bo = numpyro.sample("bo", dist.Uniform(0.0,1.))
                
                #parametrize f, theta using a unit disk
                hk = numpyro.sample("hk", distx.UnitDisk())
                f = numpyro.deterministic("f", (hk[0] ** 2 + hk[1] ** 2)/2)
                theta = numpyro.deterministic("theta", jnp.arctan2(hk[1], hk[0])/2)
                
                # The duration
                log_d = numpyro.sample("log_d", numpyro.distributions.Normal(jnp.log(duration_true), 0.01))
                duration = numpyro.deterministic("duration", jnp.exp(log_d))
                
                params = {
                    'period':period_true,
                    't0': 0.0,
                    "radius": jnp.sqrt(r_circ**2/(1-f)),
                    'bo':bo,
                    'u': u,
                    'f':f, 
                    'theta':theta,
                    'duration': duration
                }
                y_pred = oblate_lightcurve(params, t-params['t0'])
                numpyro.deterministic("light_curve", y_pred)
                numpyro.sample(
                    "flux",
                    dist.Normal(y_pred, jnp.sqrt(yerr**2) #+ jnp.exp(2 * log_jitter))
                                ),
                    obs=y,
                )
            init_params = {'r_circ':params['r_circ'],
               'u':jnp.array([0.2, 0.1]),
                'bo':params['bo'],
                'log_d': jnp.log(dur),
                'hk':jnp.array([jnp.sqrt(0.11)*jnp.sin(jnp.radians(theta_deg)), jnp.sqrt(0.11)*jnp.cos(jnp.radians(theta_deg))])
            }
            
            run_optim = numpyro_ext.optim.optimize(
                model,
                init_strategy=numpyro.infer.init_to_value(values=init_params),
                return_info=True
            )

            opt_params, status = run_optim(jax.random.PRNGKey(2), t, noise, y=lc)
            print({x: opt_params[x] for x in opt_params if x not in {'flux'}})
            print(status)

            #how to initialize each walker at a slightly different point away from the MLE
            init_pos_uncon = numpyro.infer.util.unconstrain_fn(model=model,model_args=(t, noise, lc),model_kwargs={}, params=opt_params)
            init_pos_random = {}

            for key, value in init_pos_uncon.items():

                if key == 'u' or key=='hk':

                    init_pos_random[key] = np.tile(init_pos_uncon[key][:,jnp.newaxis], (1,NUM_CORES)).T+np.random.normal(0, 0.01, (2,NUM_CORES)).T

                else:
                    print(key)
                    if not (key == 'light_curve' or key =='flux' or key =='f' or key == 'theta' or key =='duration'):
                        init_pos_random[key] = np.tile(value, NUM_CORES)+np.random.normal(0, 0.001, (NUM_CORES,))*value
            print("Random initial starting point for MCMC:")
            print(init_pos_random)
            sampler = infer.MCMC(
                infer.NUTS(
                    model,
                    target_accept_prob=0.8,
                    dense_mass=True,
                    regularize_mass_matrix=False,
                    max_tree_depth=8,
                    #init_strategy=numpyro.infer.init_to_value(values=opt_params),
                ),
                num_warmup=2000,
                num_samples=2000,
                num_chains=NUM_CORES,
                progress_bar=True,
            )
            
            sampler.run(jax.random.PRNGKey(10), t, noise, lc, init_params=init_pos_random) #jax.random.PRNGKey(11)
            sampler.print_summary()
            inf_data = az.from_numpyro(sampler)
            az.summary(inf_data, var_names=['r_circ', 'bo', 'u', 'f', 'theta', 'duration', 'hk'])
            inf_data.to_netcdf(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
        else:
            raise ValueError("Either duration or impact parameter are nan and cannot be calculated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create simulation scripts for given planet_id.')
    parser.add_argument('--planet_id', type=int, required=True, help='The ID of the planet.')
    parser.add_argument('--theta_id', type=int, required=True, help='The number in the theta grid')

    args = parser.parse_args()

    main(args.planet_id, args.theta_id)
