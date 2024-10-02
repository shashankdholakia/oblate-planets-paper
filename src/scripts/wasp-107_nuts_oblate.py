NUM_CORES = 2
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
import paths

oblate_lightcurve = jit(oblate_lightcurve_dict)

df = pd.read_csv(paths.data/'Sing_2024_Fig1_WASP107b_white_light_curve_data.csv')
t = jnp.array(df['bjd_tdb(days)'] - 2460118.948861)
nsr_1_f = jnp.array(df['NRS1_wlc_flux'])
nsr_1_f_err = jnp.array(df['NRS1_wlc_flux_err'])
nrs_2_f = jnp.array(df['NRS2_wlc_flux'])
nrs_2_f_err = jnp.array(df['NRS2_wlc_flux_err'])

init_params_nrs1 = {}
init_params_nrs1['f'] = 0.001
init_params_nrs1['theta'] = np.radians(20)
init_params_nrs1['u'] = jnp.array([0.15, 0.18])
init_params_nrs1['duration']=2.753/24
init_params_nrs1['bo'] = 0.11650
init_params_nrs1['t0']=0.0
init_params_nrs1['radius'] = 0.1446
init_params_nrs1['period'] = 5.7214742

init_params_nrs2 = init_params_nrs1.copy()
init_params_nrs2['u'] = jnp.array([0.11,0.13])

def model(t, yerr, y=None):
    yerr_nrs1, yerr_nrs2 = yerr
    y_nrs1, y_nrs2 = y


    r_circ = numpyro.sample("r_circ", dist.Uniform(0.02, 0.2))
    u_nrs1 = numpyro.sample("u_nrs1", distx.QuadLDParams())
    u_nrs2 = numpyro.sample("u_nrs2", distx.QuadLDParams())
    bo = numpyro.sample("bo", dist.Uniform(0.01,0.9))
    
    #parametrize f, theta using a unit disk
    hk = numpyro.sample("hk", distx.UnitDisk())
    f = numpyro.deterministic("f", (hk[0] ** 2 + hk[1] ** 2)/2)
    theta = numpyro.deterministic("theta", jnp.arctan2(hk[1], hk[0])/2)
    
    # The duration
    log_d = numpyro.sample("log_d", numpyro.distributions.Normal(jnp.log(init_params_nrs1['duration']), 0.01))
    duration = numpyro.deterministic("duration", jnp.exp(log_d))
    t0 = numpyro.sample("t0", dist.Uniform(-0.0001, 0.0001))
    
    params_nrs1 = {
        'period':init_params_nrs1['period'],
        't0': t0,
        "radius": jnp.sqrt(r_circ**2/(1-f)),
        'bo':bo,
        'u': u_nrs1,
        'f':f, 
        'theta':theta,
        'duration': duration
    }
    
    params_nrs2 = {
        'period':init_params_nrs1['period'],
        't0': t0,
        "radius": jnp.sqrt(r_circ**2/(1-f)),
        'bo':bo,
        'u': u_nrs2,
        'f':f, 
        'theta':theta,
        'duration': duration
    }
    
    #jitter term in ppm
    nrs1_log_jitter = numpyro.sample("nrs1_log_jitter", dist.Normal(jnp.log(jnp.median(yerr_nrs1)), 1.0))
    nrs2_log_jitter = numpyro.sample("nrs_2log_jitter", dist.Normal(jnp.log(jnp.median(yerr_nrs2)), 1.0))
    
    #add a linear trend centered on a slope of 0 and an intercept of 1
    #in ppm
    nrs1_slope = numpyro.sample("nrs1_slope", dist.Uniform(-0.001,0.001))
    nrs2_slope = numpyro.sample("nrs2_slope", dist.Uniform(-0.001, 0.001))
    nrs1_intercept = numpyro.sample("nrs1_intercept", dist.Uniform(-0.001, 0.001))
    nrs2_intercept = numpyro.sample("nrs2_intercept", dist.Uniform(-0.001, 0.001))
    
    
    y_pred_nrs1 = oblate_lightcurve(params_nrs1, t)+nrs1_slope*t+nrs1_intercept
    y_pred_nrs2 = oblate_lightcurve(params_nrs2, t)+nrs2_slope*t+nrs2_intercept
    
    numpyro.deterministic("nrs1_light_curve", y_pred_nrs1)
    numpyro.deterministic("nrs2_light_curve", y_pred_nrs2)
    numpyro.sample(
        "nrs1_flux",
        dist.Normal(y_pred_nrs1, jnp.sqrt(yerr_nrs1**2+jnp.exp(2 * nrs1_log_jitter))
                    ),
        obs=y_nrs1,
    )
    
    numpyro.sample(
        "nrs2_flux",
        dist.Normal(y_pred_nrs2, jnp.sqrt(yerr_nrs2**2+jnp.exp(2 * nrs2_log_jitter))
                    ),
        obs=y_nrs2,
    )
    
opt_init_params = {
    "r_circ": 0.1446,
    "u_nrs1": jnp.array([0.15, 0.18]),
    "u_nrs2": jnp.array([0.11, 0.13]),
    "bo": 0.11650,
    "hk": jnp.array([0.01, 0.01]),
    "log_d": jnp.log(2.753/24),
    "t0": 0.0,
    "nrs1_log_jitter": 0.0,
    "nrs2_log_jitter": 0.0,
    "nrs1_slope": 0.0,
    "nrs2_slope": 0.0,
    "nrs1_intercept": 0.0,
    "nrs2_intercept": 0.0
}

run_optim = numpyro_ext.optim.optimize(
    model,
    init_strategy=numpyro.infer.init_to_value(values=opt_init_params),
    return_info=True
)
opt_params, status = run_optim(jax.random.PRNGKey(2), t, jnp.array([nsr_1_f_err, nrs_2_f_err]), y=jnp.array([nsr_1_f, nrs_2_f]))

print(opt_params)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
ax1.scatter(t,nsr_1_f,c='k',s=1)
ax2.scatter(t,nrs_2_f,c='k',s=1)
ax1.plot(t,opt_params['nrs1_light_curve'],c='r')
ax1.set_title("NRS1")
ax2.plot(t,opt_params['nrs2_light_curve'],c='r')
ax2.set_title("NRS2")
plt.savefig(paths.figures/'nrs1_nrs2_optim.png',dpi=300)


sampler = infer.MCMC(
    infer.NUTS(
        model,
        target_accept_prob=0.8,
        dense_mass=True,
        regularize_mass_matrix=False,
        max_tree_depth=8,
        init_strategy=numpyro.infer.init_to_value(values=opt_params),
    ),
    num_warmup=2000,
    num_samples=2000,
    num_chains=NUM_CORES,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(10), t, jnp.array([nsr_1_f_err, nrs_2_f_err]), y=jnp.array([nsr_1_f, nrs_2_f]))

sampler.print_summary()
inf_data = az.from_numpyro(sampler)
az.summary(inf_data, var_names=['t0','r_circ', 'bo', 'u_nrs1','u_nrs2', 'f', 'theta', 'duration', 'hk','nrs1_log_jitter', 'nrs2_log_jitter', 'nrs1_slope', 'nrs2_slope', 'nrs1_intercept', 'nrs2_intercept'])
inf_data.to_netcdf(paths.data/"wasp_107_oblate_mcmc_jointfit_posterior.h5")

