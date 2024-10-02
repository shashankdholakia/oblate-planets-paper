
import sys
import os
sys.path.append("..")

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
import numpyro_ext.distributions, numpyro_ext.optim


import paths


numpyro.set_host_device_count(
    2
)  # For multi-core parallelism (useful when running multiple MCMC chains in parallel)
numpyro.set_platform("cpu")  # For CPU (use "gpu" for GPU)
jax.config.update(
    "jax_enable_x64", True
)
#jax.config.update('jax_disable_jit', True)

from jaxoplanet import light_curves, orbits
from jaxoplanet.light_curves import limb_dark_light_curve
import arviz as az
import corner

from eclipsoid.legacy.light_curve import oblate_lightcurve_dict, compute_bounds
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

oblate_lightcurve = jit(oblate_lightcurve_dict)
grad = jit(jax.jacrev(oblate_lightcurve_dict))

for ppm in [10, 100]:

# SIMULATE DATA

    np.random.seed(11)
    period_true = np.random.uniform(5, 20)
    t = np.linspace(0.9,1.1,1000)

    true_params = {'period':period_true,
                't0':1.0,
                    'radius':0.1446,
                    'bo':0.6,
                    'u':jnp.array([0.3, 0.2]),
                    'f':0.1,
                    'theta':np.radians(35),
                    'duration': 0.12
    } 

    print(true_params)
    yerr = ppm*1e-6
    # Compute a limb-darkened light curve using starry
    lc_true = oblate_lightcurve(true_params, t-true_params['t0'])

    lc = lc_true #+ yerr*np.random.normal(size=len(t))

# DEFINE MODEL
    def model(t, yerr, y=None):
        # If we wanted to fit for all the parameters, we could use the following,
        # but we'll keep these fixed for simplicity.
        
        #log_duration = numpyro.sample("log_duration", dist.Uniform(jnp.log(0.08), jnp.log(0.2)))
        #b = numpyro.sample("b", dist.Uniform(0.0, 1.0))

        #log_jitter = numpyro.sample("log_jitter", dist.Normal(jnp.log(yerr), 1.0))
        r_circ = numpyro.sample("r_circ", dist.Uniform(0.01, 0.2))
        u = numpyro.sample("u", distx.QuadLDParams())
        bo = numpyro.sample("bo", dist.Uniform(0.0,1.))
        
        #parametrize f, theta using a unit disk
        hk = numpyro.sample("hk", distx.UnitDisk())
        f = numpyro.deterministic("f", (hk[0] ** 2 + hk[1] ** 2)/2)
        theta = numpyro.deterministic("theta", jnp.arctan2(hk[1], hk[0])/2)
        
        # The duration
        log_d = numpyro.sample("log_d", numpyro.distributions.Normal(jnp.log(0.12), 0.01))
        duration = numpyro.deterministic("duration", jnp.exp(log_d))
        
        params = {
            'period':period_true,
            't0': 1.0,
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
        
# INITIALIZE PARAMETERS
    init_params = {'r_circ':jnp.sqrt(true_params['radius']**2*(1-true_params['f'])),
                'u':jnp.array([0.3, 0.2]),
                    'bo':0.61,
                    'log_d': jnp.log(0.12),
                    'hk':jnp.array([jnp.sqrt(0.1)*jnp.sin(jnp.radians(20)), jnp.sqrt(0.1)*jnp.cos(jnp.radians(20))])
                
    }

# RUN OPTIMIZATION
    run_optim = numpyro_ext.optim.optimize(
            model,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
            return_info=True
        )

    opt_params, status = run_optim(jax.random.PRNGKey(2), t, yerr, y=lc)
    print(opt_params)
# RANDOMIZE STARTING POSITIONS
    CHAINS = 2
    #how to initialize each walker at a slightly different point away from the MLE
    init_pos_uncon = numpyro.infer.util.unconstrain_fn(model=model,model_args=(t, yerr, lc),model_kwargs={}, params=opt_params)
    init_pos_random = {}

    for key, value in init_pos_uncon.items():

        if key == 'u' or key=='hk':

            init_pos_random[key] = np.tile(init_pos_uncon[key][:,jnp.newaxis], (1,CHAINS)).T+np.random.normal(0, 0.001, (2,CHAINS)).T

        else:
            print(key)
            if not (key == 'light_curve' or key =='flux' or key =='f' or key == 'theta' or key =='duration'):
                init_pos_random[key] = np.tile(value, CHAINS)+np.random.normal(0, 0.001, (CHAINS,))

# RUN MCMC        
    sampler_wn = infer.MCMC(
        infer.NUTS(
            model,
            target_accept_prob=0.8,
            dense_mass=True,
            regularize_mass_matrix=False,
            max_tree_depth=7,
            #init_strategy=numpyro.infer.init_to_value(values=opt_params)
        ),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )
    sampler_wn.run(jax.random.PRNGKey(11), t, yerr, lc, init_params=init_pos_random)

# SAVE RESULTS
    inf_data_wn = az.from_numpyro(sampler_wn)
    az.summary(inf_data_wn, var_names=['r_circ', 'bo', 'u', 'f', 'theta', 'duration'])

    inf_data_wn.to_netcdf(paths.data / f"average_radius_NUTS_{ppm}ppm_bo_{true_params['bo']}_2.h5")