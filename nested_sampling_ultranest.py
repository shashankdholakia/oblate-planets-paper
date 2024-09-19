#module load openmpi/4.1.6-gcc-5my
import sys
import os
cores = 1
#os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"
sys.path.append("..")
import jax.numpy as jnp
import numpy as np
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from numpyro_ext import distributions as distx
from numpyro_ext import info, optim
numpyro.set_host_device_count(cores)  # For multi-core parallelism (useful when running multiple MCMC chains in parallel)
numpyro.set_platform("cpu")  # For CPU (use "gpu" for GPU)
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_disable_jit', True)
from jaxoplanet import light_curves, orbits
from jaxoplanet.light_curves import limb_dark_light_curve
import arviz as az
import corner
from eclipsoid.light_curve import legacy_oblate_lightcurve_fast, compute_bounds
import os
import pandas as pd
from numpyro.contrib.nested_sampling import NestedSampler
from jax.scipy.stats.norm import ppf
from ultranest import ReactiveNestedSampler
from functools import partial
import corner
import time

oblate_lightcurve = jit(legacy_oblate_lightcurve_fast)
print(jax.devices())
print(jax.default_backend())

def ns_fit_lc(lclist1,lclist2):
    t_1 = lclist1[0]
    lc_1 = lclist1[1]
    yerr_1 = lclist1[2]
    t_2 = lclist2[0]
    lc_2 = lclist2[1]
    yerr_2 = lclist2[2]
    parameters = ['t0', 'r_circ', 'q1_nrs1', 'q2_nrs1', 'q1_nrs2', 'q2_nrs2', 'bo', 'f', 'theta', 'logd']
    @jit
    @partial(vmap,in_axes=0)
    def prior_transform(cube):
        print(cube)
        t0,r_circ,q1_nrs1,q2_nrs1,q1_nrs2,q2_nrs2,bo,f,theta,logd = cube
        t0 = t0*0.0002 - 0.0001
        r_circ = r_circ*0.02 + 0.135
        q1_nrs1 = ppf(q1_nrs1, jnp.mean(q1_pdist_nrs1), 3*jnp.std(q1_pdist_nrs1))
        q2_nrs1 = ppf(q2_nrs1, jnp.mean(q2_pdist_nrs1), 3*jnp.std(q2_pdist_nrs1))
        q1_nrs2 = ppf(q1_nrs2, jnp.mean(q1_pdist_nrs2), 3*jnp.std(q1_pdist_nrs2))
        q2_nrs2 = ppf(q2_nrs2, jnp.mean(q2_pdist_nrs2), 3*jnp.std(q2_pdist_nrs2))
        bo = bo*0.90
        f = f*0.3
        theta = theta*jnp.pi - jnp.pi/2
        logd = ppf(logd, jnp.log(0.12), 0.01)
        return jnp.array([t0,r_circ,q1_nrs1,q2_nrs1,q1_nrs2,q2_nrs2,bo,f,theta,logd])
    @jit
    @vmap
    def log_likelihood(params):
        t0,r_circ,q1_nrs1,q2_nrs1,q1_nrs2,q2_nrs2,bo,f,theta,logd = params
        u_nrs1 = q_to_u(q1_nrs1, q2_nrs1)
        u_nrs2 = q_to_u(q1_nrs2, q2_nrs2)
        params_1 = {
            'period':5.7214742,
            't0': t0,
            "radius": jnp.sqrt(r_circ**2/(1-f)),
            'bo':bo,
            'u': u_nrs1,
            'f':f, 
            'theta':theta,
            'duration': jnp.exp(logd)
        }
        params_2 = {
            'period':5.7214742,
            't0': t0,
            "radius": jnp.sqrt(r_circ**2/(1-f)),
            'bo':bo,
            'u': u_nrs2,
            'f':f,
            'theta':theta,
            'duration': jnp.exp(logd)
        }
        y_pred = jnp.array([oblate_lightcurve(params_1, t),oblate_lightcurve(params_2, t)])
        return jnp.sum(-0.5*((y_pred - jnp.array([lc_1,lc_2]))/jnp.array([yerr_1,yerr_2]))**2)

    def np_loglike(value):
        return np.array(log_likelihood(value))
    def np_prior_transform(cube):
        return np.array(prior_transform(cube))

    sampler = ReactiveNestedSampler(parameters, np_loglike, np_prior_transform,log_dir='50_pt_run_v1.2',vectorized=True,
        wrapped_params=[False, False, False, False, False, False, False, False, True, False],
    )
    result = sampler.run(min_num_live_points=64, dKL=jnp.inf, min_ess=100)
    print(result)
    sampler.plot_corner()
    sampler.plot_run()
    sampler.plot_trace()

    '''
    ndim = len(parameters)
    plt.scatter(t, lc_1, label='NRS1',s=1, color='black')
    for i in range(1):
        param_tests = prior_transform(jnp.array([np.random.uniform(size=ndim)]))
        t0,r_circ,q1_nrs1,q2_nrs1,q1_nrs2,q2_nrs2,bo,f,theta,logd = param_tests[0]
        u_nrs1 = q_to_u(q1_nrs1, q2_nrs1)
        params_dict = {
            'period':5.7214742,
            't0': t0,
            "radius": jnp.sqrt(r_circ**2/(1-f)),
            'bo':bo,
            'u': u_nrs1,
            'f':f, 
            'theta':theta,
            'duration': jnp.exp(logd)
        }
        plt.plot(t, oblate_lightcurve(params_dict, t), label='NRS1 Model', color='red',alpha=0.2)
    plt.savefig('NRS1_lightcurve_prior_pred.png')
    '''




def q_to_u(q1, q2): 
    u1 = 2.*jnp.sqrt(jnp.abs(q1))*q2
    u2 = jnp.sqrt(jnp.abs(q1))*(1-(2*q2))
    return jnp.array([u1,u2])
def u_to_q(u1, u2):
    q1 = (u1+u2)**2
    q2 = 0.5*u1/(u1+u2)
    return jnp.array([q1,q2])
u1_mean_nrs1 = 0.15195592139632444
u2_mean_nrs1 = 0.18587019822663192
u1_std_nrs1 = 0.0019117648527172749
u2_std_nrs1 = 0.0024034088242590812
u1_pdist_nrs1 = jnp.array(np.random.normal(u1_mean_nrs1, u1_std_nrs1, 10000))
u2_pdist_nrs1 = jnp.array(np.random.normal(u2_mean_nrs1, u2_std_nrs1, 10000))
q1_pdist_nrs1, q2_pdist_nrs1 = u_to_q(u1_pdist_nrs1, u2_pdist_nrs1)   
u1_mean_nrs2 = 0.11891449249168831
u2_mean_nrs2 = 0.13178213621370768
u1_std_nrs2 = 0.00439467826657133
u2_std_nrs2 = 0.0024725737355863735
u1_pdist_nrs2 = jnp.array(np.random.normal(u1_mean_nrs2, u1_std_nrs2, 10000))
u2_pdist_nrs2 =jnp.array(np.random.normal(u2_mean_nrs2, u2_std_nrs2, 10000))
q1_pdist_nrs2, q2_pdist_nrs2 = u_to_q(u1_pdist_nrs2, u2_pdist_nrs2)

df = pd.read_csv('Sing_2024_Fig1_WASP107b_white_light_curve_data.csv')
t = jnp.array(df['bjd_tdb(days)'] - 2460118.948861)
nsr_1_f = jnp.array(df['NRS1_wlc_flux'])
nsr_1_f_err = jnp.array(df['NRS1_wlc_flux_err'])
nrs_2_f = jnp.array(df['NRS2_wlc_flux'])
nrs_2_f_err = jnp.array(df['NRS2_wlc_flux_err'])
lclist_nrs1 = jnp.array([t, nsr_1_f, nsr_1_f_err])
lclist_nrs2 = jnp.array([t, nrs_2_f, nrs_2_f_err])

if __name__ == '__main__':
    #for i in range(100):
    #    output = pm_search(ticlist[i])
    #    if output == 1:
    #        break
    ns_fit_lc(lclist_nrs1, lclist_nrs2)


