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
numpyro.set_host_device_count(2)
jax.config.update('jax_disable_jit', True)

from jaxoplanet import light_curves, orbits
import arviz as az
import corner

from eclipsoid.legacy.light_curve import oblate_lightcurve_dict
oblate_lightcurve = jax.jit(oblate_lightcurve_dict)
import paths

#Fiducial planet parameters:
params = {'period':300.456,
          'radius':0.1,
          'u':jnp.array([0.3,0.2]),
          'f':0.3,
          't0':0.0,
          'bo':0.7,
          'theta':jnp.radians(35),
          'duration': 0.4
}
# The light curve calculation requires an orbit

# Compute a limb-darkened light curve using starry
t = jnp.linspace(-0.3, 0.3, 1000)
#oblate_lightcurve = jit(oblate_lightcurve)
lc = oblate_lightcurve(params,t)

grad = jax.jacrev(oblate_lightcurve)
grads = grad(params, t)
grads.pop("period")
_, axes = plt.subplots(len(grads.keys())+1, 1, figsize=(5, 10), sharex=True, gridspec_kw={'height_ratios':[2]+[1]*len(grads.keys())})
axes[0].plot(t, lc, color="C0", lw=2)
axes[0].set_ylabel("relative flux", fontsize=11)
plt.xlabel("Time [days]", fontsize=13)
axes[0].set_xlim(t.min(), t.max())
for n, key in enumerate(grads.keys()):
    axes[n+1].plot(t, grads[key])
    axes[n+1].set_ylabel(r"$\frac{\partial \mathrm{F}}{\partial \mathrm{ %s }}$" % key, fontsize=13)
_.align_ylabels(axes)
plt.savefig(paths.figures/"partials.png", dpi=300, bbox_inches="tight")

