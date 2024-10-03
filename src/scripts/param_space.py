import os
import re
import glob
import pandas as pd
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import scipy.spatial as sp
import scipy.integrate as integrate
from scipy.stats import chi2
from tqdm import tqdm

df = pd.read_csv('planet_params_full.csv', header=0)
planet_names = df['name'].tolist()

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

import pickle


def estimate_convex_hull(h, k, percentage=0.95):
    """
    Estimates the minimum convex hull containing the specified percentage of points.

    Args:
        h: Array of x-coordinates.
        k: Array of y-coordinates.
        percentage: Percentage of points to include in the convex hull.

    Returns:
        Convex hull vertices as a NumPy array.
    """

    # Combine x and y coordinates into a single array
    points = np.column_stack((h, k))

    # Calculate the centroid
    centroid = np.mean(points, axis=0)

    # Find the N closest points to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    N = int(percentage * len(points))
    sorted_indices = np.argsort(distances)
    closest_N_points = points[sorted_indices[:N]]

    # Construct the convex hull
    hull = sp.ConvexHull(closest_N_points)

    return hull

def estimate_contour_area(f, theta, percentage):
    kde = gaussian_kde(np.array([f, theta]))
    x, y = np.linspace(0, .5, 100), np.linspace(-np.pi/2, np.pi/2, 100)
    xx, yy = np.meshgrid(x, y)
    vals = kde.pdf(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    Z = np.log(vals)
    stepsizes = [x[1]-x[0], y[1]-y[0]]
    sortlnL = np.sort(Z.flatten())
    cumsum = np.cumsum(np.exp(sortlnL))*stepsizes[0]*stepsizes[1]
    #print(cumsum.min())
    # Make interpolator
    lnLatpvalue = interp1d(cumsum,sortlnL)
    level = np.exp(lnLatpvalue(1.-percentage))
    mask = vals > level
    area = np.sum(mask)*stepsizes[0]*stepsizes[1]
    return area

parameter_space = {}
thetas = [0,20,40,60,80]
for planet_name in tqdm(planet_names):
    param_arr = []
    for theta_deg in thetas:
        file_pattern = f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5"
        # Search for matching file in the output folder
        matching_files = glob.glob(file_pattern)
        if len(matching_files) >0:
            inf_data = az.from_netcdf(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
            f_samples = np.concatenate(inf_data.posterior.f.to_numpy(), axis=0)
            theta_samples = np.concatenate(inf_data.posterior.theta.to_numpy(), axis=0)
            
            f_low, f_high = np.percentile(f_samples, q=(5., 95.))
            f_median = np.median(f_samples)
            theta_low, theta_high = np.percentile(theta_samples, q=(5., 95.))
            theta_median = np.median(theta_samples)

            period = df[df['name']==planet_name]['pl_orbper'].values[0]
            #h_samples, k_samples = np.concatenate(inf_data.posterior.hk.to_numpy(), axis=0).T
            try:
                hull = estimate_convex_hull(f_samples, theta_samples)
                ftheta_information = hull.volume
            except:
                hull = np.nan
                print(f"Convex hull estimation failed for planet: {planet_name}")
                ftheta_information = np.nan
            
            try:
                ftheta_contour_area = estimate_contour_area(f_samples, theta_samples, 0.95)
            except:
                ftheta_contour_area = np.nan
                print(f"Contour area estimation failed for planet: {planet_name}")
            min_ess_bulk = az.summary(inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['ess_bulk'].min()
            min_ess_tail = az.summary(inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['ess_tail'].min()
            max_r_hat = az.summary(inf_data, var_names=['r_circ', 'bo', 'u', 'hk', 'log_d'])['r_hat'].max()
            param_arr.append([period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat])
            inf_data.close()
        else:
            param_arr.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            print("Did not find a file for planet: ", planet_name)

    param_arr = np.array(param_arr)
    print(param_arr.shape)
    parameter_space[planet_name]=param_arr
        
np.savez('param_space/parameter_space.npz', **parameter_space)


