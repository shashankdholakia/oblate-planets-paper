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
import textalloc as ta
import matplotlib.cm as cm

df = pd.read_csv('planet_params_full.csv', header=0)
planet_names = df['name'].tolist()

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import pickle

parameter_space = np.load('param_space/parameter_space.npz')

def forward(x):
    return np.sign(x) * (np.abs(x)) ** (1 / 2)

def inverse(x):
    return x**2
########## ALIGNED CASE ##########

#period vs convex hull area
# Variable to store all annotation objects
texts = []
periods = []
ftheta_infos = []
f_lows = []
info_threshold = 0.7
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for key, value in parameter_space.items():
    if key != 'EPIC 248847494 b':
        period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = value[0]
        periods.append(period)
        ftheta_infos.append(ftheta_information)
        if min_ess_bulk < 10 or min_ess_tail < 10 or max_r_hat > 1.6:
            #ax.scatter(period, ftheta_information, c='r', s=50)
            print("Problematic planet: ", key, min_ess_bulk, min_ess_tail, max_r_hat)
        #plt.scatter(period, ftheta_information, c='b')
        #plt.plot(period, ftheta_information, c='k')
        f_lows.append(f_low)
        # Annotate points with y-value less than 0.1
        if ftheta_information < info_threshold:
            
            # annotation_x = period + 0.05*period 
            # annotation_y = ftheta_information-0.015

            # text = plt.annotate(key, 
            #                     xy=(period, ftheta_information), 
            #                     xytext=(annotation_x, annotation_y),  # Randomized x-position
            #                     fontsize=7,  # Reduced font size
            #                     arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
            texts.append(key)
        else:
            texts.append('')
cbar = ax.scatter(periods, ftheta_infos, c=np.array(f_lows), cmap=cm.viridis, vmax=0.1, zorder=3)
periods = np.array(periods)
ftheta_infos = np.array(ftheta_infos)
texts = np.array(texts)
periods_to_annotate = periods[ftheta_infos < info_threshold]
fthetas_to_annotate = ftheta_infos[ftheta_infos < info_threshold]
texts_to_annotate = texts[ftheta_infos < info_threshold]
ax.set_xscale('log')
ax.set_yscale('function', functions=(forward, inverse))
ax.invert_yaxis()
ta.allocate(ax,periods_to_annotate,fthetas_to_annotate,
            texts_to_annotate,
            x_scatter=periods_to_annotate, y_scatter=fthetas_to_annotate,
            textsize=7, draw_lines=True, linecolor='k', avoid_label_lines_overlap=True,plot_kwargs={'alpha':0.1}, seed=2, direction='east', va='bottom')

colorbar = plt.colorbar(cbar, pad=0.0)
colorbar.set_label('Oblateness lower bound', rotation=270, labelpad=15, fontsize=12)
ax.set_ylabel("Information metric on oblateness and obliquity", fontsize=12)
ax.set_xlabel("Period (days)", fontsize=12)
plt.title("Aligned planet population", fontsize=14)
plt.tight_layout()
plt.savefig(f"param_space/parameter_space_plot_aligned_convex_hull.pdf")

#f vs theta
# Variable to store all annotation objects
texts = []
plt.figure(figsize=(5, 5))
for key, value in parameter_space.items():
    period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = value[0]
    
    # Annotate points with y-value less than 0.1

    if ftheta_information < 1.0:
        plt.errorbar(f_median, theta_median, xerr=[[f_median-f_low], [f_high-f_median]], yerr=[[theta_median-theta_low], [theta_high-theta_median]], fmt='o', color='b')
        # Randomize the x-position slightly for the annotation
        annotation_x = f_median + 0.1*f_high #random_x_offset
        annotation_y = theta_median #+ 0.1#+np.random.uniform(0.0, 0.01)

        text = plt.annotate(key, 
                            xy=(f_median, theta_median), 
                            xytext=(annotation_x, annotation_y),  # Randomized x-position
                            fontsize=7,  # Reduced font size
                            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
        texts.append(text)
plt.ylabel("Obliquity $\theta$")
plt.xlabel("Oblateness (f)")
plt.savefig(f"param_space/f_theta_aligned_convex_hull.pdf")


def bo_func(a,rs,inc):
    a = a*23454.8
    rs = rs*109.076
    return jnp.abs(a*jnp.cos(jnp.deg2rad(inc))/rs)

# Variable to store all annotation objects
texts = []
plt.figure(figsize=(5, 5))

first = True

for key, value in parameter_space.items():
    period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = value[0]
    
    a = df[df['name']==key]['pl_orbsmax'].values[0]
    rstar = df[df['name']==key]['st_rad'].values[0]
    inc = df[df['name']==key]['pl_orbincl'].values[0]
    imp_par = bo_func(a, rstar, inc)
    #if min_ess_bulk < 100 or min_ess_tail < 100 or max_r_hat > 1.1:
    plt.scatter(imp_par, ftheta_information, color='b')
    # Annotate points with y-value less than 0.1
    # if ftheta_information < 1.0:
    #     # Randomize the x-position slightly for the annotation
    #     annotation_x = imp_par + 0.1 #random_x_offset
    #     annotation_y = ftheta_information-0.01 #+ 0.1#+np.random.uniform(0.0, 0.01)

    #     text = plt.annotate(key, 
    #                         xy=(imp_par, ftheta_information), 
    #                         xytext=(annotation_x, annotation_y),  # Randomized x-position
    #                         fontsize=7,  # Reduced font size
    #                         arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
    #     texts.append(text)
plt.ylabel(r"95% Convex Hull Area on $f$ and $\theta$")
plt.xlabel("Impact Parameter")
plt.savefig(f"param_space/imppar_aligned_convex_hull.pdf")
plt.show()


########## MISALIGNED CASE ##########

#period vs convex hull area
# Variable to store all annotation objects
texts = []
periods = []
ftheta_infos = []
f_lows = []
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

info_threshold = 0.7
for key, value in parameter_space.items():
    if key != 'EPIC 248847494 b':

        period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = np.nanmean(value[1:], axis=0)
        
        periods.append(period)
        ftheta_infos.append(ftheta_information)
        f_lows.append(f_low)
        if min_ess_bulk < 10 or min_ess_tail < 10 or max_r_hat > 1.6:
            #ax.scatter(period, ftheta_information, c='r', s=50)
            print("Problematic planet: ", key, min_ess_bulk, min_ess_tail, max_r_hat)
        # Annotate points with y-value less than 0.1
        if ftheta_information < info_threshold:
    
            # annotation_x = period + 0.05*period 
            # annotation_y = ftheta_information-0.015
            # text = plt.annotate(key, 
            #                     xy=(period, ftheta_information), 
            #                     xytext=(annotation_x, annotation_y),  # Randomized x-position
            #                     fontsize=7,  # Reduced font size
            #                     arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
            texts.append(key)
        else:
            texts.append('')
cbar = ax.scatter(periods, ftheta_infos, c=np.array(f_lows), cmap=cm.viridis, vmax=0.1,zorder=3)
periods = np.array(periods)
ftheta_infos = np.array(ftheta_infos)
texts = np.array(texts)
periods_to_annotate = periods[ftheta_infos < info_threshold]
fthetas_to_annotate = ftheta_infos[ftheta_infos < info_threshold]
texts_to_annotate = texts[ftheta_infos < info_threshold]
ax.set_xscale('log')
ax.set_yscale('function', functions=(forward, inverse))
ta.allocate(ax,periods_to_annotate,fthetas_to_annotate,
            texts_to_annotate,
            x_scatter=periods_to_annotate, y_scatter=fthetas_to_annotate,
            textsize=7, draw_lines=True, linecolor='k', avoid_label_lines_overlap=True,plot_kwargs={'alpha':0.1}, seed=10, va='top')
colorbar = plt.colorbar(cbar, pad=0.0)
colorbar.set_label('Oblateness lower bound', rotation=270, labelpad=15, fontsize=12)
ax.invert_yaxis()
ax.set_ylabel("Information metric on oblateness and obliquity", fontsize=12)
ax.set_xlabel("Period (days)", fontsize=12)
plt.title("Misaligned planet population", fontsize=14)
plt.tight_layout()
plt.savefig(f"param_space/parameter_space_plot_misaligned_convex_hull.pdf")

#f vs theta
# Variable to store all annotation objects
texts = []
plt.figure(figsize=(5, 5))
for key, value in parameter_space.items():
    period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = value.T
    
    # Annotate points with y-value less than 0.1

    if np.nanmean(ftheta_information) < 1.0:
        plt.errorbar(f_median-0.1, theta_median-np.radians([0, 20, 40, 60, 80]), xerr=[f_median-f_low, f_high-f_median], yerr=[theta_median-theta_low, theta_high-theta_median], fmt='o', color='b')
        # Randomize the x-position slightly for the annotation
        annotation_x = np.nanmedian(f_median + 0.1*f_median) #random_x_offset
        annotation_y = np.nanmedian(theta_median) #+ 0.1#+np.random.uniform(0.0, 0.01)

        text = plt.annotate(key, 
                            xy=(np.nanmedian(f_median), np.nanmedian(theta_median)), 
                            xytext=(annotation_x, annotation_y),  # Randomized x-position
                            fontsize=7,  # Reduced font size
                            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
        texts.append(text)
plt.ylabel("Obliquity $\theta$")
plt.xlabel("Oblateness (f)")
plt.savefig(f"param_space/f_theta_misaligned_convex_hull.pdf")


# Variable to store all annotation objects
texts = []
plt.figure(figsize=(5, 5))

first = True
for key, value in parameter_space.items():
    period, ftheta_information, ftheta_contour_area, f_low, f_median, f_high, theta_low, theta_median, theta_high, min_ess_bulk, min_ess_tail, max_r_hat = np.nanmean(value[1:], axis=0)
    
    a = df[df['name']==key]['pl_orbsmax'].values[0]
    rstar = df[df['name']==key]['st_rad'].values[0]
    inc = df[df['name']==key]['pl_orbincl'].values[0]
    imp_par = bo_func(a, rstar, inc)
    #if min_ess_bulk < 100 or min_ess_tail < 100 or max_r_hat > 1.1:
    plt.scatter(imp_par, ftheta_information, color='b')
    # Annotate points with y-value less than 0.1
    # if ftheta_information < 1.0:
    #     # Randomize the x-position slightly for the annotation
    #     annotation_x = imp_par + 0.1 #random_x_offset
    #     annotation_y = ftheta_information-0.01 #+ 0.1#+np.random.uniform(0.0, 0.01)

    #     text = plt.annotate(key, 
    #                         xy=(imp_par, ftheta_information), 
    #                         xytext=(annotation_x, annotation_y),  # Randomized x-position
    #                         fontsize=7,  # Reduced font size
    #                         arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.0))
    #     texts.append(text)
plt.ylabel(r"95% Convex Hull Area on $f$ and $\theta$")
plt.xlabel("Impact Parameter")
plt.savefig(f"param_space/imppar_misaligned_convex_hull.pdf")
plt.show()