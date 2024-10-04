NUM_CORES = 1
import argparse
import pandas as pd

import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.stats import circmean

jax.config.update("jax_enable_x64", True)

import arviz as az



def main(planet_id):
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
    #theta_id in the range 0-9
    theta_arr = jnp.linspace(0, jnp.pi, 10)
    plt.clf()
    # Create a subplot with polar projection
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,4.8))
    plt.subplots_adjust(left=0, bottom=-1, right=1, top=1, wspace=0, hspace=0)

    ax.set_theta_direction('clockwise')
    if str(df_noise_pop['Planet Name'][i]) in list(planet_pop['name']):
        planet_name = df_noise_pop['Planet Name'][i]
        index = planet_pop.loc[planet_pop['name'] == planet_name].index[0]
        print(planet_name)
        for index, true_theta in enumerate(np.linspace(0,np.pi,10)[:-5]):
            
            theta_deg = int(np.round(np.degrees(true_theta)))
            print("Theta:", true_theta)
                
            inf_data = az.from_netcdf(f"mcmc_sim/posteriors/average_radius_NUTS_{theta_deg}_{planet_name.replace(" ", "")}.h5")
            
            #calculate the 95% percentiles for f and theta from the MCMC chains
            f_samples = np.concatenate(inf_data.posterior.f.to_numpy(), axis=0)
            theta_samples = np.concatenate(inf_data.posterior.theta.to_numpy(), axis=0)
            f_mcmc = np.percentile(f_samples, q=(5., 95.))
            theta_mcmc = np.percentile(theta_samples, q=(5., 95.))
            
            # Create a scatter plot
            ax.scatter(theta_samples, f_samples*100, alpha=0.1, s=1, c=colors[index], zorder=1+2.5)
            xerr = np.abs(np.array([theta_mcmc]).T - true_theta)
            print(np.degrees(xerr))
            yerr=np.abs(np.array([f_mcmc]).T-0.1)
            ax.scatter(circmean(theta_samples), np.mean(f_samples)*100, c=colors[index], linewidth=1, zorder=1+2.5, s=20, label=f'$\\theta={int(theta_deg)}\\degree$')
            ax.scatter(circmean(theta_samples), np.mean(f_samples)*100, c=colors[index], edgecolor='black', linewidth=0.5, zorder=3+2.5, s=20)
            #hack to add a border around the error bars to make them more visible
            ax.errorbar(circmean(theta_samples), np.mean(f_samples)*100, xerr=xerr+0.005, yerr=yerr*100+0.05, fmt="", ecolor='k', zorder=1.9+2.5,elinewidth=2.1)
            ax.errorbar(circmean(theta_samples), np.mean(f_samples)*100, xerr=xerr, yerr=yerr*100, fmt="", c=colors[index], zorder=2+2.5)   
            
        ax.grid(which='both', axis='both', zorder=-1.0)
        ax.set_thetamin(-90)
        ax.set_theta_zero_location('N')
        ax.set_thetamax(90)
        ax.tick_params(labelleft=True, labelright=True)
        start, end = ax.get_ylim()
        ax.yaxis.set_major_locator(MultipleLocator(10))

        # For the minor ticks, use no labels; default NullFormatter.
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        #ax.yaxis.get_ticklocs(minor=True)
        #ax.minorticks_on()
        
        ax.yaxis.set_tick_params(rotation=0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(-0.1, 1))
        plt.tight_layout()
        plt.savefig(f"polar_plots/polar_plot_{planet_name.replace(' ', '')}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create simulation scripts for given planet_id.')
    parser.add_argument('--planet_id', type=int, required=True, help='The ID of the planet.')

    args = parser.parse_args()

    main(args.planet_id)
