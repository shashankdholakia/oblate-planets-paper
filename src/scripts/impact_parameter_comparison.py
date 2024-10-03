import arviz as az
import numpy as np
import pandas as pd

from chainconsumer import ChainConsumer, Chain, PlotConfig, ChainConfig
from chainconsumer.plotting import plot_contour
import matplotlib.pyplot as plt
import paths
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset



def load_posterior_samples(ppm, bo):
    file_path = paths.data / f"average_radius_NUTS_{ppm}ppm_bo_{bo}.h5"
    inf_data = az.from_netcdf(file_path)
    return inf_data

def plot_posteriors():
    # Load posterior samples
    posterior_01 = load_posterior_samples(10, 0.1)
    posterior_03 = load_posterior_samples(10, 0.3)
    posterior_05 = load_posterior_samples(10, 0.5)
    posterior_07 = load_posterior_samples(10, 0.7)

    # Extract the samples for 'f' and 'theta'
    f_samples_01 = np.concatenate(posterior_01.posterior.f.to_numpy(), axis=0).T
    theta_samples_01 = np.degrees(np.concatenate(posterior_01.posterior.theta.to_numpy(), axis=0))
    f_samples_03 = np.concatenate(posterior_03.posterior.f.to_numpy(), axis=0).T
    theta_samples_03 = np.degrees(np.concatenate(posterior_03.posterior.theta.to_numpy(), axis=0))
    f_samples_05 = np.concatenate(posterior_05.posterior.f.to_numpy(), axis=0).T
    theta_samples_05 = np.degrees(np.concatenate(posterior_05.posterior.theta.to_numpy(), axis=0))
    f_samples_07 = np.concatenate(posterior_07.posterior.f.to_numpy(), axis=0).T
    theta_samples_07 = np.degrees(np.concatenate(posterior_07.posterior.theta.to_numpy(), axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Create ChainConsumer instance
    nuts_01 = Chain(samples=pd.DataFrame(np.array([f_samples_01, theta_samples_01]).T, columns=[r'$f$', r'$\theta$']), name=r"$b_o = 0.1$", color="#1b9e77", shade_alpha=0.3)
    nuts_03 = Chain(samples=pd.DataFrame(np.array([f_samples_03, theta_samples_03]).T, columns=[r'$f$', r'$\theta$']), name=r"$b_o = 0.3$", color="#d95f02", shade_alpha=0.3)
    nuts_05 = Chain(samples=pd.DataFrame(np.array([f_samples_05, theta_samples_05]).T, columns=[r'$f$', r'$\theta$']), name=r"$b_o = 0.5$", color="#7570b3", shade_alpha=0.3)
    nuts_07 = Chain(samples=pd.DataFrame(np.array([f_samples_07, theta_samples_07]).T, columns=[r'$f$', r'$\theta$']), name=r"$b_o = 0.7$", color="#e7298a", shade_alpha=0.3)
    #plot_contour(ax, chain=nuts_01, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1., show_legend=True, label='test'))
    #plot_contour(ax, chain=nuts_03, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))
    #plot_contour(ax, chain=nuts_05, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))
    #plot_contour(ax, chain=nuts_07, px='f',py='theta', config=PlotConfig(serif=True, max_ticks=5, spacing=1.))
    
    #ax.set_xlabel(r'$f$')
    #ax.set_ylabel(r'$\theta$')

    # Set labels
    #plt.xlabel('f')
    #plt.ylabel('theta')
    c = ChainConsumer()
    c.add_chain(nuts_01)
    c.add_chain(nuts_03)
    c.add_chain(nuts_05)
    c.add_chain(nuts_07)
    c.set_plot_config(
        PlotConfig(
            plot_hists=False,
    ))
    fig = c.plotter.plot()
    ax = fig.axes[0]
    axins = zoomed_inset_axes(ax, zoom=2, loc='center right', borderpad=1.0)
    plot_contour(axins, chain=nuts_05, px=r'$f$',py=r'$\theta$', config=PlotConfig(serif=True, max_ticks=5, spacing=1., show_legend=True, label='test'))
    plot_contour(axins, chain=nuts_07, px=r'$f$',py=r'$\theta$', config=PlotConfig(serif=True, max_ticks=5, spacing=1., show_legend=True, label='test'))
    #axins.yaxis.set_visible(False)
    #axins.xaxis.set_visible(False)  
    xlim = axins.get_xlim()
    ylim = axins.get_ylim()
    #plot_contour(axins, chain=nuts_01, px=r'$f$',py=r'$\theta$', config=PlotConfig(serif=True, max_ticks=5, spacing=1., show_legend=True, label='test'))
    #plot_contour(axins, chain=nuts_03, px=r'$f$',py=r'$\theta$', config=PlotConfig(serif=True, max_ticks=5, spacing=1., show_legend=True, label='test'))
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    for s in ['top', 'bottom', 'left', 'right']:
        axins.spines[s].set(color='grey', lw=1, linestyle='solid', alpha=1.0)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='gray', alpha=0.5)
    # Customize y-axis tick labels to include degree symbol
    y_ticks = ax.get_yticks()
    y_tick_labels = [f"{tick:.0f}°" for tick in y_ticks]
    x_ticks = ax.get_xticks()
    x_tick_labels = [f"{tick:.2f}" for tick in x_ticks]
    ax.set_yticklabels(y_tick_labels, fontsize=14)
    ax.set_xticklabels(x_tick_labels, fontsize=14)
    ax.set_ylabel(r'Obliquity ($\theta$)', fontsize=16)  # Set the font size for the y-axis label
    ax.set_xlabel(r'Oblateness ($f$)', fontsize=16)
    
    y_ticks = axins.get_yticks()
    y_tick_labels = [f"{tick:.1f}°" for tick in y_ticks]
    axins.set_yticklabels(y_tick_labels)
    # Show the plot
    plt.savefig(paths.figures/"impact_parameter_comparison.pdf")

if __name__ == "__main__":
    plot_posteriors()