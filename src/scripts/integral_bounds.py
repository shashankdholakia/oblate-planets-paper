from jax0planet.numpy_src import draw_oblate
import paths

fig, ax = draw_oblate(0.7, 0.8, 0.8, 0.4)
fig.savefig(paths.figures / f"oblate_planet.pdf", dpi=300)