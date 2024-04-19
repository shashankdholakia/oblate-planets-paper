from jax0planet.numpy_src import draw_oblate

fig, ax = draw_oblate(0.7, 0.8, 0.8, 0.4)
fig.savefig("oblate_planet.pdf", dpi=300)