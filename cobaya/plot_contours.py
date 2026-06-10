import matplotlib
matplotlib.use('Agg')
from getdist import loadMCSamples, plots

# Lae täis-Planck nuisance-jooks
s_planck = loadMCSamples('chains/fullplanck')

g = plots.get_subplot_plotter(width_inch=7)
g.settings.legend_fontsize = 13
g.settings.axes_fontsize = 12
g.settings.axes_labelsize = 15

g.plot_2d([s_planck], 'w', 'wa', filled=True, colors=['#2c7fb8'])

# Marker LCDM-i jaoks (w0=-1, wa=0)
import matplotlib.pyplot as plt
ax = g.subplots[0,0]
ax.plot(-1, 0, marker='*', color='red', markersize=18, zorder=10)
ax.annotate(r'$\Lambda$CDM', xy=(-1, 0), xytext=(-1.15, 0.4),
            fontsize=13, color='red')

g.export('section9_w0wa.png')
print('Graafik salvestatud: section9_w0wa.png')
