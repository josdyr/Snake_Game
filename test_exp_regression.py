# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly as py
import plotly.graph_objs as go

# MatPlotlib
import matplotlib.pyplot as plt
from matplotlib import pylab

# Scientific libraries
import numpy as np
from scipy.optimize import curve_fit


py.tools.set_credentials_file(username='josdyr', api_key='h6Tm3Q06lY2Xr3BVlTIw')



x = np.array([399.75, 989.25, 1578.75, 2168.25, 2757.75, 3347.25, 3936.75, 4526.25, 5115.75, 5705.25])
y = np.array([109,62,39,13,10,4,2,0,1,2])

def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c


popt, pcov = curve_fit(exponenial_func, x, y, p0=(1, 1e-6, 1))

xx = np.linspace(300, 6000, 1000)
yy = exponenial_func(xx, *popt)

plt.plot(x,y,'o', xx, yy)
pylab.title('Exponential Fit')
ax = plt.gca()
ax.set_facecolor((0.898, 0.898, 0.898))
fig = plt.gcf()
py.plotly.plot_mpl(fig, filename='Exponential-Fit-with-matplotlib')
plt.show()
