import matplotlib.pyplot as plt
import numpy as np

import ipdb; ipdb.set_trace()
length = np.random.random(10)
length.sort()
time = np.random.random(10)
time.sort()
slope, intercept = np.polyfit(np.log(length), np.log(time), 1)
print(slope)
plt.loglog(length, time, '--')
plt.show()
