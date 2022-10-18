# Import the usual packages
import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# Create an array of 1000 random numbers
x = np.random.rand(1000)

# Create an array of 1000 random numbers
y = np.random.rand(1000)

# Draw a histogram of the data with 20 bins

plt.hist(x, bins=20, label='x', histtype='step')
plt.hist(y, bins=20, label='y', histtype='step')
plt.legend()
plt.show()
