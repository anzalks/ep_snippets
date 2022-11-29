import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize import curve_fit


#fitting function
def exp_func(x,a,b):
    return a*np.exp(b*x)

#recording data
xData = np.array ([1, 2, 3, 4, 5])
Data = np.array ([1, 9, 50, 300, 1500])

plt.plot(xData,yData,'bo',label= 'experimental-data')

 
