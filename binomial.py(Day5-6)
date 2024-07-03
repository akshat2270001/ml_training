import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import binom

n=12220;p=8.5
values=np.arange(1,n+1)
pmf=binom.pmf(values,n,p)

#print(pmf)
plt.bar(values,pmf)
plt.xlabel("Data Diagram")
plt.ylabel("Probability Data")
plt.show()
