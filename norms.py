import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
np.random.seed(0)
n=np.random.normal(size=1000)
sns.displot(n,kde=True)
plt.show()