# annova &its principle

#from scipy.stats import f_oneway
from scipy.stats import *
population1=[10,20,30,40,50]
population2=[70,80,40,50,60]
population3=[10,20,30,30,20]
population4=[40,50,20,30,10]

e=f_oneway(population1,population2,population3,population4)
print(e)
