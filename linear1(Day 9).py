import numpy as np
import pandas as pd
data =pd.read_csv("tvmarketing.csv")
# print (data.head(10))\
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
#print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3, random_state=0)

from sklearn.linear_model import LinearRegression
leg=LinearRegression()
leg.fit(x_train,y_train)
predict=leg.predict(x_train)
print("prediction   ", predict)

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,predict,color='red')
plt.title("TV MARKETING DATA")
plt.xlabel("no. of product")
plt.ylabel("Sales data")
plt.show()
