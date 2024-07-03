import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("apples_and_oranges_Classification.csv")
# print(data.head(5))
sum=pd.isna(data)
# print(sum)
# print(data)

X=data.iloc[:,0:2].values
y=data.iloc[:,-1].values
# print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)
print(X_test)

knn=KNeighborsClassifier()
knn.fit(X,y)
predict=knn.predict([[20,2.7]])
# predict=knn.predict([[60,3.5]])
predict=knn.predict(X)
print(predict)

#data visualization
xPlot=data.Weight
yPlot=data.Size
sns.scatterplot(x=xPlot,y=yPlot,hue=data.Class)
plt.xlabel("Weight")
plt.xlabel("size")
plt.show()
