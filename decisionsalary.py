# import numpy as np 
# import pandas as pd 

# data=pd.read_csv("salary_prediction_data.csv")
# d=data.dropna(inplace=True)

# x=data.iloc[:,4]
# # print (x)
# y=data.iloc[:,-1]
# # print(y)

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# from sklearn.ensemble import RandomForestClassifier

# forest=RandomForestClassifier(n_estimators=100,criterion="gini")
# fit=forest.fit(x_train,y_train)
# # forest.fit(x_train,y_train)
# # forest.transform(x,y)
# #predict=forest.predict([[2.3,3.3,3.2,1.7]])
# predict=forest.predict([[3]])
# print(predict)

# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# score=accuracy_score(y,predict)
# print("Accuracy Score ::",score)
# con=confusion_matrix(y,predict)
# print(con)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Salary Data.csv")
d=data.dropna(inplace=True)
# print(d)
# print(data)

X=data.iloc[:,[0,5]].values
# print(X)
y=data.iloc[:,-1].values
# print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=10)
forest.fit(X_train,y_train)
predict=forest.predict([[24,1]])
# predict=forest.predict(X)
print(predict)