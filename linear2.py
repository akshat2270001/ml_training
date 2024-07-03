import pandas as pd
import numpy as np 

data=pd.read_csv("salary_data.csv")
print(data.head(10))

# salarydata=data.iloc[:,[4,5]]
# multiDimArray=np.array(salarydata['years of experience'])
# data.isnull.sum
dropNullSalary=data.dropna()

x=dropNullSalary.iloc[:,0:1].values
y=dropNullSalary.Salary.values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,
                                                 random_state=0)

from sklearn.linear_model import LinearRegression
leg=LinearRegression()
leg.fit(x,y)
pretict=leg.predict([[2.3]])
print(pretict) 


