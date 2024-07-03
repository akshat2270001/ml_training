import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("suv_data.csv")
print(data.head(10))

#changing the original data
edata=data.drop(["User ID","Gender"],inplace=True,axis=1)
# print(edata)

#independent data
X=data.drop(["Purchased"],axis=1)

#dependent data
y=data['Purchased']
# print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit(X_test)

from sklearn.naive_bayes import GaussianNB

gaus=GaussianNB()
gaus.fit(X_train,y_train)

#predict
predict=gaus.predict(X)
print(predict)

#matrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(y_test)
score=accuracy_score(y_train,predict)
print("Score::",score) #score::0.891666666667

con=confusion_matrix(y_test,predict)
print(con)

report=classification_report(y_test,predict)
print(report)






from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max+1,step=.01),
                  np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max+1,step=.01)         )

plt.contour(X1,X2,gaus.predict(
    np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,cmap=ListedColormap('red','green'))

plt.xlim(X1.min(),X2.max())
plt.ylim(X1.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
    
plt.title("Gauusian NB Algorithm salaried SUV data")
plt.xlabel("salaried data for suv ")
plt.ylabel("purchased suv car")
plt.show()
