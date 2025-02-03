<h3>Date:</h3>
<h2>Exp: 09</h2>

# Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the csv file
2. Import necessary libraries 
3. Get the Prediction value and accuracy value
4. End the Program

## Program:


Program to implement the SVM For Spam Mail Detection..

Developed by: Sabari Akash A 

RegisterNumber:  212222230124
```py
import pandas as pd
data=pd.read_csv('spam.csv',encoding='windows-1252')
data.head()
data.shape

x=data["v2"].values
y=data["v1"].values
x.shape
y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40)
x_train
x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics  import accuracy_score,confusion_matrix,classification_report

accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img src=image.png width=700 height=300>

### X_Shape
![output](image-5.png)
### Y_Shape
![output](image-6.png)

<img src=image-8.png width=300 height=150>

### X_train.shape
![output](image-9.png)
### Y-Predict
![output](image-10.png)
###  Accuracy
![output](image-11.png)
### Confusion matrix
![output](image-12.png)
### Classification report
<img src=image-13.png width=400 height=200>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
