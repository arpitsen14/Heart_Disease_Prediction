import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data=pd.read_csv("C:/Users/Gaming 3/Desktop/heart.csv")

#gathering of data
#target values are the desired output values.

Y=data.target.values
X=data.drop(['target'],axis=1)

#preparing data
#spliting of data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


#logistic regression model

model=LogisticRegression()
model.fit(X_train,Y_train) #training model

#Accuracy Score of training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy Of training Data:',training_data_accuracy)
#Accuracy score of testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy Of training Data:',test_data_accuracy)

#Building a predictive system
input_data=(41,0,1,130,204,0,0,172,0,1,4,2,0)
#change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for one row
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print('The Person Does Not have heart Disease')
else:
    print('The Person has heart Disease')

