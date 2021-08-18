import numpy as np #importing numpy and pandas libray
import pandas as pd
from sklearn.model_selection import train_test_split #importing train test split to train the data
from sklearn import svm #importing supported vector machine
import pickle #importing pickle
df=pd.read_csv("breast_cancer.csv") #firstly we load the file 
print(df.head()) #then we check the first five data
df.drop("id" , axis=1 , inplace = True ) #then we drop unneccassry coulmns
df.drop("Unnamed: 32" , axis = 1 , inplace = True )#drop unneccassry coulmns
print(df.head()) #then we check our dataset again
df["diagnosis"] = df["diagnosis"] . map ({"M" : 1 , "B" : 0}) #Since machine lerning only understands numbers we give labels to M and B
x=df.drop("diagnosis" , axis =1 , inplace = False) #then we split the data into x and y to train the module by them
y=df["diagnosis"] #y is our target
np.random.seed(42) # we write random seed so we can get the same values from random values
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2) #then we train the module
model=svm.SVC() #save the module in a variable model
model.fit(x_train,y_train) #train the module with x_train and y_train data
print(model.score(x_train,y_train)) #then we find the score of the model
print(model.score(x_test,y_test))
pickle.dump(model,open("Breast_cancer_classifer_model.pkl" , "wb")) # then we save this model in a file so it can be accessed 
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr() , cbar= True)
plt.show()

