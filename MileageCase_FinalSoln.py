# ------Mileage of Car: Project-------- #

import pandas as pd 

car_data = pd.read_csv("F:/WORK/SkillEdge/Car_Mileage_Project_Case_Study/T6_Luxury_Cars.csv")


#There are lot of string value var in dataset which have to be converted to numerical
#values for applying machine learing algoritm. Hence, we will now convert string var 
#to numerical var.
car_data.info()

pd.get_dummies(car_data["Make"])
pd.get_dummies(car_data["Make"],drop_first=True)
maker_dummy = pd.get_dummies(car_data["Make"],drop_first=True)

pd.get_dummies(car_data["Type"])
pd.get_dummies(car_data["Type"],drop_first=True)
type_dummy = pd.get_dummies(car_data["Type"],drop_first=True)

pd.get_dummies(car_data["Origin"])
pd.get_dummies(car_data["Origin"],drop_first=True)
origin_dummy = pd.get_dummies(car_data["Origin"],drop_first=True)

pd.get_dummies(car_data["DriveTrain"])
pd.get_dummies(car_data["DriveTrain"],drop_first=True)
DriveTrain = pd.get_dummies(car_data["DriveTrain"],drop_first=True)

#Now, lets concatenate these dummy var columns in our dataset.
car_data = pd.concat([car_data,maker_dummy,type_dummy,origin_dummy,DriveTrain],axis=1)
car_data.head(5)

#droping columns of whoes dummy variable is created
car_data.drop(["Make","Type","Origin","DriveTrain","Model"],axis=1,inplace=True)
car_data.head()

#Taking dependent and independent variables
x=car_data.drop("MPG (Mileage)",axis=1)
y=car_data["MPG (Mileage)"]

#Splitting the dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#this is giving accurecy of 83.6%

#Coefficient
regressor.coef_

# Intercept
regressor.intercept_

#----To Improve the accuracy of the model, lets go with Backward ELimination Method &
# rebuild the logisitc model again with few independent variables--------
car_data_1 = car_data
car_data_1.head(5)

#--------------------------Backward Elimination--------------------------------
#Backward elimination is a feature selection technique while building a machine learning model. It is used
#to remove those features that do not have significant effect on dependent variable or prediction of output.

#Step: 1- Preation of Backward Elimination:
#Importing the library:
import statsmodels.api as sm

#Adding a column in matrix of features:
x1=car_data_1.drop("MPG (Mileage)",axis=1)
y1=car_data_1["MPG (Mileage)"]

import numpy as nm
x1 = nm.append(arr = nm.ones((426,1)).astype(int), values=x1, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt=x1[:,:]

#for fitting the model, we will create a regressor_OLS object of new class OLS of statsmodels library. 
#Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


#In the above summary table, we can clearly see the p-values of all the variables. 
#And remove the ind var with p-value greater than 0.05
x_opt=x1[:, [0,2,3,4,8,16,21,27,37,38,41,44,45,46,47,48,51]]
regressor_OLS=sm.OLS(endog=y1, exog=x_opt).fit()
regressor_OLS.summary()


#-------Building Multiple Regression model using Significant ind var

#Extracting Independent and dependent Variable  
x_BE= x_opt
y_BE= y1
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

y_pred= regressor.predict(x_BE_test)

from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)

#Accuracy = 85.07%
#With the help of Sign Ind var, the accuracy increased from 83.6% to 85.07%

#Coefficient
regressor.coef_
# Intercept
regressor.intercept_
