# Your task will be to create a simple linear regression model from one of these features(columns) to predict CO2 emissions of unobserved cars based on that feature.  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
df.sample(5)

# prints the first 5 rows of the whole dataset
# print(df.sample(5))

# prints a statistical summary of the datasets
# print(df.describe())

# # select a few features(columns) that may be indicative of co2 emissions
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
# print(cdf.sample(5))


# visualize using a histogram
# viz=cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# scatter plot btn co2 emisions and fuel consumption combined(comb)
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("EMISSION")
plt.show()

# # scatter plot btn co2 emisions and engine size
# plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("ENGINESIZE")        
# plt.ylabel("EMISSION")
# plt.xlim(0,27)
# plt.show()

# plot cylinder against co2 emissions
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color="blue")
plt.xlabel("CYLINDERS")
plt.ylabel("EMISSION")
plt.show()

# We are using engine size to predict co2 emissions
# X is the input feature and y is the target variable
X=cdf.ENGINESIZE.to_numpy()
y=cdf.CO2EMISSIONS.to_numpy()



# create and train the model test=20% training=80%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print (X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Building the regression model
from sklearn import linear_model
regressor=linear_model.LinearRegression()
# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1,1),y_train)

# there is only one coefficient and we extract it fron 1 by 1 array
print('Coefficients:',regressor.coef_[0])
print('Intercept:',regressor.intercept_)

# You can visualize the goodness-of-fit of the model to the training data by plotting the fitted line over the data.
# The regression model is the line given by y = intercept + coefficient * x.
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regressor.coef_ * X_train + regressor.intercept_,'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# model evaluatiom
from sklearn.metrics import mean_absolute_error ,mean_squared_error,r2_score
y_pred=regressor.predict(X_test.reshape(-1,1))

# evaluate
print("Mean Absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean Squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2-score: %.2f" % r2_score(y_test, y_pred))