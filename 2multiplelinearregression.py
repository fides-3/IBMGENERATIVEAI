import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df.sample(5))
print(df.describe())

# drop categoricals
df=df.drop(["MODELYEAR","MODEL","VEHICLECLASS","MAKE","TRANSMISSION","FUELTYPE"],axis=1)

print(df.corr())

# ENGINESIZE and CYLINDERS are highly correlated..ENGINESIZE is more correlated to the target CO2EMISSIONS so we drop CYLINDERS
# All the four fuel consumption features are highly correlated.FUELCONSUMPTION_COMB_MPG is more correlated to the target hence we drop the others
df=df.drop(['CYLINDERS',"FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB"],axis=1)
print(df.head(5))

# input features are ENGINE SIZE and FUELCONSUMPTION_COMB_MPG targget variable is CO2EMISSIONS
# extract input features
X=df.iloc[:,[0,1]].to_numpy()
y=df.iloc[:,[2]].to_numpy()

# standardize the input feature 
from sklearn import preprocessing
std_scaler=preprocessing.StandardScaler()
X_std=std_scaler.fit_transform(X)

# create and train datasets 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.2,random_state=42)

from sklearn import linear_model
regressor=linear_model.LinearRegression()
regressor.fit(X_train,y_train)

coef_=regressor.coef_
intercept_=regressor.intercept_

print("coefficients:",coef_)
print("Intercept:",intercept_)


# RESEARCH ON THIS MORE
# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)

# PLOT THE REGRESSION LINE FOR BOTH FEATURES
# ENGINE SIZE
plt.scatter(X_train[:,0],y_train,color="blue")
# coef[0,0]..means slope of first output and first input feature...input feature is engine size....output is CO2 emissions
# intercept[0] is the y intercept
plt.plot(X_train[:,0],coef_[0,0]*X_train[:,0] +intercept_[0],"-r")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()


# FUELCONOSUUMPTON_COMB_MPG
plt.scatter(X_train[:,1],y_train,color="blue")
# coef[0,1] means slope of first output and second input feature...input feature is FUELCONSUMPTION_COMB_MPG....output is CO2 emissions
plt.plot(X_train[:,1],coef_[0,1]*X_train[:,1]+intercept_[0],"-r")
plt.xlabel("Fuel consumption Comb Mpg")
plt.ylabel("CO2 Emissions")
plt.show()










































# EX 1
# from sklearn import linear_model

# Take only the first feature from X_train
# X_train_1 = X_train[:, 0]

# Create model
# regressor_1 = linear_model.LinearRegression()

# Train using one feature (must be 2D)
# regressor_1.fit(X_train_1.reshape(-1, 1), y_train)

# Get parameters
# coef_1 = regressor_1.coef_
# intercept_1 = regressor_1.intercept_

# Display results
# print("Coefficients:", coef_1)
# print("Intercept:", intercept_1)


# EX2
# plt.scatter(X_train_1, y_train,  color='blue')
# plt.plot(X_train_1, coef_1[0] * X_train_1 + intercept_1, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")

# EX3
# X_test_1 = X_test[:,0]
# plt.scatter(X_test_1, y_test,  color='blue')
# plt.plot(X_test_1, coef_1[0] * X_test_1 + intercept_1, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("CO2 Emission")

# EX4
# Click here for the solution
# X_train_2 = X_train[:,1]
# regressor_2 = linear_model.LinearRegression()
# regressor_2.fit(X_train_2.reshape(-1, 1), y_train)
# coef_2 =  regressor_2.coef_
# intercept_2 = regressor_2.intercept_
# print ('Coefficients: ',coef_2)
# print ('Intercept: ',intercept_2)

# EX5
# X_test_2 = X_test[:,1]
# plt.scatter(X_test_2, y_test,  color='blue')
# plt.plot(X_test_2, coef_2[0] * X_test_2 + intercept_2, '-r')
# plt.xlabel("combined Fuel Consumption (MPG)")
# plt.ylabel("CO2 Emission")