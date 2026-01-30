from __future__ import print_function;
import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import normalize;
from sklearn.metrics import mean_squared_error;
import warnings
warnings.filterwarnings("ignore")

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data=pd.read_csv(url)
print(raw_data.head(5))

# finding correlation between target varible(tip_amount) and other features
correlation_values=raw_data.corr()['tip_amount'].drop('tip_amount')
print(abs(correlation_values).sort_values(ascending=False)[:3])

correlation_values.plot(kind='barh',figsize=(10,6))
plt.show()

# DATA PREPROCESSING 
# . values converts it to a numpy array
# .astype converts the datatype to float32
y=raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the input features
proc_data=raw_data.drop(['tip_amount'],axis=1)

# X is a numpy array of the input features
X=proc_data.values

# normalize the input features
X=normalize(X,axis=1,norm='l1',copy=False)

# DATASET TRAIN/TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# BUILDING A TREE REGRESSOR MODEL
# import the decison tree regressor model from scikit-learn
# criterion is the function used to measure error
# max_depth we shall use 8
from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg=DecisionTreeRegressor(criterion='squared_error',max_depth=8,random_state=35)
dt_reg.fit(X_train,y_train)

y_pred=dt_reg.predict(X_test)
mse_score=mean_squared_error(y_test,y_pred)

print('MSE score:{0: .3f}'.format(mse_score))

r2_score=dt_reg.score(X_test,y_test)
print('R2 score :{0: .3f}'.format(r2_score))





