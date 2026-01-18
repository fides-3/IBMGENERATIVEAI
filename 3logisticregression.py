# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss 
import warnings 
warnings.filterwarnings("ignore")


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

churn_df=pd.read_csv(url)
# DATA PREPROCESSING
churn_df=churn_df[["tenure","age","address","income","ed","employ","equip","churn"]]
print(churn_df.head(5))
# changes datatype to int
churn_df["churn"]=churn_df["churn"].astype("int")

# churn column is out target variable so it is y...the otheders are input features and they take x 
# changes the df to array
X=np.asarray(churn_df[["tenure","age","address","income","ed","employ","equip"]])
print(X[0:5])
y=np.asarray(churn_df["churn"])
print(y[0:5])

# standardize the dataset
X_norm=StandardScaler().fit(X).transform(X)
print(X_norm[0:5])

# Splitting the dataset
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=42)

# Logistice regression classifier modelling
LR=LogisticRegression().fit(X_train,y_train)

# Predicting churn parameter of the test model
yhat=LR.predict(X_test)
yhat[:10]
# Predicting the probability of churn
yhat_prob=LR.predict_proba(X_test)
yhat_prob[:10]

# Since the purpose here is to predict the 1 class more acccurately, you can also examine what role each input feature has to play in the prediction of the 1 class. Consider the code below.

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

log_loss(y_test, yhat_prob)