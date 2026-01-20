import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data=pd.read_csv(file_path)
print(data.head())

# DISTRIBUTION OF THE TARGET VARIABLE
sns.countplot(y="NObeyesdad",data=data)
plt.title("Distribution of obesity levels")
plt.show()

# FEATURE SCALING
# takes columns that are of data type float64
continuous_columns=data.select_dtypes(include=["float64"]).columns.to_list()

scaler=StandardScaler()
# a numpy array that takes only data to be scaled
scaled_features=scaler.fit_transform(data[continuous_columns])
# a dataframe that contains the scaled features
# get_feature_name_out is used to get the names of the scaled features after transformation
scaled_df=pd.DataFrame(scaled_features,columns=scaler.get_feature_names_out(continuous_columns))
# original continuous columns is dropped and the scaled dataframe is concatenated
# axis 1=works in columns
# axis 0=works in rows
scaled_data=pd.concat([data.drop(columns=continuous_columns),scaled_df],axis=1)

# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encoding the target variable
# puts the target variable as category
# .cat.codes converts categories to numerical labels eg..normal weight=0,overweight =1
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

# preparing the final data set
X=prepped_data.drop("NObeyesdad",axis=1)
y=prepped_data['NObeyesdad']

# splitting the data
# stratify=y the data is split in such a way that the distribution/proportions of target variable is maintained in both train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# TRAINING LOGISTIC REGRESSION MODEL USING ONE VS ALL APPROACH
model_ova=LogisticRegression(multi_class="ovr",max_iter=1000)
model_ova.fit(X_train,y_train)

# prediction
y_pred_ova=model_ova.predict(X_test)

# Evaluation metrics for OVA
print("One-vs-all strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test,y_pred_ova),2)}")


# TRAINING LOGISTIC REGRESSION MODEL USING ONE VS ONE APPROACH
model_ovo=OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train,y_train)

# prediction
y_pred_ovo=model_ovo.predict(X_test)
print("One Vs One strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test,y_pred_ovo),2)}")