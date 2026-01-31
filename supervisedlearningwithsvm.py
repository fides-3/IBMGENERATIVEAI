# CREDIT CARD FRAUD DETECTION USING SVM

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

# load the dataset

raw_data=pd.read_csv("creditcard.csv")
print(raw_data.head(5))
# get the set of distinct classes
labels=raw_data['Class'].unique()
print("Distinct classes in the dataset:",labels)

# get the count of each class
# .values changes it to a numpy array
sizes=raw_data["Class"].value_counts().values
print("Count of each class:",sizes)

# plot the class value counts
# use matplotlib
fig,ax=plt.subplots()
ax.pie(sizes,labels=labels,autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

correlation_values=raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh',figsize=(10,6))
plt.show()

# DATA PREPROCESSING 
