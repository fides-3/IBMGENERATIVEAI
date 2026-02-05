# CREDIT CARD FRAUD DETECTION USING SVM

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
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
# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:,1:30]=StandardScaler().fit_transform(raw_data.iloc[:,1:30])
# changes the data matrix  to a numpy array
data_matrix=raw_data.values

# exclude the 'Time' feature which is usually column 0
X=data_matrix[:,1:30]

# target variable(Class) is in index 30
y=data_matrix[:,30]

# data normalization
X=normalize(X,norm='l1')

# train/test split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# build a decision tree classifier
# This balances the minority classes of y_train
w_train=compute_sample_weight('balanced',y_train)

dt=DecisionTreeClassifier(max_depth=4,random_state=35)
dt.fit(X_train,y_train,sample_weight=w_train)

# build a support vector machine with scikit learn
svm=LinearSVC(class_weight='balanced',random_state=31,loss='hinge',fit_intercept=False)
svm.fit(X_train,y_train)

# Run the following cell to compute the probabilities of the test samples belonging to the class of fraudulent transactions
# class 0 is the non-fraudulent class and class 1 is the fraudulent class
y_pred_dt=dt.predict_proba(X_test)[:,1]

# use ROC AUC score to eveluate the models ability to distinguish negative and posistive classes considering all probabilityy thresholds.The better the value the better the model is at distinguishing between the two classes
roc_auc_dt=roc_auc_score(y_test,y_pred_dt)
print("Decision Tree ROC-AUC score :{0: .3f}".format(roc_auc_dt))

# compute the decision function values for the test samples using the SVM model
y_pred_svm=svm.decision_function(X_test)
roc_auc_svm=roc_auc_score(y_test,y_pred_svm)
print("SVM ROC-AUC score :{0: .3f}".format(roc_auc_svm))