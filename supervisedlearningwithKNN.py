# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new or unknown case.

# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.

# The target field, called custcat, has four possible service categories that correspond to the four customer groups, as follows:

# Basic Service
# E-Service
# Plus Service
# Total Service
# Our objective is to build a classifier to predict the service category for unknown cases. We will use a specific type of classification called K-nearest neighbors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print(df.head())

# looks at count of each category
# 3-plusservice 1-basic service 4-totalservice  217-eservice
print(df['custcat'].value_counts())

# correlation 
correlation_matrix=df.corr()

# plt.figure creates a new figure with the specified size (10,8) inches. This sets up the canvas for the heatmap that will be plotted next.
# if you do sns only without figure..the heatmap will looked squished
plt.figure(figsize=(9,7))

# heatmap creates a color coded matrix of values annot true -writes the correlation values inside each square cmap coolwarm-sets the color scheme fmt .2f fromats the annotations to 2dp linewidths-adds small lines between each square 
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# The following code snippet will give us a list of features sorted in the descending order of their absolute correlation values with respect to the target field.
# retire and gender have least effect on custcat
correlation_values=abs(df.corr()['custcat'].drop('custcat').sort_values(ascending=False))
print(correlation_values)

# separate in[ut and target features
# axis=1 means column custcat
X=df.drop('custcat',axis=1)
y=df['custcat']

# standardize the data
# Since normalization scales each feature to have zero mean and unit variance, it puts all features on the same scale (with no feature dominating due to its larger range
X_norm=StandardScaler().fit_transform(X)

# Train  test split
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.3,random_state=4)

# KNN CLASSIFICATION
# Training
k=3
knn_classifier=KNeighborsClassifier(n_neighbors=k)
knn_model=knn_classifier.fit(X_train,y_train)

# predicting
yhat=knn_model.predict(X_test)

# accuracy .shows how close the actual labels and predicted labels are matched in the test set
print("Test set Accuracy:",accuracy_score(y_test,yhat))


# Choosing the correct value of k
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


# Plot the model accuracy for a different number of neighbors
# plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 
