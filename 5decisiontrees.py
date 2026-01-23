import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data=pd.read_csv(path)

print(my_data.info())

# label encoder converts categorical variables to numerical labels....eg male=1 female=0
label_encoder=LabelEncoder()
my_data['Sex']=label_encoder.fit_transform(my_data["Sex"])

my_data['BP']=label_encoder.fit_transform(my_data["BP"])

my_data['Cholesterol']=label_encoder.fit_transform(my_data["Cholesterol"])
my_data

# To evaluate the correlation of the target variable with the input features, it will be convenient to map the different drugs to a numerical value. Execute the following cell to achieve the same.
custom_map={'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num']=my_data['Drug'].map(custom_map)
# my_data=my_data.drop(columns=['Drug'])

# print(my_data.corr())
    
# RECORDS OF EACH DRUG RECOMMENDATION
category_counts=my_data['Drug'].value_counts()
# .index means Drg a,Drug B   .values means their counts
plt.bar(category_counts.index,category_counts.values,color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title("Drug Distribution")
plt.show()

# TRAINING AND TESSTING
# FIRST MODEL THE INPUT FEATUTES AND TARGET VARIABLE
X=my_data.drop(columns=['Drug','Drug_num'])
y=my_data['Drug']

# testing 30%
X_trainset,X_testset,y_trainset,y_testset=train_test_split(X,y,test_size=0.3,random_state=32)

# define the Decision tree classifier as drugTree and train it with the training data
drugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
drugTree.fit(X_trainset,y_trainset)

# generate the predictions on the test set.
tree_prediction=drugTree.predict(X_testset)

#  accuracy of our model by using the accuracy metric
print("Decision tree's accuracy:",metrics.accuracy_score(y_testset,tree_prediction))

# VISUALIZE THE DECISION TREE
plot_tree(drugTree)
plt.show()