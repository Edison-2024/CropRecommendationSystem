#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")
plt.style.use("dark_background")


# In[5]:


df = pd.read_csv("Crop_recommendation.csv")
df.head()


# In[6]:


df.isnull().sum()


# In[9]:


df.info()


# In[8]:


pd.set_option('display.float_format','{:.2f}'.format)
df.describe()


# In[10]:


df.columns


# In[11]:


feature_columns = [
    'N', 'P', 'K', 'temperature', 
    'humidity', 'ph', 'rainfall' 
]

for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")


# In[19]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0,strategy='mean',copy=False)
df[feature_columns] = fill_values.fit_transform(df[feature_columns])

for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")


# In[21]:


from sklearn.model_selection import train_test_split


X = df[feature_columns]
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[48]:


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")


# In[60]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

#Logistic Regression classifier
estimators = []
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train,y_train)
predicted_values = log_reg.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("Logistic Regression's Accuracy is: ", x)
estimators.append(('Logistic', log_reg))

#DecisionTree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
predicted_values = tree.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("DecisionTrees's Accuracy is: ", x*100)
estimators.append(('Tree', tree))

#SVM classifier
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
svm_clf = SVC(gamma='scale')
estimators.append(('SVM', svm_clf))
svm_clf.fit(X_train_norm,y_train)
predicted_values = svm_clf.predict(X_test_norm)
x = metrics.accuracy_score(y_test, predicted_values)
print("SVM's Accuracy is: ", x)

#NaiveBayes classifier
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train,y_train)
predicted_values = NaiveBayes.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("Naive Bayes's Accuracy is: ", x)
estimators.append(('NaiveBayes',NaiveBayes))

#RandomForest classifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train,y_train)
predicted_values = RF.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("RF's Accuracy is: ", x)
estimators.append(('RandomForest',RF))

#XGBoost classifier
XB = xgb.XGBClassifier()
XB.fit(X_train,y_train)
predicted_values = XB.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("XGBoost's Accuracy is: ", x)
estimators.append(('XGBoost',XGBoost))


voting = VotingClassifier(estimators=estimators,voting='hard')
voting.fit(X_train, y_train)
evaluate(voting, X_train, X_test, y_train, y_test)


# In[37]:


data = np.array([[14,8, 70, 21.603016, 6.3, 62.7, 14.91]])
prediction = voting.predict(data)
print(prediction)


# In[ ]:




