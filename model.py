#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\KomalA\Downloads\breast_cancer.csv")
data.head()


# In[3]:


#drop the unneccessary columns to reduce curse of dimensionality
data.drop(['id','Unnamed: 32'],axis=1, inplace=True)

data.head()


# In[4]:


data["diagnosis"].replace({"M": 0, "B": 1}, inplace=True)
data.head()


# In[5]:


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# In[6]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[7]:


#Correlation with output variable
cor_target = abs(cor["diagnosis"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.7]
relevant_features


# In[8]:


y = data['diagnosis']


# In[9]:


X = data[['area_mean', 'radius_mean', 'perimeter_mean','concave points_mean']].copy()


# In[10]:


X


# In[11]:


#Normalization
X = (X - np.min(X))/(np.max(X)-np.min(X)).values


# In[12]:


X


# In[13]:


X = X.rename(columns={'area_mean': 'Area', 'radius_mean': 'Radius', 'perimeter_mean': 'Perimeter', 'concave points_mean': 'concave_points'})


# In[14]:


X


# In[15]:


#model training using cross validation
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split



scores = []
model = svm.SVC(kernel='linear')
cv = StratifiedKFold(n_splits=10, random_state=None) 
for train_index, test_index in cv.split(X,y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index],y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))


# In[16]:


#calculating model accuracy
Accuracy=np.mean(scores)
print(Accuracy)


# In[18]:


import pickle
pickle_out=open(r"C:\Users\KomalA\Documents\deploy\breast-cancer-app\cancer-app.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()


# In[22]:


model.predict([[0.731113,0.620776,0.668310,0.912027]])


# In[ ]:




