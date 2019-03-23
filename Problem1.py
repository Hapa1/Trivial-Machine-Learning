#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Upscaling code taken from https://elitedatascience.com/imbalanced-classes

df = pd.DataFrame(pd.read_csv("C:/Users/Steve/filename.csv"))
df.drop(['Id','Unnamed: 0'], axis=1, inplace=True) #remove id column 
for index, row in df.iterrows():
    if row["Class"] == "benign":
        df.at[index, "Class"] = 0
    else:
        df.at[index, "Class"] = 1
df_majority = df[df.Class==0]
df_minority = df[df.Class==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=444,  # to match majority class
                                 random_state=999) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

y = df_upsampled.Class
y=y.astype('int') #convert Column 'Class' type to number from object 
X = df_upsampled.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999) #Split Data
clf_1 = LogisticRegression().fit(X_train, y_train) # Train model
pred_y_1 = clf_1.predict(X_test) # Predict on testing set
print( accuracy_score(y_test, pred_y_1) ) # Print accuracy?


# In[ ]:





# In[ ]:





# In[ ]:




