#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Windy | AirQualityGood | Hot | PlayTennis
# No         No               No       No
# Yes        No               Yes      Yes
# Yes        Yes              No       Yes
# Yes        Yes              Yes      No
import math
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

df = pd.DataFrame(pd.read_csv("C:/Users/Steve/Desktop/Code/Tennis.csv"))

dict = {}

for column in df:
    dict[df[column].name] = calcEntropy(df[column].values)
    

def calcEntropy(arr):
    size = len(arr)
    count = 0
    for i in arr:
        if i == 'yes':
            count = count + 1   
    return -(count/size)*math.log(count/size,2)-((size-count)/size)*math.log(size-count/size,2)
    


# In[ ]:




