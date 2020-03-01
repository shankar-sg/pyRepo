#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


# In[104]:


#
# Load the train and test files
#
InDF = pd.read_csv("C:\\Training\\R\\kaggle\\titanic\\train.csv")


# In[106]:


#
# Find the columns that have na elements
# They need to be cleaned up
#
InDF.isna().sum()


# In[107]:


#
# Find average age by 'Pclass' and 'Sex' ... this can be used to set missing Age
#

#TestDF.groupby(['Pclass','Sex'])['Age'].mean()
InDF.groupby(['Pclass','Sex'])['Age'].mean()


# In[108]:


#
# Average age of passenger by "Pclass" and "Sex" is found above. Set AGE NAs as this average AGE  
# This is better than fitting the same average value for all missing NAs
#

InDF.loc[(InDF['Pclass']==1) & (InDF['Sex']=='female') & (InDF['Age'].isna()),'Age'] = 34
InDF.loc[(InDF['Pclass']==2) & (InDF['Sex']=='female') & (InDF['Age'].isna()),'Age'] = 29
InDF.loc[(InDF['Pclass']==3) & (InDF['Sex']=='female') & (InDF['Age'].isna()),'Age'] = 22
InDF.loc[(InDF['Pclass']==1) & (InDF['Sex']=='male') & (InDF['Age'].isna()),'Age'] = 41
InDF.loc[(InDF['Pclass']==2) & (InDF['Sex']=='male') & (InDF['Age'].isna()),'Age'] = 31
InDF.loc[(InDF['Pclass']==3) & (InDF['Sex']=='male') & (InDF['Age'].isna()),'Age'] = 26

InDF.isna().sum()


# In[109]:


#
# Subset columns that get to be a part of the model
#
InDF2 = InDF[['Pclass','Age','Sex','Survived']]
#
# Expand factors of 'Sex' using dummies or one hot encoder
#
InDF2 = pd.get_dummies(InDF2,columns=['Sex'])
 

X = InDF2.loc[:,InDF2.columns !='Survived']
y = InDF2.loc[:,InDF2.columns =='Survived']


# In[110]:


train_X,test_X,train_y, test_y = train_test_split(X,y)


# In[111]:


model = RandomForestClassifier()
model.fit(train_X,train_y)
pred_y = model.predict(test_X)


# In[112]:


mean_absolute_error(pred_y,test_y)


# In[113]:


#
#  Load the test csv file
#
TestDF = pd.read_csv("C:\\Training\\R\\kaggle\\titanic\\test.csv")
#
# Set the same AGE defaults as model for test DF ... if any na are found in age
#
TestDF.loc[(TestDF['Pclass']==1) & (TestDF['Sex']=='female') & (TestDF['Age'].isna()),'Age'] = 34
TestDF.loc[(TestDF['Pclass']==2) & (TestDF['Sex']=='female') & (TestDF['Age'].isna()),'Age'] = 29
TestDF.loc[(TestDF['Pclass']==3) & (TestDF['Sex']=='female') & (TestDF['Age'].isna()),'Age'] = 22
TestDF.loc[(TestDF['Pclass']==1) & (TestDF['Sex']=='male') & (TestDF['Age'].isna()),'Age'] = 41
TestDF.loc[(TestDF['Pclass']==2) & (TestDF['Sex']=='male') & (TestDF['Age'].isna()),'Age'] = 31
TestDF.loc[(TestDF['Pclass']==3) & (TestDF['Sex']=='male') & (TestDF['Age'].isna()),'Age'] = 26
#
#  Subset the columns to feed to model
# 
TestDF2 = TestDF[['Pclass','Age','Sex']]
#
# Expand factors of age using dummies or one hot encoder
#
TestDF2 = pd.get_dummies(TestDF2,columns=['Sex'])
#
#  Try predicting the test file
#
Test_Predict = model.predict(TestDF2)


# In[114]:


#
#  Format and write to the out put CSV - 'PassengerId' and 'Survived' are the two columns needed
#

Test_Out = pd.DataFrame(TestDF['PassengerId'])
Test_Out['Survived'] = Test_Predict
Test_Out.to_csv("C:\\Training\\R\\kaggle\\titanic\\test_out.csv",header=True,index=False)

