#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries  

# In[85]:


import matplotlib.pyplot as mtp  
import pandas as pd  
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


# # Importing Datasets  

# In[86]:


data_set= pd.read_csv(r"C:\Users\Ravi Chauhan\Documents\forest-cover-v1.csv")  


# In[87]:


i = "Forest Area"
[int(s) for s in i.split() if s.isdigit()]


# In[88]:


droped_years = []
for i in data_set.columns:
    year = [int(s) for s in i.split() if s.isdigit()]
    if year != []:
        year = [int(s) for s in i.split() if s.isdigit()][0]
        if (year % 5) != 0:
            droped_years.append(i)
        
        
print(droped_years)    


# In[89]:


df = data_set.drop(droped_years, axis=1)
df.head()


# In[90]:


from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy='mean')
df['Forest Area 1990']=imp.fit_transform(df['Forest Area 1990'].values.reshape(-1,1))
df['Forest Area 1995']=imp.fit_transform(df['Forest Area 1995'].values.reshape(-1,1))
df['Forest Area 2000']=imp.fit_transform(df['Forest Area 2000'].values.reshape(-1,1))
df['Forest Area 2005']=imp.fit_transform(df['Forest Area 2005'].values.reshape(-1,1))
df['Forest Area 2010']=imp.fit_transform(df['Forest Area 2010'].values.reshape(-1,1))
df['Forest Area 2015']=imp.fit_transform(df['Forest Area 2015'].values.reshape(-1,1))
df['Forest Area 2020']=imp.fit_transform(df['Forest Area 2020'].values.reshape(-1,1))
#Continent
Continent=pd.get_dummies(df['Continent'],drop_first=False)
Continent=pd.concat([df,Continent],axis=1)
Continent.drop(['Continent'],axis=1,inplace=True)


# In[91]:


x=df.iloc[:,4:15]
x


# In[92]:


y=df.iloc[:,15:16]
y


# In[93]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)

#view transformed values
print(y)


# In[94]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.40, random_state=0)  


# In[95]:


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)    


# In[96]:


score=pd.DataFrame()
r21=[]
mse1=[]
mae=[]
rmse=[]
score['Algorithms']=['Linear Regression','Decision Tree','Random Forest','Support Vector Machine']


# # LINEAR REGRESSION ALGORITHM

# In[97]:


from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train) 


# In[98]:


y_pred= regressor.predict(x_test) 
x_pred= regressor.predict(x_train)


# In[99]:


print('Train Score: ', regressor.score(x_train, y_train)*100)  
print('Test Score: ', regressor.score(x_test, y_test)*100)  


# In[100]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error is:",mse)
print("R-Squared:",r2)
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
r21.append(r2)
mse1.append(mse)
mae.append(metrics.mean_absolute_error(y_test,y_pred))
rmse.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # DECISION TREE ALGORITHM

# In[101]:


#Fitting Decision Tree classifier to the training set  
from sklearn.tree import DecisionTreeRegressor 
regressor= DecisionTreeRegressor(max_depth=10, random_state=0)  
regressor.fit(x_train, y_train)


# In[102]:


#Predicting the test set result  
y_pred= regressor.predict(x_test)  


# In[103]:


y_pred= regressor.predict(x_test) 
x_pred= regressor.predict(x_train)


# In[104]:


print('Train Score: ', regressor.score(x_train, y_train)*100)  
print('Test Score: ', regressor.score(x_test, y_test)*100)  


# In[105]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error is:",mse)
print("R-Squared:",r2)
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
r21.append(r2)
mse1.append(mse)
mae.append(metrics.mean_absolute_error(y_test,y_pred))
rmse.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # RANDOM FOREST ALGORITHM

# In[106]:


#Fitting Decision Tree classifier to the training set  
from sklearn.ensemble import RandomForestRegressor  
regressor= RandomForestRegressor(n_estimators= 10)  
regressor.fit(x_train, y_train)  


# In[107]:


#Predicting the test set result  
y_pred= regressor.predict(x_test)  


# In[108]:


y_pred= regressor.predict(x_test) 
x_pred= regressor.predict(x_train)


# In[109]:


print('Train Score: ', regressor.score(x_train, y_train)*100)  
print('Test Score: ', regressor.score(x_test, y_test)*100)  


# In[110]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error is:",mse)
print("R-Squared:",r2)
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
r21.append(r2)
mse1.append(mse)
mae.append(metrics.mean_absolute_error(y_test,y_pred))
rmse.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # Support Vector Machine Algorithm

# In[111]:


from sklearn.svm import SVR # "Support vector classifier"  
regressor = SVR(kernel='linear')  
regressor.fit(x_train, y_train)  


# In[112]:


#Predicting the test set result  
y_pred= regressor.predict(x_test)  


# In[113]:


y_pred= regressor.predict(x_test) 
x_pred= regressor.predict(x_train)


# In[114]:


print('Train Score: ', regressor.score(x_train, y_train)*100)  
print('Test Score: ', regressor.score(x_test, y_test)*100)  


# In[115]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error is:",mse)
print("R-Squared:",r2)
print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
r21.append(r2)
mse1.append(mse)
mae.append(metrics.mean_absolute_error(y_test,y_pred))
rmse.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[116]:


score['R2 Score']=r21
score['Mean Absolute Error']=mae
score['Mean Square Error']=mse1
score['Root Mean Square Error']=rmse


# In[117]:


score


# In[118]:



mtp.plot(score["R2 Score"],'-o',mfc='r')
labels=score['Algorithms']
x=[0,1,2,3]
mtp.xticks(x,labels,rotation='vertical')
mtp.xlabel("Algorithms")
mtp.ylabel("Score")
mtp.legend(['R2 Score'])
mtp.show


# In[ ]:





# In[ ]:




