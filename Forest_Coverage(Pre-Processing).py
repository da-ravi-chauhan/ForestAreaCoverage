#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp


# # Reading The Dataset

# In[3]:


dataset=pd.read_csv(r"C:\Users\Ravi Chauhan\Documents\forest-cover-v1.csv")


# In[4]:


dataset


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.info()


# In[7]:


dataset.shape


# # I will not deal with missing data as there is missing data in a wide variety of places. Instead, I will first try to visualize by decomposing according to continents.

# In[8]:


dataset["Continent"].unique()


# # Let's reduce the "Forest Area" columns to make our job easier:
# # 1990-1995-2000-2005-2010-2015-2020

# In[9]:


dataset.columns


# In[10]:


i = "Forest Area"
[int(s) for s in i.split() if s.isdigit()]


# In[11]:


droped_years = []
for i in dataset.columns:
    year = [int(s) for s in i.split() if s.isdigit()]
    if year != []:
        year = [int(s) for s in i.split() if s.isdigit()][0]
        if (year % 5) != 0:
            droped_years.append(i)
        
        
print(droped_years)    


# In[12]:


df = dataset.drop(droped_years, axis=1)
df.head()


# In[13]:


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


# In[14]:


df.isnull().sum()


# In[15]:


x=df.iloc[:,4:15]
x


# In[16]:


y=df.iloc[:,15:16]
y


# # Splitting the dataset into training and test set.  

# In[17]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


# # feature Scaling  

# In[18]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)


# # Data Visualization Libraries 

# In[22]:


import seaborn as sns 
import matplotlib.pyplot as plt


# # New DataFrame which has index as a "Country Name"

# In[23]:


data = df.set_index("Country Name")
a = data.loc["China"]
a


# # Change in forest areas of 10 countries with the highest population growth rate:

# In[24]:


top_ten = df.sort_values("Population Rank")["Country Name"][0:10]
top_ten


# # VISUALIZATION

# # Box Plot

# In[30]:


#Comparison by continent, how has forest area changed over the years?
years = ['Forest Area 1990','Forest Area 1995', 'Forest Area 2000',
         'Forest Area 2005','Forest Area 2010', 'Forest Area 2015', 'Forest Area 2020']
for i in years:
    sns.boxplot(x=i, y="Continent", data=df,
                whis=[0, 100], width=.6, palette="vlag")
    
    sns.stripplot(x=i, y="Continent", data=df,
              size=4, color=".3", linewidth=0)
    plt.show()


# In[31]:


top_ten = top_ten.values.tolist()
top_ten


# In[32]:


x = []
for i in top_ten:
    y = data.loc[i][14] - data.loc[i][8]
    x.append(y)
print(x)


# #  x includes Forest Area 2020 - Forest Area 1990
# # top_ten includes name of the countries which has highest population rank

# In[33]:



fig, ax = plt.subplots()
new_df = pd.DataFrame({'Timing_Event':top_ten, 'Diff_Latency':x})
new_df['sign'] = new_df['Diff_Latency'] > 0

new_df['Diff_Latency'].plot(kind='bar', color=new_df.sign.map({True: (1.0, 0, 0, 0.7), False: (0, 0.6, 0, 0.7)}), 
                      ax=ax)
ax.axhline(0, color='k')

ax.set_xticklabels(top_ten) # changing x numereic labels into countries
plt.grid()


# # Top 10 countries with the highest Human population / Country Area ratio

# In[34]:


top_ten2 = df.sort_values("Population Density (per km²)")["Country Name"][0:10]
top_ten2


# In[35]:


a = data.sort_values("Population Density (per km²)")
a.head()


# In[36]:


x = a.columns[0:8]


# In[37]:


a.drop(x, axis=1, inplace=True)
a.head()


# In[38]:


b = a.iloc[0:10].T
b


# In[39]:


b.index


# In[40]:


b.index = ['1990', '1995', '2000','2005', '2010', '2015','2020']
b.index


# In[41]:


c= b.columns.values.tolist()
c


# In[42]:


b.head()


# In[43]:


b.Canada.plot(legend=True) #plot Canada column

plt.grid()


# In[44]:


for i in b.columns:
    b[i].plot(legend=True)
    plt.xlabel = ["Years"]
    plt.ylabel = ["Forest Area"]
    plt.show()


# In[45]:


#Other Method 
b.plot(subplots=True, figsize=(10, 8), layout=(5, 2))


# In[46]:


sns.barplot(data=df,x='Continent',y='Forest Area 2020')


# In[47]:


plt.title('Country Name vs. Forest Area 2020')
sns.scatterplot(data=df,x='Country Name',y='Forest Area 2020',alpha=0.7,s=15)


# In[48]:


plt.title('Continent vs. Forest Area 2020')
sns.scatterplot(data=df,x='Continent',y='Forest Area 2020',alpha=0.7,s=15)


# # Percentage of Continent

# In[49]:



plt.figure(figsize = (5,5))
df['Continent'].value_counts().plot(kind = "pie", textprops={'color':'black'}, autopct = "%.2f",cmap = "summer" )
plt.title('Percentage of Continent',fontsize=15)
plt.show()


# In[ ]:





# In[50]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap= 'coolwarm')


# In[51]:


sns.pairplot(df, hue='Forest Area 2020', height=2)


# In[ ]:




