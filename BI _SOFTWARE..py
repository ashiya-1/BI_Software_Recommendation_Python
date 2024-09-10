#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('BI_Software_recommendation_dataset.csv')


# ### Display Rows of Dataset

# In[3]:


df.head()


# In[4]:


df.tail()


# ### Dataset Overview

# In[5]:


print('size of dataset :',df.size)
print('No.of rows :',df.shape[0])
print('No.of columns :',df.shape[1])


# In[6]:


df.columns


# In[7]:


df.info()


# ### Data Cleaning

# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.nunique()


# ### Statistical Summary about Dataframe

# In[11]:


df.describe(include='all')


# In[12]:


df['rating'].describe()


# ### Data Exploration

# In[13]:


df['Business_scale'].value_counts() ,df['category'].value_counts()


# In[14]:


df['industry'].value_counts()


# In[15]:


df['category'].value_counts()


# In[16]:


df['user_type'].value_counts()


# In[17]:


df['no_of_users'].value_counts()


# In[18]:


df['deployment'].value_counts()


# In[19]:


df['OS'].value_counts()


# In[20]:


df['mobile_apps'].value_counts()


# In[21]:


df['pricing'].value_counts()


# In[22]:


df['rating'].value_counts()


# ### Finding Insight

# **Checking for rating max,min,mean**

# In[23]:


print(df['rating'].agg(['max','mean','min']))


# **Unique Industries by Business Scale**

# In[24]:


pd.set_option('display.max_colwidth', None)# use to set column width
df.groupby('Business_scale')['industry'].unique()


# **Checking Unique Pricing**

# In[25]:


df['pricing'].unique()


# **Analyzing Entries with Open Source Pricing**

# In[26]:


df[df['pricing']=='Open Source']


# **Descriptive Statistics of Ratings for Open Source Pricing**

# In[27]:


df[df['pricing'] == 'Open Source']['rating'].describe()


# **Analyzing Entries with Enterprise Pricing**

# In[28]:


df[df['pricing']=='Enterprise']


# **Descriptive Statistics of Ratings for Open Source Pricing**

# In[29]:


df[df['pricing'] == 'Enterprise']['rating'].describe()


# **Descriptive Statistics of Ratings for Freemuim Pricing**

# In[30]:


df[df['pricing'] == 'Freemium']


# **Descriptive Statistics of Ratings for Freemium Pricing**

# In[31]:


print(df[df['pricing'] == 'Freemium']['rating'].describe())


# **Unique Industries by Pricing Category**

# In[32]:


unique_industries_by_pricing=df.groupby('pricing')['industry'].unique()
for pricing,industry in unique_industries_by_pricing.items():
     print(pricing,':',industry,end='\n')


# **Enterprise Pricing: User Type and Number of Users**

# In[33]:


Enterprise_pivot= df[df['pricing'] == 'Enterprise'].pivot_table(index='user_type',columns='no_of_users',aggfunc='size',fill_value=0)
print(Enterprise_pivot)


# **OpenSource Pricing: User Type and Number of Users**

# In[34]:


Opensource_pivot= df[df['pricing'] == 'Open Source'].pivot_table(index='user_type',columns='no_of_users',aggfunc='size',fill_value=0)
print(Opensource_pivot)


# **Freemium Pricing: User Type and Number of Users**

# In[35]:


Freemium_pivot= df[df['pricing'] == 'Freemium'].pivot_table(index='user_type',columns='no_of_users',aggfunc='size',fill_value=0)
print(Freemium_pivot)


# **Extracting Columns: Category, OS, and Mobile Apps**

# In[36]:


df[['category','OS','mobile_apps']]


# **Extracting Rows from Index 2 to 5**

# In[37]:


df.loc[2:5]


# **Extracting Row at Index 5**

# In[38]:


df.loc[5]


# **Extracting Columns from Index 1 to 5**

# In[39]:


df.iloc[:,1:6]


# **Unique Categories by OS**

# In[40]:


unique_category_os=df.groupby('OS')['category'].unique()
for OS,category in unique_category_os.items():
    print(f"\n{OS}:")
    for category in category:
             print(f" -{category}")


# **Top 10 Categories with Highest Ratings**

# In[41]:


category_ratings = df.groupby('category')['rating'].max()
top_10_categories = category_ratings.nlargest(10)
print(top_10_categories)


# **Filtering Rows with Ratings Between 3.0 and 4.0**

# In[42]:


df[df['rating'].between(3.0,4.0)]


# **Average Rating by Category, Industry, and Business Scale (Ratings Below 4)**

# In[43]:


filtered_df = df[df['rating'] < 4]
grouped_summary = filtered_df.groupby(['category', 'industry', 'Business_scale']).agg(avg_rating=('rating', 'mean'))
sorted_summary = grouped_summary.sort_values(by='avg_rating')
print(sorted_summary)


# **Summary of Ratings (3.0 to 4.0) by Category**
# 

# In[44]:


filtered_df = df[df['rating'].between(3, 4)]
grouped_summary = filtered_df.groupby('category').agg( count=('rating', 'size'), avg_rating=('rating', 'mean'))
print(grouped_summary)


# **Unique Ratings in the Dataset**

# In[45]:


df['rating'].unique()


# **Minimum Ratings by Category (Ratings Below 4.0)**

# In[46]:


result=df[df['rating']<4.0]
result.groupby('category').min()


# ### Visualization

# **Rating Distribution**

# In[47]:


plt.figure(figsize=(6, 6))
sns.histplot(df['rating'], kde=True, color='blue', bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# **Ratings by Category**

# In[48]:


plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='rating', data=df, ci=None, palette='viridis')
plt.title('Average Ratings by Category')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# **Ratings by Industry**

# In[49]:


plt.figure(figsize=(10, 6))
sns.barplot(x='industry', y='rating', data=df, ci=None, palette='magma')
plt.title('Average Ratings by Industry')
plt.xlabel('Industry')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# **Ratings by Business Scale**

# In[50]:



plt.figure(figsize=(8, 6))
sns.boxplot(x='Business_scale', y='rating', data=df, palette='Set2')
plt.title('Ratings by Business Scale')
plt.xlabel('Business Scale')
plt.ylabel('Rating')
plt.show()


# **Ratings by Deployment Type**

# In[51]:


plt.figure(figsize=(5, 6))
sns.barplot(x='deployment', y='rating', data=df, ci=None, palette='coolwarm')
plt.title('Average Ratings by Deployment Type')
plt.xlabel('Deployment Type')
plt.ylabel('Average Rating')
plt.show()


# **Pricing vs. Rating**

# In[52]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='pricing', y='rating', data=df, hue='category', palette='tab10', s=100)
plt.title('Pricing vs Rating')
plt.xlabel('Pricing')
plt.ylabel('Rating')
plt.legend(title='category', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# **User Scale vs. Rating**

# In[53]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='no_of_users', y='rating', data=df, hue='Business_scale', palette='husl', s=100)
plt.title('Number of Users vs Rating')
plt.xlabel('Number of Users')
plt.ylabel('Rating')
plt.show()


# **Category Distribution Pie Chart**

# In[54]:


category_counts = df['category'].value_counts()
plt.figure(figsize=(7, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.0f%%', colors=plt.cm.Paired(range(len(category_counts))))
plt.title('Category Distribution')
plt.show()


# **OS and Mobile App Support vs. Rating**

# In[56]:


plt.figure(figsize=(7, 6))
sns.barplot(x='OS', y='rating', hue='mobile_apps', data=df, ci=None, palette='Spectral')
plt.title('Average Ratings by OES and Mobile App Support')
plt.xlabel('OES Support')
plt.ylabel('Average Rating')
plt.legend(title='Mobile App', loc='upper right')
plt.show()


# In[ ]:





# In[ ]:




