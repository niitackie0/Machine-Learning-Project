#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[55]:


df = pd.read_csv ('C:\\Users\\henry\\Desktop\\henry\\Bengaluru_House_Data.csv')


# In[56]:


df.head()


# In[57]:


df.shape


# In[107]:


dfn = df.drop(columns = ['area_type', 'society', 'availability', "balcony"])
dfn.shape


# In[110]:


dfn.head()


# In[59]:


dfn.describe()


# In[60]:


dfn.isnull().sum()


# In[61]:


df3 = dfn.dropna()


# In[62]:


df3.isnull().sum()


# In[63]:


df3.shape


# In[64]:


df3['size'].unique()


# In[65]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[66]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[67]:


df3[~df3['total_sqft'].apply(is_float)]


# the purpose of this function to convert the column from the data type 'str' to the data type 'int' or 'float' which will later be be used to calculate the aveage since they are in a range form.

# In[68]:


def convert_sqft_to_num(x):
    tokens = x.split('-') 
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[69]:


df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
df3=df3[df3.total_sqft.notnull()]


# In[70]:


df3.head()


# In[71]:


df3.loc[30]
(2100+2850)/2


# it is important to create a column to indicate price per sqft because in real esate this is a key indicator to determine the price of the property

# In[72]:


df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
df4.head()


# In[73]:


df4.location.nunique()


# (.nunique) this is used to find the unique count of features in the location column

# In[74]:


df4_stats = df4['price_per_sqft'].describe()
df4_stats


# In[75]:


df4.to_csv("bhp.csv", index=False)


# In[76]:


df4.location = df4.location.apply(lambda x: x.strip())

location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[77]:


len(location_stats[location_stats<=10])


# In[78]:


location_stats_less_tha_10 = location_stats[location_stats<=10]
location_stats_less_tha_10


# In[79]:


df4.loc[df4['location'].isin(location_stats), 'location'] = 'other'
unique_locations_count = len(df4['location'].unique())


# In[80]:


len(df4.location.unique())


# In[81]:


df4.location = df4.location.apply(lambda x: "other" if x in location_stats_less_tha_10 else x)
len(df4.location.unique())


# In[82]:


df4.head(15)


# In[83]:


df4[df4.total_sqft/df4.bhk<300].head()


# In[84]:


df4.shape


# In[85]:


df5 = df4[~(df4.total_sqft/df4.bhk<300)]
df5.shape


# In[86]:


df5.price_per_sqft.describe()


# In[87]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_pps_outliers(df5)
df6.shape


# In[88]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df6,"Rajaji Nagar")


# In[89]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7= remove_bhk_outliers(df6)
# df8 = df7.copy()
df7.shape


# In[90]:


df7[df7.bath>10]


# In[91]:


df7[df7.bath>df7.bhk+2]


# In[92]:


df8 = df7[df7.bath<df7.bhk+2]
df8.shape


# In[93]:


df9 = df8.drop(columns = ["size", "price_per_sqft"])
df9.head(3)


# In[94]:


dummies = pd.get_dummies(df9.location)
dummies.head(3)


# In[95]:


df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df10.head()


# In[96]:


df11 = df10.drop('location',axis='columns')
df11.head(2)


# In[97]:


x = df11.drop(columns = "price")
x.head(3)


# In[98]:


x.shape


# In[99]:


y = df11.price
y.head(3)


# In[100]:


y.shape


# In[101]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[102]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
lr_clf.score(x_test, y_test)


# In[103]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)


# In[104]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    x = np.zeros(len(x.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[105]:


import pickle
with open("banglore_home_prices_model.pickle", "wb") as f:
    pickle.dump(lr_clf,f)


# In[106]:


import json
columns = {
    "data_columns" : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
     


# In[ ]:




