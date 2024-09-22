#!/usr/bin/env python
# coding: utf-8

# # DRAGON REAL ESTATE-PRICE PREDICTOR

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_excel("ML_PROJECT_NO_1.xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df["CHAS"].value_counts()


# In[6]:


df.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#for plotting histogram
import matplotlib.pyplot as plt
df.hist(bins=50,figsize=(20,15))


# # Train-Test Splitting

# In[9]:


#for learning purpose
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42) #for stoping the random values
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


#train_set, test_set = split_train_test(df,0.2)


# In[11]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split #sklearn.model_selection is package,splits arrays or matrices into random subsets for train and test data
train_set,test_set =train_test_split(df,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit #Provides train/test indices to split data in train/test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index,test_index in split.split(df,df["CHAS"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index] #loc is label-based, which means that you have to specify rows and columns based on their row and column labels


# In[14]:


strat_test_set["CHAS"].value_counts()


# In[15]:


strat_train_set["CHAS"].value_counts()


# In[16]:


#95/7


# In[17]:


#376/28


# In[18]:


df = strat_train_set.copy()


# # looking for correlations

# In[19]:


corr_matrix = df.corr()
corr_matrix["MEDV"].sort_values(ascending=False)
#used to find the pairwise correlation of all columns in the Pandas Dataframe in Python
#Correlation - summarizes the strength and direction of the linear (straight-line) association between two quantitative variables. Denoted by r, it takes values between -1 and +1. A positive value for r indicates a positive association, and a negative value for r indicates a negative association


# In[20]:


#from pandas.plotting import scatter_matrix #used to easily generate a group of scatter plots between all pairs of numerical features
#attributes = ["MEDV","RM","ZN","LSTAT"]
#scatter_matrix(df[attributes],figsize = (12,8))


# In[21]:


df.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)


# # Trying out  Attribute combinations

# In[22]:


df["TAXRM"] = df["TAX"]/df["RM"]


# In[23]:


df.head()


# In[24]:


corr_matrix = df.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[25]:


df = strat_train_set.drop("MEDV",axis=1)
df_labels = strat_train_set["MEDV"].copy()


# # Missing Attributes

# In[26]:


# To take care of missing attributes,you have 3 options:
#1.Get rid of the missing datapoints
#2.Get rid of the whole attribute
#3.Set the value to some value(0,mean or median)


# In[27]:


a = df.dropna(subset=["RM"]) #option 1
a.shape
# Note that the original  df dataframe will remain unchanged


# In[28]:


df.drop("RM",axis=1).shape  #option 2
# Note that there is no RM column and also note that the original df dataframe will remain unchanged 


# In[29]:


median = df["RM"].median() # Compute median for Option 3


# In[30]:


df["RM"].fillna(median) # Option 3
# Note that the original df dataframe will remain unchanged 


# In[31]:


df.shape


# In[32]:


df.describe() # before we started filling missing attributes


# In[33]:


from sklearn.impute import SimpleImputer #a class in the sklearn. impute module that can be used to replace missing values in a dataset, using a variety of input strategies
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
imputer.fit(df)


# In[34]:


imputer.statistics_


# In[35]:


X = imputer.transform(df)


# In[36]:


df_tr = pd.DataFrame(X, columns = df.columns)


# In[37]:


df_tr.describe()


# # Scikit-learn Design

# Primarily,three types of object:
# 
# 1.Estimators - it estimates some parameter based on a dataset.eg-imputer
#  it has a fit method and transform method.
#  Fit method -fits the dataset and calculates internal parameters
#  
# 2.Transformers - transform method takes input and returns output based on the learning from fit(). it also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3.Predictors - LinearRegression model  is an example of predictor.fit() and predict() are two common function.It also gives score() funtion which will evaluate the predictions.

""" Feature Scaling
Primarily two types of feature scaling(combine all feature scale) methods:
1.Min- max scaling(Normalization)
   (value-min)/(max-min)
   sklearn provide a class called MinMaxScaler for this
2.standardization
    (value - mean)/std
    sklearn provide a class called StandardScaler for this
 Creating a pipeline """

# In[38]:


from sklearn.pipeline import Pipeline #pipeline return numpyarray,A numpy array is a grid of values, all of the same type, and indexing is same as array
from sklearn.preprocessing import StandardScaler # StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way.
my_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    #.......... add as many as you want in your pipeline
    ("std_scaler", StandardScaler()),
])


# In[39]:


df_num_tr = my_pipeline.fit_transform(df)


# In[40]:


df_num_tr.shape


# # Selecting a desired model for Dragon Real Estates

# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#use for predicting the value
#model = LinearRegression()
#model = RandomForestRegressor
model = DecisionTreeRegressor()
model.fit(df_num_tr, df_labels)


# In[42]:


some_data = df.iloc[:5]


# In[43]:


some_labels = df_labels.iloc[:5]


# In[44]:


prepared_data = my_pipeline.transform(some_data)


# In[45]:


model.predict(prepared_data)


# In[46]:


list(some_labels)


# # Evaluating the model

# In[47]:


from sklearn.metrics import mean_squared_error
df_predictions = model.predict(df_num_tr)
mse = mean_squared_error(df_labels,df_predictions)
rmse = np.sqrt(mse)


# In[48]:


rmse  # Error can't be 0 ,it can when overfitting problem


# # Using better evaluation technique -Cross Validation

# In[49]:


# 1 2 3 4 5 6 7 8 9 10 
from sklearn.model_selection import cross_val_score # Cross validation-a statistical method used to estimate the performance of machine learning models
# here we use k-fold cross validation method
scores = cross_val_score(model, df_num_tr,df_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[50]:


rmse_scores


# In[51]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[52]:


print_scores(rmse_scores)


# # Saving the model

# In[53]:


from joblib import dump,load
dump(model,"Dragon.joblib")


# # Testing the model on test data

# In[54]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[55]:


final_rmse # It is lesser than model that we used it's good
#  And Random Forest Regression model is better than other


# In[56]:


prepared_data[0]


# # Using the model

# In[57]:


from joblib import dump,load
import numpy as np
model = load("Dragon.joblib")
features = np.array([[-7.43942006,  7.20172957, -1.12165014, -8.27288841, -1.42262747,
       -.002469957 , -0.31238772,  1.61111401, -.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.87965678]])
model.predict(features)

