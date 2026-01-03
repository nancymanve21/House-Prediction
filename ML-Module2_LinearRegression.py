#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# In[14]:


#Load the dataset
house = pd.read_csv('Copy of housing_prices.csv')
house.head()


# In[ ]:


# Exploratory data analysis


# In[15]:


house.info()


# In[16]:


house.describe()


# In[17]:


# Pairplot to visualize relationships between variables
import seaborn as sns
sns.pairplot(house)
plt.show()



# In[18]:


# Correlation matrix
correlation_matrix = house.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot = True, cmap ='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[19]:


#Histograms of individual features
house.hist(figsize=(12, 10), bins = 20)
plt.show()


# In[20]:


#Box plots to check outliers
plt.figure(figsize = (12, 10))
for i, column in enumerate(house.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(house[column])
    plt.title(column)
plt.tight_layout()
plt.show()


# ## Simple Linear Regression

# In[21]:


# Define the target variable and predictor variable
X = house[['MedInc']] #Predictor
y = house['Price'] #Target


# In[22]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)


# In[23]:


# Create and train the model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)


# In[24]:


# Predictions
y_pred = simple_model.predict(X_test)

#Evaluate the model
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

#Plot the regression line
plt.scatter(X_test, y_test, color = 'blue', label = 'Actual data')
plt.plot(X_test, y_pred, color = 'red', linewidth = 2, label = 'Regression line')
plt.xlabel('MedInc')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


# In[26]:


# Assuming the model has already been trained and the relevant liraries have been imported

#Input Median Income value from the user
medinc_value = float(input("Enter the MedInc value:"))

# Convert the input to a 2D array for the model
medinc_val_arr = np.array([[medinc_value]])

#Predict the house price using the trained model
predicted_price = simple_model.predict(medinc_val_arr)

#Output the prediction
print(f"The prediction house price for MedInc value of {medinc_value} is: ${predicted_price[0]:.2f}")


# ## Multiple Linear Regression

# In[28]:


# Define multiple predictors
X_multi = house[['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude']]
y_multi = house['Price']


# In[29]:


#Split the dataset into traning and testing sets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size = 0.2, random_state = 42)


# In[30]:


#Creating a model
multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)


# In[32]:


# Prediction
y_pred_multi = multi_model.predict(X_test_multi)

#Evaluate the model
print(f'Mean Squarred error :{mean_squared_error(y_test_multi, y_pred_multi)}')
print(f'R^2 Score: {r2_score(y_test_multi,y_pred_multi)}')


# In[33]:


# Display the coefficients of the model
coefficients = pd.DataFrame(multi_model.coef_, X_multi.columns, columns=['Coefficient'])
print(coefficients)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




