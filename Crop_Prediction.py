import numpy as np
import pandas as pd

cd=pd.read_csv('Crop Prediction dataset.csv')
#Gives the number of rows and columns
print(cd.shape)
#Print's first 5 rows of dataset
print(cd.head())
#Print's the column names
print(cd.columns)
#Print's count,mean, std, min and max
print(cd.describe())
#Checking for null values in each attribute
print(cd.isnull().sum())
#Filling the mean of production values at null spaces 
cd['Production']=cd['Production'].fillna(cd['Production'].mean())
#Cross Verifying
print(cd.isnull().values.any())
# Yield = Production/Area
cd['Yield']=(cd['Production']/cd['Area'])
print(cd.head(10))
