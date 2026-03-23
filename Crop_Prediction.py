import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

cd=pd.read_csv('crop_yield.csv')
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
cd['Production'] = cd['Production'].fillna(cd['Production'].mean())
#df = cd.dropna(subset=['Production'])
#Cross Verifying
print(cd.isnull().values.any())
# Yield = Production/Area
cd['Yield']=(cd['Production']/cd['Area'])
print(cd.head(10))


sns.histplot(x=cd['Season'])
plt.title("Histogram of seasons")
plt.show() 

sns.boxplot(x=cd["State"],y=cd["Crop_Year"])
plt.show()

data = cd.drop(['State','Season','Crop'], axis = 1)
data.corr()
sns.heatmap(data.corr(), annot =True, fmt='.4f')
plt.title('Correlation Matrix')





