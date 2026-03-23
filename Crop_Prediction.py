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

cd.groupby('State')['Annual_Rainfall'].mean().plot(kind='pie')
plt.title('Statewise Annual Rainfall')
plt.tight_layout()
plt.show()

cd.groupby('Crop_Year')['Yield'].mean().plot(kind='line', marker='o')
plt.title('Average Yield Over Years')
plt.xlabel('Year')
plt.show()

cd.groupby('State')['Production'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 States by Total Production')
plt.show()

plt.plot(cd['Annual_Rainfall'], cd['Yield'], alpha=0.3)
plt.xlabel('Annual Rainfall')
plt.ylabel('Yield')
plt.title('Rainfall vs Yield')
plt.show()

data = cd.drop(['State','Season','Crop'], axis = 1)
data.corr()
sns.heatmap(data.corr(), annot =True, fmt='.4f')
plt.title('Correlation Matrix')





