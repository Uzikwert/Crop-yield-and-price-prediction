import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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


# Most crops are grown in the Kharif season, making it the most important agricultural season in the dataset.
sns.histplot(x=cd['Season'])
plt.title("Histogram of seasons")
plt.show() 

# There is strong regional variation in rainfall, which influences crop selection and productivity.
cd.groupby('State')['Annual_Rainfall'].mean().plot(kind='pie')
plt.title('Statewise Annual Rainfall')
plt.tight_layout()
plt.show()

# Crop yield is not stable over time and may be affected by external factors like virus outbreaks or natural calamities.
cd.groupby('Crop_Year')['Yield'].mean().plot(kind='line', marker='o')
plt.title('Average Yield Over Years')
plt.xlabel('Year')
plt.show()

# Production is highly concentrated in a few states, with Kerala leading significantly.
cd.groupby('State')['Production'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 States by Total Production')
plt.show()

# 1000mm to 2000mm of rainfall is the optimal range for a great yield. Also noticaed that rainfal above 3800mm negetively impacts yield.
plt.plot(cd['Annual_Rainfall'], cd['Yield'], alpha=0.3)
plt.xlabel('Annual Rainfall')
plt.ylabel('Yield')
plt.title('Rainfall vs Yield')
plt.show()

# Larger farming areas require significantly more fertilizer and pesticide.
data = cd.drop(['State','Season','Crop'], axis = 1)
data.corr()
sns.heatmap(data.corr(), annot =True, fmt='.4f')
plt.title('Correlation Matrix')

#Scaling all the values using StandardScaler so that no column overshadows another
col_scaled=['Area','Production','Annual_Rainfall','Fertilizer','Pesticide','Yield']
scaler=StandardScaler()
cd[col_scaled] = scaler.fit_transform(cd[col_scaled])
print(cd.head())

# Dropped a column that was not relevant to the model
cd.drop('Crop_Year',axis=1,inplace=True)
# Performed LabelEncoder on Crop & State columns and OneHotEncoding on Season column 
le =LabelEncoder()
cd["Crop"]=le.fit_transform((cd["Crop"]))
cd["State"]=le.fit_transform((cd["State"]))
cd = pd.get_dummies(cd,columns=["Season"],drop_first=True)

# Train test split 80/20
X=cd.drop('Yield', axis=1)
y=cd['Yield']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
print(f"Training Shape: {X_train.shape}\nTesting Shape: {X_test.shape}")

# Linear Regression our baseline model
model=LinearRegression()
model.fit(X_train,y_train)
baseline_pred=model.predict(X_test)
baseline_score=r2_score(y_test, baseline_pred)
print(baseline_score)


