import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

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
#Cross Verifying
print(cd.isnull().values.any())
print(cd.head(10))
#dropping zero values
cd = cd[(cd['Yield'] > 0) & (cd['Production'] > 0) & (cd['Area'] > 0)]

# Fix potential whitespace issue
cd['Crop'] = cd['Crop'].str.strip()
cd['State'] = cd['State'].str.strip()
cd['Season'] = cd['Season'].str.strip()

# Drop different-unit crops
crops_to_drop = ['Coconut', 'Sugarcane', 'Banana', 'Tapioca', 'Sweet potato']
cd = cd[~cd['Crop'].isin(crops_to_drop)]
print(f"Rows after drop: {len(cd)}")
print(cd.groupby('Crop')['Yield'].median().sort_values(ascending=False).head(5))

# Remove outliers per crop
def remove_outliers_per_crop(df, col, factor=1.5):
    before = len(df)
    def filter_group(group):
        Q1 = group[col].quantile(0.25)
        Q3 = group[col].quantile(0.75)
        IQR = Q3 - Q1
        return group[(group[col] >= Q1 - factor*IQR) & 
                     (group[col] <= Q3 + factor*IQR)]
    df = df.groupby('Crop', group_keys=False).apply(filter_group)
    print(f"{col}: removed {before - len(df)} rows")
    return df

cd = remove_outliers_per_crop(cd, 'Yield')
cd = remove_outliers_per_crop(cd, 'Production')
cd = remove_outliers_per_crop(cd, 'Area')

# Capping and checking
for col in ['Yield', 'Production', 'Area']:
    upper = cd[col].quantile(0.99)
    lower = cd[col].quantile(0.01)
    cd[col] = cd[col].clip(lower=lower, upper=upper)

print(f"\nFinal rows: {len(cd)}")
print(f"Skewness: {cd['Yield'].skew():.2f}")
print(cd['Yield'].describe())


# Most crops are grown in the Kharif season, making it the most important agricultural season in the dataset.
sns.histplot(x=cd['Season'])
plt.title("Histogram of seasons")
plt.show() 

# There is strong regional variation in rainfall, which influences crop selection and productivity.
cd.groupby('State')['Annual_Rainfall'].mean().plot(kind='pie')
plt.title('Statewise Annual Rainfall')
plt.tight_layout()
plt.show()

# Crop yield has generally increased over the years.
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
plt.show()


#std = cd['Yield'].std()
#print(std)
# Dropped a column that is not relevant to the model
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


#Scaling all the values using StandardScaler so that no column overshadows another
col_scaled=['Area','Production','Annual_Rainfall','Fertilizer','Pesticide']
scaler=StandardScaler()
X_train[col_scaled] = scaler.fit_transform(X_train[col_scaled])
X_test[col_scaled] = scaler.transform(X_test[col_scaled])
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)


# Linear Regression our baseline model
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)
baseline_pred=lr_model.predict(X_test)
baseline_score=r2_score(y_test, baseline_pred)
mse = mean_squared_error(y_test, baseline_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print("R2 Score: ",baseline_score)

train1_pred = lr_model.predict(X_train)
print(f"Train R²: {r2_score(y_train, train1_pred):.4f}")
print(f"Test R²:  {r2_score(y_test, baseline_pred):.4f}")

# RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"MSE  : {mean_squared_error(y_test, rf_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")
print(f"R²   : {r2_score(y_test, rf_pred):.4f}")


# Check if model is overfitting
train_pred = rf_model.predict(X_train)
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²:  {r2_score(y_test, rf_pred):.4f}")

# XGBoost model
xg_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xg_model.fit(X_train, y_train)
xg_pred = xg_model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, xg_pred)))
print("R2:", r2_score(y_test, xg_pred))

train3_pred = xg_model.predict(X_train)
print(f"Train R²: {r2_score(y_train, train3_pred):.4f}")
print(f"Test R²:  {r2_score(y_test, xg_pred):.4f}")


# Comparision of all models
model_names = ['Linear Regression', 'Random Forest', 'XGBoost']
predictions = [baseline_pred, rf_pred, xg_pred]

mse_scores  = [mean_squared_error(y_test, p) for p in predictions]
rmse_scores = [np.sqrt(m) for m in mse_scores]
r2_scores   = [r2_score(y_test, p) for p in predictions]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

metrics = ['MSE', 'RMSE', 'R²']
scores  = [mse_scores, rmse_scores, r2_scores]
colors  = ['#e74c3c', '#3498db', '#2ecc71']

for ax, metric, score, color in zip(axes, metrics, scores, colors):
    bars = ax.bar(model_names, score, color=color, alpha=0.8, edgecolor='black')
    ax.set_title(metric, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    for bar, val in zip(bars, score):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()



