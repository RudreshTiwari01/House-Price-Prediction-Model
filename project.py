import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/Rudresh Kumar Tiwari/Downloads/Bengaluru_House_Data.csv")

df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if isinstance(x, str) else None)

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

def convert_sqft(x):
    if isinstance(x, str):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        if is_float(x):
            return float(x)
        return None
    return x

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft', 'bhk'])

df['location'] = df['location'].astype(str).str.strip()
df['location'] = df['location'].replace('', 'other')

location_count = df['location'].value_counts()
rare_loc = location_count[location_count <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_loc else x)

df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())

df = df[(df['total_sqft'] / df['bhk']) >= 250]

df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
df = df[df['price_per_sqft'].between(
    df['price_per_sqft'].quantile(0.05),
    df['price_per_sqft'].quantile(0.95)
)]
print(df.info())
print(df.columns)
df = df.drop(['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11','Unnamed: 12'], axis=1)
print(df.columns)
df.drop(columns=['area_type'], inplace=True, errors='ignore')

le = LabelEncoder()
df['availability'] = le.fit_transform(df['availability'].astype(str))

df['society'] = le.fit_transform(df['society'].astype(str))
df['location'] = le.fit_transform(df['location'].astype(str))

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

x = df[['location','total_sqft','bhk','bath','balcony']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(x_train, y_train)

print("XGBoost Model Score:", xgb_model.score(x_test, y_test))

encoded_loc = le.transform(['Whitefield'])[0]
input_data = [[encoded_loc,4200,4,4,2]]
print("Predicted Price:", xgb_model.predict(input_data)[0])
