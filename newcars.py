import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

#CARGA DE DATOS
df_ori = pd.read_csv('car_price_prediction.csv')
df = df_ori.copy()

#LIMPIEZA DE DATOS

df['Has_Turbo'] = df['Engine volume'].astype(str).str.contains('Turbo').astype(int)
df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '').astype(float)

df['Car_Age'] = 2025 - df['Prod. year']

top_models = df['Model'].value_counts().head(50).index
df['Model_Grouped'] = df['Model'].apply(lambda x: x if x in top_models else 'Other')

df.drop(columns=['ID', 'Model', 'Prod. year'], inplace=True)

df['Levy'] = pd.to_numeric(df['Levy'].replace('-', np.nan))
df['Levy'] = df['Levy'].fillna(0)

df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(int)

door_mapping = {'04-may': 4, '4-5': 4, '02-mar': 2, '2-3': 2, '>5': 5}
df['Doors'] = df['Doors'].replace(door_mapping)
df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce').fillna(4).astype(int)

#OUTLIERS

df = df[df['Price'] > 500]
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= (Q1 - 1.5 * IQR)) & (df['Price'] <= (Q3 + 1.5 * IQR))]

#ELIMINACION DE DUPLICADOS

df.drop_duplicates(inplace=True)

#MODELO DE PREDICCION

le = LabelEncoder()
df['Manufacturer'] = le.fit_transform(df['Manufacturer'])

cols_encode = ['Category', 'Fuel type', 'Gear box type', 'Drive wheels',
               'Leather interior', 'Wheel', 'Color', 'Model_Grouped']
df_model = pd.get_dummies(df, columns=cols_encode, drop_first=True)

X = df_model.drop(columns=['Price'])
y = df_model['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MODELO RandomFOrestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.0f}")

#GRAFICA DISPERSION

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#4c72b0', edgecolor='k')
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
plt.title('Dispersión: Predicción vs Realidad', fontsize=14)
plt.xlabel('Precio Real ($)')
plt.ylabel('Precio Predicho ($)')

#GRAFICA DISTRIBUCION

plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label="Datos Reales", fill=True, color='green', alpha=0.3, linewidth=2)
sns.kdeplot(y_pred, label="Predicción", fill=True, color='blue', alpha=0.3, linewidth=2)
plt.title('Distribución de Precios (Curvas)', fontsize=14)
plt.xlabel('Precio ($)')
plt.ylabel('Densidad')
plt.legend()

plt.tight_layout()
plt.show()

import joblib

joblib.dump(rf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
