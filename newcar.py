import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Configuración de estilo visual
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 5)

print("--- 1. CARGA Y PREPARACIÓN ---")
df_raw = pd.read_csv('car_price_prediction.csv') # Copia sucia para comparar
df = df_raw.copy()                               # Copia de trabajo

# ==============================================================================
# 2. LIMPIEZA Y FEATURE ENGINEERING
# ==============================================================================

# A. Ingeniería de Características (Nuevas variables)
# ---------------------------------------------------
# Extraer Turbo antes de limpiar
df['Has_Turbo'] = df['Engine volume'].astype(str).str.contains('Turbo').astype(int)
df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '').astype(float)

# Edad del auto (Más útil que el año)
df['Car_Age'] = 2025 - df['Prod. year']

# Agrupar Modelos (Top 50)
top_models = df['Model'].value_counts().head(50).index
df['Model_Grouped'] = df['Model'].apply(lambda x: x if x in top_models else 'Other')

# B. Limpieza de Columnas
# -----------------------
df.drop(columns=['ID', 'Model', 'Prod. year'], inplace=True)

# Levy (Impuesto): Rellenar huecos
df['Levy'] = pd.to_numeric(df['Levy'].replace('-', np.nan))
df['Levy'] = df['Levy'].fillna(df['Levy'].median())

# Mileage: Limpiar texto
df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(int)

# Doors: Corregir errores de escritura
door_mapping = {'04-may': 4, '4-5': 4, '02-mar': 2, '2-3': 2, '>5': 5}
df['Doors'] = df['Doors'].replace(door_mapping)
df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce').fillna(4).astype(int)

# C. Filtro de Outliers (Precios extremos)
# ----------------------------------------
df = df[df['Price'] > 500]
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= (Q1 - 1.5 * IQR)) & (df['Price'] <= (Q3 + 1.5 * IQR))]

df.drop_duplicates(inplace=True)

print(f"Limpieza finalizada. Datos listos: {df.shape[0]} filas.")


# ==============================================================================
# 3. VISUALIZACIÓN: ANTES VS DESPUÉS (Lo que pediste)
# ==============================================================================
print("\n--- MOSTRANDO CAMBIOS EN LOS DATOS (Cierra las ventanas para continuar) ---")

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 5)

def plot_comparison(raw_col, clean_col, title_raw, title_clean, kind='hist'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Izquierda: Datos Sucios
    if kind == 'hist':
        sns.histplot(raw_col, ax=axes[0], color='gray', bins=30)
    elif kind == 'count':
        sns.countplot(y=raw_col, ax=axes[0], color='gray', order=raw_col.value_counts().index)
    axes[0].set_title(f"ANTES: {title_raw}", fontsize=13, color='red')

    # Derecha: Datos Limpios
    if kind == 'hist':
        sns.histplot(clean_col, ax=axes[1], color='#2ecc71', bins=30)
    elif kind == 'count':
        sns.countplot(x=clean_col, ax=axes[1], color='#2ecc71')
    axes[1].set_title(f"DESPUÉS: {title_clean}", fontsize=13, color='green')
