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
df_raw = pd.read_csv('/content/car_price_prediction.csv - car_price_prediction.csv.csv') # Copia sucia para comparar
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

    plt.tight_layout()
    plt.show()

# Generar comparaciones
plot_comparison(df_raw['Price'][df_raw['Price'] < 100000], df['Price'], "Precio (Con basura)", "Precio (Limpio)")
plot_comparison(pd.to_numeric(df_raw['Levy'].replace('-', np.nan)), df['Levy'], "Levy (Con vacíos)", "Levy (Rellenado)")
plot_comparison(df_raw['Doors'], df['Doors'], "Doors (Texto sucio)", "Doors (Numérico)", kind='count')
plot_comparison(df_raw['Engine volume'], df['Engine volume'], "Motor (Texto)", "Motor (Numérico)", kind='hist')


# ==============================================================================
# 4. ENTRENAMIENTO Y RESULTADOS FINALES (Con ambas gráficas)
# ==============================================================================
print("\n--- ENTRENANDO MODELO FINAL... ---")

le = LabelEncoder()
df['Manufacturer'] = le.fit_transform(df['Manufacturer'])

cols_encode = ['Category', 'Fuel type', 'Gear box type', 'Drive wheels',
               'Leather interior', 'Wheel', 'Color', 'Model_Grouped']
df_model = pd.get_dummies(df, columns=cols_encode, drop_first=True)

X = df_model.drop(columns=['Price'])
y = df_model['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Ganador
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Métricas
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.0f}")

# --- GRÁFICAS FINALES DE EVALUACIÓN ---
plt.figure(figsize=(16, 6))

# 1. Dispersión (Scatter Plot)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#4c72b0', edgecolor='k')
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--')
plt.title('Dispersión: Predicción vs Realidad', fontsize=14)
plt.xlabel('Precio Real ($)')
plt.ylabel('Precio Predicho ($)')

# 2. Distribución (KDE Plot) - ¡LA QUE FALTABA!
plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label="Datos Reales", fill=True, color='green', alpha=0.3, linewidth=2)
sns.kdeplot(y_pred, label="Predicción IA", fill=True, color='blue', alpha=0.3, linewidth=2)
plt.title('Distribución de Precios (Curvas)', fontsize=14)
plt.xlabel('Precio ($)')
plt.ylabel('Densidad')
plt.legend()

plt.tight_layout()
plt.show()
