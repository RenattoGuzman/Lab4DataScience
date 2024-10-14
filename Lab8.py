# Para ejecutar el programa corren en un entorno python
#  
# - "pip install streamlit pandas matplotlib seaborn"

# y luego correr el comando en la terminal
# - "streamlit run Lab8.py"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la paleta de colores
colors = ['#003087', '#6CACE4', '#FFC427', '#FFFFFF', '#0A0908']

# Cargar el conjunto de datos
df = pd.read_csv('GrammarandProductReviews.csv')

# Limpiar el dataframe (esto ya lo has hecho, pero asegúrate de que se ejecute correctamente)
df.drop_duplicates(inplace=True)
df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)

def add_black_stroke(plot):
    """Aplica un borde negro delgado a los elementos de la gráfica."""
    for patch in plot.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

# Streamlit App
st.title("Análisis de Productos y Reseñas")

# 1. Gráfico de Barras para las Marcas Más Frecuentes
st.subheader("Marcas Más Frecuentes")
brand_counts = df['brand'].value_counts()
plt.figure(figsize=(10, 6))
plot = sns.barplot(x=brand_counts.index[:10], y=brand_counts.values[:10], palette=colors)
add_black_stroke(plot)
plt.title('Frecuencia de Productos por Marca')
plt.xlabel('Marca')
plt.ylabel('Número de Productos')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()

# 2. Histograma de Reseñas por Calificación
st.subheader("Histograma de Reseñas por Calificación")
plt.figure(figsize=(10, 6))
plot = sns.histplot(df['reviews.rating'], bins=5, kde=False, color=colors[1], edgecolor='black', linewidth=0.5)
plt.title('Distribución de Calificaciones de Reseñas')
plt.xlabel('Calificación')
plt.ylabel('Frecuencia')
st.pyplot(plt)
plt.clf()

# 3. Gráfico Circular para Categorías de Productos
st.subheader("Distribución de Productos por Categorías")
category_counts = df['categories'].str.split(',').explode().value_counts()
plt.figure(figsize=(10, 6))
plt.pie(category_counts[:5], labels=category_counts.index[:5], autopct='%1.1f%%', colors=colors, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
plt.title('Distribución de Productos por Categorías Principales')
st.pyplot(plt)
plt.clf()


# 4. Gráfico de Dispersión para la Relación entre Fecha de Agregado y Calificación
st.subheader("Relación entre Fecha de Agregado y Calificación")
df['dateAdded'] = pd.to_datetime(df['dateAdded'])
plt.figure(figsize=(10, 6))
sns.scatterplot(x='dateAdded', y='reviews.rating', data=df, color=colors[0])
plt.title('Relación entre Fecha de Agregado y Calificación')
plt.xlabel('Fecha de Agregado')
plt.ylabel('Calificación')
st.pyplot(plt)
plt.clf()  # Limpia la figura

# 5. Gráfico de Barras para el Número de Reseñas Útiles por Marca
st.subheader("Número de Reseñas Útiles por Marca")
df['reviews.numHelpful'] = pd.to_numeric(df['reviews.numHelpful'], errors='coerce')
helpful_counts = df.groupby('brand')['reviews.numHelpful'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plot = sns.barplot(x=helpful_counts.index[:10], y=helpful_counts.values[:10], palette=colors)
add_black_stroke(plot)
plt.title('Número de Reseñas Útiles por Marca')
plt.xlabel('Marca')
plt.ylabel('Total de Reseñas Útiles')
plt.xticks(rotation=45)
st.pyplot(plt)
plt.clf()
