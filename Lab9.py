import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Cargar los datos
@st.cache_data
def load_data():
    data = pd.read_csv('GrammarandProductReviews.csv')
    data = data.drop_duplicates()
    data = data.dropna(subset=['reviews.text', 'reviews.rating'])
    return data

data = load_data()



# Título
st.title("Cuadro de Mando Interactivo - Análisis de Reviews de Productos")

# Sección 1: Exploración de Datos
st.subheader("1. Exploración de Datos")

# Selección de columnas para explorar
columns = st.multiselect("Seleccione las columnas que desea explorar:", data.columns.tolist(), default=['brand', 'reviews.rating', 'reviews.text'])

# Muestra los datos seleccionados
st.write(data[columns].head(10))

# Sección 2: Modelos de Predicción / Clasificación
st.subheader("2. Modelos de Predicción y Clasificación")

# Selección de la columna objetivo
target_col = st.selectbox("Seleccione la columna objetivo para predicción/clasificación:", ['reviews.rating'])

# Selección de características para el modelo
feature_cols = st.multiselect("Seleccione las características para el modelo:", data.columns.tolist(), default=['reviews.didPurchase', 'reviews.numHelpful'])

# Separación de datos en train/test
X = data[feature_cols].fillna(0)
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos de predicción simples
st.write("Seleccione un modelo para ver los resultados:")
model_option = st.selectbox("Modelo", ["Regresión Logística", "Random Forest", "SVM"])

if model_option == "Regresión Logística":
    model = LogisticRegression()
elif model_option == "Random Forest":
    model = RandomForestClassifier()
else:
    model = SVC()

# Entrenar el modelo y hacer predicciones
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.write(f"Precisión del modelo {model_option}: {accuracy:.2f}")

# Sección 3: Visualizaciones Interactivas
st.subheader("3. Visualizaciones Interactivas")

# Visualización 1: Gráfico de dispersión interactivo
st.write("Gráfico de dispersión interactivo - Calificaciones vs Longitud de Reseñas")

# Calcular la longitud de las reseñas
data['review_length'] = data['reviews.text'].apply(lambda x: len(x))

fig_scatter = px.scatter(data, x='review_length', y='reviews.rating', color='brand', hover_data=['reviews.text'], title="Calificación vs Longitud de Reseñas")
st.plotly_chart(fig_scatter)

# Visualización 2: Gráfico de barras interactivo (Número de reseñas por marca)
st.write("Gráfico de barras interactivo - Número de Reseñas por Marca")

top_brands = data['brand'].value_counts().head(10)
fig_bar = px.bar(top_brands, x=top_brands.index, y=top_brands.values, labels={'y':'Número de Reseñas', 'x':'Marca'}, title="Top 10 Marcas con Más Reseñas")
st.plotly_chart(fig_bar)

# Visualización 3: Gráfico de caja interactivo (Distribución de calificaciones por marca)
st.write("Gráfico de caja interactivo - Distribución de Calificaciones por Marca")

fig_box = px.box(data, x='brand', y='reviews.rating', points="all", title="Distribución de Calificaciones por Marca", color='brand')
st.plotly_chart(fig_box)

# Visualización 4: Histograma interactivo (Distribución de Calificaciones)
st.write("Histograma interactivo - Distribución de Calificaciones")

fig_hist = px.histogram(data, x='reviews.rating', nbins=10, title="Distribución de Calificaciones de Reseñas", labels={'x':'Calificación', 'y':'Frecuencia'})
st.plotly_chart(fig_hist)

# Sección 4: Filtrar visualizaciones por calificación
st.subheader("4. Filtros de Visualización")

# Control deslizante para filtrar por calificación
rating_filter = st.slider("Filtrar por calificación mínima de reseñas", 1, 5, 3)
filtered_data = data[data['reviews.rating'] >= rating_filter]

# Mostrar datos filtrados
st.write(f"Mostrando datos con calificación mayor o igual a {rating_filter}")
st.write(filtered_data.head(10))

# Gráfico de dispersión filtrado
st.write("Gráfico de dispersión (Fecha de agregado vs Calificación)")

fig_filtered_scatter = px.scatter(filtered_data, x='dateAdded', y='reviews.rating', color='brand', hover_data=['reviews.text'], title=f"Calificaciones filtradas por {rating_filter} y superiores")
st.plotly_chart(fig_filtered_scatter)

colors = ['#003087', '#6CACE4', '#FFC427', '#FFFFFF', '#0A0908']

# Visualización 5: Gráfico Circular para Categorías de Productos
st.subheader("5. Distribución de Productos por Categorías")

# Procesar las categorías
category_counts = data['categories'].str.split(',').explode().value_counts()

# Seleccionar los 5 principales
top_categories = category_counts.head(5)

# Crear gráfico circular interactivo con Plotly
fig_pie = px.pie(
    top_categories, 
    values=top_categories.values, 
    names=top_categories.index, 
    title='Distribución de Productos por Categorías Principales',
    color_discrete_sequence=colors
)

st.plotly_chart(fig_pie)


# Visualización 6: Gráfico de Barras para las Marcas Más Frecuentes
st.subheader("6. Marcas Más Frecuentes")

# Calcular las 10 marcas más frecuentes
brand_counts = data['brand'].value_counts().head(10)

# Crear gráfico de barras interactivo con Plotly
fig_brand_bar = px.bar(
    brand_counts,
    x=brand_counts.index,
    y=brand_counts.values,
    labels={'x': 'Marca', 'y': 'Número de Productos'},
    title='Frecuencia de Productos por Marca',
    color_discrete_sequence=colors
)

# Ajustar rotación de las etiquetas del eje x
fig_brand_bar.update_layout(xaxis_tickangle=45)

# Mostrar gráfico en Streamlit
st.plotly_chart(fig_brand_bar)

# Visualización 7: Gráfico de Barras para las Top 10 Ciudades con Más Reseñas
st.subheader("7. Top 10 Ciudades con Más Reseñas")

# Calcular las 10 ciudades más frecuentes
city_counts = data['reviews.userCity'].value_counts().head(10)

# Colores personalizados

# Crear gráfico de barras interactivo con Plotly
fig_city_bar = px.bar(
    city_counts,
    x=city_counts.index,
    y=city_counts.values,
    labels={'x': 'Ciudad', 'y': 'Número de Reseñas'},
    title='Top 10 Ciudades con Más Reseñas',
    color_discrete_sequence=[colors[2]]  # Color personalizado
)

# Ajustar la rotación de las etiquetas del eje X
fig_city_bar.update_layout(xaxis_tickangle=45)

# Mostrar gráfico en Streamlit
st.plotly_chart(fig_city_bar)


# Visualización 8: Histograma de Categorías Más Comunes por Ciudad
st.subheader("8. Histograma de Categorías por Ciudad")

# Calcular las 10 ciudades con más reseñas
top_cities = data['reviews.userCity'].value_counts().head(10).index.tolist()

# Crear checkboxes para seleccionar las ciudades
selected_cities = st.multiselect(
    "Seleccione las ciudades a incluir:",
    options=top_cities,
    default='Chicago'
)

# Filtrar los datos según las ciudades seleccionadas
filtered_data = data[data['reviews.userCity'].isin(selected_cities)]

# Calcular las categorías más comunes en las ciudades seleccionadas
category_counts = (
    filtered_data['categories']
    .str.split(',')
    .explode()
    .value_counts()
    .head(10)  # Mostrar solo las 10 categorías más comunes
)

# Crear el histograma interactivo con Plotly
fig_hist = px.bar(
    category_counts,
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Categoría', 'y': 'Frecuencia'},
    title=f'Categorías Más Comunes en Ciudades Seleccionadas',
    color_discrete_sequence=[colors[1]]
)

# Ajustar la rotación de las etiquetas del eje X
fig_hist.update_layout(xaxis_tickangle=45)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig_hist)
