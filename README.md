# Lab9 - Análisis de Reviews de Productos

### Este proyecto es una aplicación interactiva de Streamlit para analizar reseñas de productos. Incluye visualizaciones y modelos de predicción que ayudan a comprender mejor los datos.

 
### Integrantes:

- David Aragón
- Renatto Guzmán


## 🚀 Instrucciones para ejecutar el proyecto
Sigue los pasos a continuación para instalar las dependencias y correr el programa:

### 1. Clonar el repositorio
Si aún no tienes el código, clónalo en tu máquina:

```bash
git clone https://github.com/RenattoGuzman/Lab4DataScience/tree/lab9
cd lab9
```

### 2. Crear un entorno virtual
Crea un entorno virtual para instalar las dependencias de forma aislada.

<strong> Windows:</strong>
```bash
python -m venv venv
.\venv\Scripts\activate
```
<strong> macOS/Linux:</strong>

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
Instala las dependencias necesarias usando el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
Inicia la aplicación de Streamlit con el siguiente comando:

```bash
streamlit run Lab9.py
```

### 5. Navegar en la aplicación
Después de ejecutar el comando anterior, se abrirá una ventana en tu navegador en la dirección:

```arduino
http://localhost:8501
```

Si no se abre automáticamente, copia y pega esta URL en tu navegador.

# 🛠 Requisitos del Sistema

- Python 3.7 o superior

- pip (gestor de paquetes de Python)

# 📄 Estructura del Proyecto

```bash
📦 lab9/
├── venv/                         # Entorno virtual (opcional)
├── Lab9.py                       # Archivo principal de la aplicación
├── requirements.txt              # Dependencias del proyecto
├── GrammarandProductReviews.csv  # Dataset utilizado
└── README.md                     # Instrucciones para correr el programa
```