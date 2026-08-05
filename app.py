import os
import json
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

# Configuración inicial de la página
st.set_page_config(page_title='Riesgo actuarial', layout='centered')
st.title('Predicción de riesgo actuarial-Josue David Del Cid-PTI-0620-03')

@st.cache_resource
def cargar_modelo():
    # Búsqueda dinámica de archivos para evitar fallos por renombramiento local vs repo
    pkl = 'kmeans_riesgo_actuarial.pkl' if os.path.exists('kmeans_riesgo_actuarial.pkl') else 'kmeans_riesgo_actuarial(2).pkl'
    meta = 'model_metadata.json' if os.path.exists('model_metadata.json') else 'model_metadata(2).json'
    
    modelo = joblib.load(pkl)

    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base():
    csv = 'insurance.csv' if os.path.exists('insurance.csv') else 'insurance(2).csv'
    return pd.read_csv(csv)

# Carga de modelo y dataset
modelo, metadata = cargar_modelo()
df = cargar_base()

# Manejo seguro y defensivo del mapa de riesgo dentro de 'kmeans'
kmeans_meta = metadata.get('kmeans', {})
mapa_riesgo_dict = kmeans_meta.get('mapa_riesgo', metadata.get('mapa_riesgo', {}))

# Conversión segura de claves a entero
mapa = {int(k): v for k, v in mapa_riesgo_dict.items()} if mapa_riesgo_dict else {}

# Nombre del proyecto / modelo seguro
nombre_proyecto = metadata.get('proyecto', metadata.get('nombre_modelo', 'Modelo de Riesgo Actuarial'))
st.caption(nombre_proyecto)

# Formulario para recabar parámetros
with st.form('datos'):
    col1, col2 = st.columns(2)

    age = col1.number_input('Edad', 18, 100, 35)
    sex = col2.selectbox('Sexo', sorted(df['sex'].unique()))

    bmi = col1.number_input('BMI', 10.0, 60.0, 28.0)
    children = col2.number_input('Hijos', 0, 10, 1)

    smoker = col1.selectbox('Fumador', sorted(df['smoker'].unique()))
    region = col2.selectbox('Región', sorted(df['region'].unique()))

    charges = st.number_input(
        'Cargos médicos estimados',
        0.0,
        100000.0,
        12000.0
    )

    enviar = st.form_submit_button('Evaluar')

if enviar:
    # Creación del DataFrame de entrada con el orden estándar de columnas
    cliente = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges
    }])

    try:
        cluster = int(modelo.predict(cliente)[0])
        riesgo = mapa.get(cluster, f'Cluster {cluster}')

        st.subheader(f'Riesgo actuarial: {riesgo}')
        st.write(f'Cluster asignado: {cluster}')

        # Evaluación con la API de Groq
        api_key = st.secrets.get(
            'GROQ_API_KEY',
            os.getenv('GROQ_API_KEY', '')
        )

        if api_key:
            prompt = f'''
            Actúa como analista actuarial.

            Explica brevemente el resultado y brinda 3 recomendaciones prudentes.

            Datos:
            edad={age}
            sexo={sex}
            bmi={bmi}
            hijos={children}
            fumador={smoker}
            región={region}
            cargos={charges}

            Resultado:
            cluster={cluster}
            riesgo={riesgo}
            '''

            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )

                respuesta = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un asesor actuarial profesional."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                texto = respuesta.choices[0].message.content
                st.info(texto)

            except Exception as e:
                st.warning(f'Error con Groq: {e}')

        else:
            st.warning('Agregue GROQ_API_KEY en los secretos de Streamlit.')

    except Exception as err:
        st.error(f"Error al ejecutar la predicción con el modelo: {err}")

st.divider()

st.write('Vista rápida de la base principal')

# Actualización del parámetro deprecado 'use_container_width' a 'width="stretch"'
st.dataframe(
    df.head(20),
    width="stretch"
)
