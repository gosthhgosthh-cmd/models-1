import os
import json
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='Riesgo actuarial', layout='centered')
st.title('Predicción de riesgo actuarial-Josue David Del Cid-PTI-0620-03')

@st.cache_resource
def cargar_modelo():
    # Rutas dentro de carpeta 'models/' o directorio raíz
    if os.path.exists('models/kmeans_riesgo_actuarial.pkl'):
        pkl = 'models/kmeans_riesgo_actuarial.pkl'
        meta = 'models/model_metadata.json'
    else:
        pkl = 'kmeans_riesgo_actuarial.pkl' if os.path.exists('kmeans_riesgo_actuarial.pkl') else 'kmeans_riesgo_actuarial(2).pkl'
        meta = 'model_metadata.json' if os.path.exists('model_metadata.json') else 'model_metadata(2).json'
    
    modelo = joblib.load(pkl)

    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base():
    if os.path.exists('models/insurance.csv'):
        return pd.read_csv('models/insurance.csv')
    csv = 'insurance.csv' if os.path.exists('insurance.csv') else 'insurance(2).csv'
    return pd.read_csv(csv)

modelo, metadata = cargar_modelo()
df = cargar_base()

# Manejo seguro de llaves anidadas
kmeans_meta = metadata.get('kmeans', {})
mapa_riesgo_dict = kmeans_meta.get('mapa_riesgo', metadata.get('mapa_riesgo', {}))

mapa = {int(k): v for k, v in mapa_riesgo_dict.items()} if mapa_riesgo_dict else {}

nombre_proyecto = metadata.get('proyecto', metadata.get('nombre_modelo', 'Modelo de Riesgo Actuarial'))
st.caption(nombre_proyecto)

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
    cliente = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges
    }])

    # Asegurar el orden exacto de columnas que espera el modelo
    cols_esperadas = metadata.get('features_kmeans', list(cliente.columns))
    
    try:
        # Intenta predecir directamente
        cluster = int(modelo.predict(cliente[cols_esperadas])[0])
    except Exception:
        # Si falla por tipos categóricos, aplica encoding numérico
        cliente_enc = cliente.copy()
        cliente_enc['sex'] = cliente_enc['sex'].map({'female': 0, 'male': 1})
        cliente_enc['smoker'] = cliente_enc['smoker'].map({'no': 0, 'yes': 1})
        reg_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        cliente_enc['region'] = cliente_enc['region'].map(reg_map)
        
        cluster = int(modelo.predict(cliente_enc[cols_esperadas])[0])

    riesgo = mapa.get(cluster, f'Cluster {cluster}')

    st.subheader(f'Riesgo actuarial: {riesgo}')
    st.write(f'Cluster asignado: {cluster}')

    api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))

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
                    {"role": "system", "content": "Eres un asesor actuarial profesional."},
                    {"role": "user", "content": prompt}
                ]
            )

            texto = respuesta.choices[0].message.content
            st.info(texto)

        except Exception as e:
            st.warning(f'Error con Groq: {e}')
    else:
        st.warning('Agregue GROQ_API_KEY en los secretos de Streamlit.')

st.divider()

st.write('Vista rápida de la base principal')
st.dataframe(df.head(20), width="stretch")
