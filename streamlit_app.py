from datetime import date

import streamlit as st
from dateutil.utils import today

from features_graph import features_graph
from load_model import load_model
from predict import predict

st.title('Used car''s price prediction')

with st.spinner('Подготовка модели...'):
    model_cb = load_model()

btn_features = st.button('Важность признаков')

if btn_features:
    with st.spinner('Построение графика...'):
        fig = features_graph(model_cb)
        st.pyplot(fig)

year = st.number_input('Year', min_value=1900, max_value=today().year, value=2014)
make = st.text_input('Make', value='Ford')
model = st.text_input('Model', value='Fusion')
trim = st.text_input('Trim', value='SE')
body = st.text_input('Body', value='Sedan')
transmission = st.radio('Transmission', ['automatic', 'manual'])
state = st.text_input('State', max_chars=2, value='mo')
condition = st.number_input('Condition', min_value=1.0, max_value=5.0, step=0.5, value=3.5)
odometer = st.number_input('Odometer', min_value=0, step=10000, value=31000)
color = st.text_input('Color', value='black')
interior = st.text_input('Interior', value='black')
seller = st.text_input('Seller', 'ars/avis budget group')
saledate = st.date_input('Sale date', value=date(2015, 2, 25))

btn_predict = st.button('Предсказать')

if btn_predict:
    predict = predict(
        year, make, model, trim, body, transmission, state, condition, odometer, color, interior, seller, saledate,
        model_cb
    )
    st.write(f'Цена: {predict:.2f} $')
