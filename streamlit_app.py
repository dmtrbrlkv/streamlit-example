import os.path

import gdown
import pandas
import streamlit as st
from catboost import CatBoostRegressor
from dateutil.utils import today

from preprocessing import preprocessing

st.title('Used car''s price prediction')


@st.cache_resource()
def load_model():
    if not os.path.exists('used-cars-catboost.model'):
        url = 'https://drive.google.com/uc?id=1wklHdPCrnsxyF9vb6XtdrcFAWC2qPZag'
        output = 'used-cars-catboost.model'
        gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars-catboost.model')
    return model
    # st.success('Готово!')


with st.spinner('Подготовка модели...') :
    model_cb = load_model()


@st.cache_data
def features_graph():
    features = model_cb.get_feature_importance(prettified=True)
    st.write(features)
    features = features.sort_values('Importances')
    ax = features.plot.barh(x='Feature Id', y='Importances')
    return ax.figure


btn_features = st.button('Важность признаков')

if btn_features:
    with st.spinner('Построение графика...'):
        fig = features_graph()
        st.pyplot(fig)
    # st.success('Готово!')

# btn_predict_form = st.button('Предсказать цену')
# if btn_predict_form:
year = st.number_input('Year', min_value=0, max_value=today().year)
make = st.text_input('Make')
model = st.text_input('Model')
trim = st.text_input('Trim')
body = st.text_input('Body')
transmission = st.radio('Transmission', ['automatic', 'manual'])
state = st.text_input('State', max_chars=2)
condition = st.number_input('Condition', min_value=1.0, max_value=5.0, step=0.5)
odometer = st.number_input('Odometer', min_value=0)
color = st.text_input('Color')
interior = st.text_input('Interior')
seller = st.text_input('Seller')
saledate = st.date_input('Sale date', today())

btn_predict = st.button('Предсказать')

if btn_predict:
    df = pandas.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'body': [body],
        'transmission': [transmission],
        'state': [state],
        'condition': [condition],
        'odometer': [odometer],
        'color': [color],
        'interior': [interior],
        'seller': [seller],
        'saledate': [saledate],
    })

    df = preprocessing(df, model_cb.feature_names_)

    predict = model_cb.predict(df)

    st.write(f'Цена: {predict[0]:.2f}')