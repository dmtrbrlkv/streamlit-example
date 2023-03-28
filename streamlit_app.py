import os.path

import gdown
import streamlit as st
from catboost import CatBoostRegressor

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
    model = load_model()


@st.cache_data
def features_graph():
    features = model.get_feature_importance(prettified=True)
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
