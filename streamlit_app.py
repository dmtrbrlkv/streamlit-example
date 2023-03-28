import gdown
import streamlit as st
from catboost import CatBoostRegressor

with st.spinner('Подготовка модели...'):
    url = 'https://drive.google.com/uc?id=1wklHdPCrnsxyF9vb6XtdrcFAWC2qPZag'
    output = 'used-cars-catboost.model'
    gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars-catboost.model')

st.title('Used car''s price prediction')
btn_features = st.button('Важность признаков')
if btn_features:
    with st.spinner('Построение графика...'):
        features = model.get_feature_importance(prettified=True).sort_values('Importances')
        st.write(features)
        ax = features.plot.barh(x='Feature Id', y='Importances')
        st.pyplot(ax.figure)

