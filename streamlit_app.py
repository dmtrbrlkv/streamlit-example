import os.path

import gdown
import streamlit as st
from catboost import CatBoostRegressor

st.title('Used car''s price prediction')

with st.spinner('Подготовка модели...') as spinner:
    if not os.path.exists('used-cars-catboost.model'):
        url = 'https://drive.google.com/uc?id=1wklHdPCrnsxyF9vb6XtdrcFAWC2qPZag'
        output = 'used-cars-catboost.model'
        gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars-catboost.model')
    # st.success('Готово!')

btn_features = st.button('Важность признаков')
if btn_features:
    with st.spinner('Построение графика...'):
        features = model.get_feature_importance(prettified=True)
        st.write(features)
        features = features.sort_values('Importances')
        ax = features.plot.barh(x='Feature Id', y='Importances')
        st.pyplot(ax.figure)
    # st.success('Готово!')
