import gdown
import py7zr
import streamlit as st
from catboost import CatBoostRegressor

url = 'https://drive.google.com/uc?id=1wklHdPCrnsxyF9vb6XtdrcFAWC2qPZag'
output = 'used-cars-catboost.model'
gdown.download(url, output, quiet=False)

model = CatBoostRegressor()
model.load_model('used-cars-catboost.model')

btn_features = st.button('Важность признаков')
if btn_features:
    features = model.get_feature_importance(prettified=True).sort_values('Importances')
    st.title('Used car''s price prediction')
    st.write(features)
    ax = features.plot.barh(x='Feature Id', y='Importances')
    st.pyplot(ax.figure)
