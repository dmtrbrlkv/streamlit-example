import streamlit as st
from catboost import CatBoostRegressor

import py7zr

import gdown

url = 'https://drive.google.com/uc?id=1EMQBErDSVz9Ij8HQWOQSa5yVlHG3Lw8T'
output = 'used-cars-catboost.model.7z'
gdown.download(url, output, quiet=False)

archive = py7zr.SevenZipFile('used-cars-catboost.model.7z', mode='r')
archive.extractall()
archive.close()


model = CatBoostRegressor()
model.load_model('used-cars-catboost.model')
features = model.get_feature_importance(prettified=True).sort_values('Importances')
st.title('User cars features importance')
st.write(features)
ax = features.plot.barh(x='Feature Id', y='Importances')
st.pyplot(ax.figure)
