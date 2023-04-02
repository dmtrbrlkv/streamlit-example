import os

import gdown
import streamlit as st
from catboost import CatBoostRegressor


@st.cache_resource()
def download_model():
    output = 'used-cars.cdm'
    if not os.path.exists(output):
        url = 'https://drive.google.com/uc?id=13VwUzi_NrOwFlBjwB8p3tX8XQE9P34ZP'
        gdown.download(url, output, quiet=False)
    return output


@st.cache_resource()
def load_model(model_file):
    model = CatBoostRegressor()
    model.load_model(model_file)
    return model
