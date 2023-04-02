import os

import gdown
import streamlit as st
from catboost import CatBoostRegressor


@st.cache_resource()
def load_model():
    if not os.path.exists('used-cars.cdm'):
        url = 'https://drive.google.com/uc?id=13VwUzi_NrOwFlBjwB8p3tX8XQE9P34ZP'
        output = 'used-cars.cdm'
        gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars.cdm')
    return model
