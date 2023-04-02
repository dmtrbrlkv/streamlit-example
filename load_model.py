import os

import gdown
import streamlit as st
from catboost import CatBoostRegressor


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists('used-cars.cdm'):
        url = 'https://drive.google.com/uc?id=1_yVEvHEF-m-H-bem_N37-swObJWIyMyk'
        output = 'used-cars.cdm'
        gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars.cdm')
    return model
