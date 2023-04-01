import os

import gdown
import streamlit as st
from catboost import CatBoostRegressor


@st.cache_resource()
def load_model():
    if not os.path.exists('used-cars-catboost.model'):
        url = 'https://drive.google.com/uc?id=1wklHdPCrnsxyF9vb6XtdrcFAWC2qPZag'
        output = 'used-cars-catboost.model'
        gdown.download(url, output, quiet=False)

    model = CatBoostRegressor()
    model.load_model('used-cars-catboost.model')
    return model
