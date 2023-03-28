import streamlit as st
from catboost import CatBoostRegressor



model = CatBoostRegressor()
model.load_model('used-cars-catboost.model')
features = model.get_feature_importance(prettified=True).sort_values('Importances')
st.title('User cars features importance')
st.write(features)
ax = features.plot.barh(x='Feature Id', y='Importances')
st.pyplot(ax.figure)
