import streamlit as st


def features_graph(model_cb):
    features = model_cb.get_feature_importance(prettified=True)
    st.write(features)
    features = features.sort_values('Importances')
    ax = features.plot.barh(x='Feature Id', y='Importances')
    return ax.figure
