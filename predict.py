import pandas

from preprocessing import preprocessing


def predict(
        year, make, model, trim, body, transmission, state, condition, odometer, color, interior, seller, saledate,
        model_cb
):
    df = pandas.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'body': [body],
        'transmission': [transmission],
        'state': [state],
        'condition': [condition],
        'odometer': [odometer],
        'color': [color],
        'interior': [interior],
        'seller': [seller],
        'saledate': [saledate],
    })

    df = preprocessing(df, model_cb.feature_names_)
    predict = model_cb.predict(df)
    return predict[0]
