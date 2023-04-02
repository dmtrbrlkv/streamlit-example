import pandas as pd


def preprocessing(df, features):
    UNKNOWN = 'UNKNOWN'

    categorical_features = ['make', 'model', 'trim', 'body', 'state', 'color', 'interior', 'seller']

    for column in categorical_features:
        df[column] = df[column].replace({'': pd.NA})

    for column in categorical_features:
        df[column] = df[column].str.lower()

    df[categorical_features] = df[categorical_features].fillna(UNKNOWN)

    df['saledate_dt'] = pd.to_datetime(df['saledate'], utc=True)
    df['sale_year'] = df['saledate_dt'].dt.year
    df['sale_month'] = df['saledate_dt'].dt.month
    df["sale_yearday"] = df["saledate_dt"].dt.dayofyear
    df["sale_weekday"] = df["saledate_dt"].dt.dayofweek
    df["sale_day"] = df["saledate_dt"].dt.day

    df['trim'] = df['trim'].replace({UNKNOWN.lower(): 'base'})

    body_replace = {
        'cts coupe': 'coupe',
        'cts wagon': 'wagon',
        'cts-v coupe': 'coupe',
        'cts-v wagon': 'wagon',
        'e-series van': 'van',
        'g convertible': 'convertible',
        'g coupe': 'coupe',
        'g sedan': 'sedan',
        'g37 convertible': 'convertible',
        'g37 coupe': 'coupe',
        'genesis coupe': 'coupe',
        'granturismo convertible': 'convertible',
        'koup': 'coupe',
        'promaster cargo van': 'van',
        'q60 convertible': 'convertible',
        'q60 coupe': 'coupe',
        'regular-cab': 'regular cab',
        'transit van': 'van',
        'tsx sport wagon': 'wagon',
        'ram van': 'van',
        'beetle convertible': 'convertible'
    }
    df['body'] = df['body'].replace(body_replace)

    df['color'] = df['color'].replace({'—': UNKNOWN})

    df['interior'] = df['interior'].replace({'—': UNKNOWN})

    df['seller'] = df['seller'].str.replace('-', ' ', regex=False)
    df['seller'] = df['seller'].str.replace('/', ' ', regex=False)
    df['seller'] = df['seller'].str.replace(',', ' ', regex=False)
    df['seller'] = df['seller'].str.replace('.', ' ', regex=False)
    df['seller'] = df['seller'].str.replace('(', ' ', regex=False)
    df['seller'] = df['seller'].str.replace(')', ' ', regex=False)
    df['seller'] = df['seller'].str.replace('   ', ' ', regex=False)
    df['seller'] = df['seller'].str.replace('  ', ' ', regex=False)

    df['condition_div_by_odometer'] = df['condition'] / df['odometer']
    df['age'] = df['sale_year'] - df['year']
    df['condition_div_by_age'] = df['condition'] / df['age']
    df['odometer_div_by_age'] = df['odometer'] / df['age']
    df['make_model'] = df['make'] + df['model']
    df['model_trim'] = df['model'] + df['trim']
    df['model_body'] = df['model'] + df['body']
    df['model_trim_body'] = df['model'] + df['trim'] + df['body']
    df['make_model_trim_body'] = df['make'] + df['model'] + df['trim'] + df['body']
    df['make_body'] = df['make'] + df['body']
    df['make_trim'] = df['make'] + df['trim']
    df['make_model_trim'] = df['make'] + df['model'] + df['trim']
    df['make_model_body'] = df['make'] + df['model'] + df['body']

    return df[features]
