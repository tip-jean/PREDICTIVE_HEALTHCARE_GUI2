from prophet import Prophet
import pandas as pd
from disease_preprocess import count_cases


def make_disease_forecast(df: pd.DataFrame, period: int = 7) -> list:
    '''
    Forecasts the number of cases for each disease in the dataset for a given period.
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'date' and 'prognosis' columns.
        period (int): Number of days to forecast into the future.
    Returns:
        forecasts (list): List of DataFrames containing the forecasted cases for each disease.
    '''

    unique_diseases = df['prognosis'].unique()

    forecasts = []

    for disease in unique_diseases:
        disease_data = df[df['prognosis'] == disease]
        disease_data = disease_data.rename(
            columns={'date': 'ds', 'cases': 'y'})

        if disease_data.dropna().shape[0] < 2:
            continue

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        model.fit(disease_data)

        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        forecast['prognosis'] = disease
        forecast['yhat'] = forecast['yhat'].round(0).astype(int)

        forecast = forecast[forecast['ds'] > disease_data['ds'].max()]
        forecast = forecast[['ds', 'yhat', 'prognosis']]
        forecast = forecast.rename(
            columns={'ds': 'date', 'yhat': 'cases'})

        forecast.reset_index(drop=True, inplace=True)

        forecasts.append(forecast)

    return forecasts
