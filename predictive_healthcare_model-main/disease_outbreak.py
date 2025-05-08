import pandas as pd
import numpy as np
from disease_preprocess import count_cases
from disease_forecast import make_disease_forecast


def detect_outbreak(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Detects disease outbreaks in a given DataFrame
    Args:
        df (pd.DataFrame): DataFrame containing disease data with 'date', 'prognosis', and 'cases' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional column 'outbreak' indicating outbreak status.
    '''
    df = df.copy()
    # Calculate average cases per prognosis
    avg_cases = df.groupby('prognosis')['cases'].transform('mean')
    # Outbreak if cases >= 2x average for that prognosis
    df['outbreak'] = (df['cases'] >= 2 * avg_cases).astype(int)

    return df


def detect_outbreak_per_day(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    '''
    Detects outbreaks per prognosis per day using the rolling mean of previous days (excluding the current day).
    Args:
        df (pd.DataFrame): DataFrame with 'date', 'prognosis', and 'cases' columns.
        window (int): Number of previous days to use for the rolling mean.
    Returns:
        pd.DataFrame: DataFrame with 'date', 'prognosis', 'cases', 'outbreak'.
    '''
    df = df.copy()

    daily = df.groupby(['date', 'prognosis'], as_index=False)['cases'].sum()
    daily = daily.sort_values(['prognosis', 'date'])

    daily['rolling_mean_prev'] = daily.groupby('prognosis')['cases'].apply(
        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
    ).reset_index(level=0, drop=True)

    daily['outbreak'] = ((daily['rolling_mean_prev'].notna()) & (
        daily['cases'] >= 2 * daily['rolling_mean_prev'])).astype(int)
    daily = daily.drop(columns=['rolling_mean_prev'])

    return daily


def predict_future_outbreaks(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    '''
    Predicts future outbreaks based on the last known cases.
    Args:
        df (pd.DataFrame): DataFrame with 'date', 'prognosis', and 'cases' columns.
        days (int): Number of days to predict.
    Returns:
        pd.DataFrame: DataFrame with predicted outbreaks.
    '''

    df = df.copy()
    historical_means = df.groupby('prognosis')['cases'].mean()

    forecasts = make_disease_forecast(df, period=days)

    results = []

    for forecast in forecasts:
        prognosis = forecast['prognosis'].unique()[0]

        forecast = forecast.rename(
            columns={'ds': 'date', 'yhat': 'cases'})

        outbreak_predictions = detect_outbreak_per_day(forecast)

        results.append(outbreak_predictions)

    return pd.concat(results, ignore_index=True)
