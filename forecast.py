import numpy as np
import pandas as pd

from physics import (
    exponential_decline,
    fit_exponential
)


def prepare_time_series(df):

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.sort_values("date")

    data = data.dropna(
        subset=["date", "oil_rate"]
    )

    start_date = data["date"].min()

    data["days"] = (
        data["date"] - start_date
    ).dt.days

    return data


def dca_forecast(df, forecast_days=90):

    data = prepare_time_series(df)

    if len(data) < 10:
        return None

    fit = fit_exponential(
        data["days"].values,
        data["oil_rate"].values
    )

    if fit is None:
        return None

    last_day = data["days"].max()

    future_days = np.arange(
        last_day + 1,
        last_day + forecast_days + 1
    )

    future_rates = exponential_decline(
        future_days,
        fit["qi"],
        fit["Di"]
    )

    future_dates = pd.date_range(
        start=data["date"].max()
        + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast = pd.DataFrame({
        "date": future_dates,
        "oil_forecast": future_rates
    })

    return {
        "forecast": forecast,
        "model": "Exponential DCA",
        "qi": fit["qi"],
        "decline_rate": fit["Di"],
        "r2": fit["r2"]
    }


def calculate_prediction_range(
    forecast,
    historical_std
):

    forecast = forecast.copy()

    forecast["p50"] = forecast[
        "oil_forecast"
    ]

    forecast["p10"] = (
        forecast["oil_forecast"]
        - 1.28 * historical_std
    )

    forecast["p90"] = (
        forecast["oil_forecast"]
        + 1.28 * historical_std
    )

    forecast["p10"] = forecast[
        "p10"
    ].clip(lower=0)

    return forecast
