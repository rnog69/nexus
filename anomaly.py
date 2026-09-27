import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def statistical_anomalies(
    df,
    column,
    window=14,
    threshold=3
):

    data = df.copy()

    rolling_mean = (
        data[column]
        .rolling(window)
        .mean()
    )

    rolling_std = (
        data[column]
        .rolling(window)
        .std()
    )

    z_score = (
        data[column] -
        rolling_mean
    ) / rolling_std.replace(0, np.nan)

    data["z_score"] = z_score

    data["statistical_anomaly"] = (
        z_score.abs() > threshold
    )

    return data


def isolation_forest_anomalies(
    df,
    features
):

    data = df.copy()

    available = [
        col for col in features
        if col in data.columns
    ]

    if len(available) < 2:
        data["ml_anomaly"] = False
        data["anomaly_score"] = 0

        return data

    model_data = data[
        available
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(model_data) < 20:
        data["ml_anomaly"] = False
        data["anomaly_score"] = 0

        return data

    scaler = StandardScaler()

    X = scaler.fit_transform(
        model_data
    )

    model = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    predictions = model.fit_predict(X)

    scores = model.decision_function(X)

    data["ml_anomaly"] = False
    data["anomaly_score"] = 0.0

    data.loc[
        model_data.index,
        "ml_anomaly"
    ] = predictions == -1

    data.loc[
        model_data.index,
        "anomaly_score"
    ] = -scores

    return data


def combined_anomaly_score(df):

    data = df.copy()

    data["anomaly_flag"] = False

    if "statistical_anomaly" in data.columns:

        data["anomaly_flag"] |= (
            data["statistical_anomaly"]
        )

    if "ml_anomaly" in data.columns:

        data["anomaly_flag"] |= (
            data["ml_anomaly"]
        )

    return data


def summarize_anomalies(df):

    total = len(df)

    anomalies = int(
        df["anomaly_flag"].sum()
    ) if "anomaly_flag" in df else 0

    if total == 0:
        percentage = 0
    else:
        percentage = anomalies / total * 100

    return {
        "total_records": total,
        "anomalies": anomalies,
        "percentage": percentage
    }
