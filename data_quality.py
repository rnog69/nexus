import pandas as pd
import numpy as np


def analyze_data_quality(df):

    report = {}

    report["rows"] = len(df)
    report["columns"] = len(df.columns)

    missing_cells = df.isna().sum().sum()

    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:
        missing_percentage = (
            missing_cells / total_cells
        ) * 100
    else:
        missing_percentage = 0

    report["missing_cells"] = int(missing_cells)
    report["missing_percentage"] = missing_percentage

    duplicates = df.duplicated().sum()

    report["duplicates"] = int(duplicates)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    report["numeric_columns"] = len(numeric_columns)

    outlier_count = 0

    for col in numeric_columns:

        series = df[col].dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count += (
            ((series < lower) | (series > upper))
            .sum()
        )

    report["outliers"] = int(outlier_count)

    score = 100

    score -= min(missing_percentage * 2, 30)

    if len(df) > 0:
        duplicate_percentage = duplicates / len(df) * 100
        score -= min(duplicate_percentage * 2, 20)

    if len(numeric_columns) > 0:
        outlier_percentage = (
            outlier_count /
            max(len(df) * len(numeric_columns), 1)
        ) * 100

        score -= min(outlier_percentage, 15)

    score = max(0, min(100, score))

    report["quality_score"] = score

    return report


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def fill_numeric_missing(df):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:
        df[col] = df[col].interpolate(
            method="linear",
            limit_direction="both"
        )

    return df
