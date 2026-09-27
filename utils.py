import pandas as pd
import numpy as np


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    return df


def find_column(df, candidates):
    columns = {str(c).lower().strip(): c for c in df.columns}

    for candidate in candidates:
        candidate = candidate.lower().strip()

        if candidate in columns:
            return columns[candidate]

    for column in df.columns:
        column_lower = str(column).lower()

        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


def convert_numeric(df, columns):
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def safe_percentage(value):
    try:
        return f"{value:.1f}%"
    except:
        return "N/A"


def generate_sample_data(n_wells=8, days=365):

    np.random.seed(42)

    dates = pd.date_range(
        end=pd.Timestamp.today(),
        periods=days,
        freq="D"
    )

    rows = []

    for i in range(1, n_wells + 1):

        well = f"NEXUS-{i:02d}"

        qi = np.random.uniform(1200, 2500)
        decline = np.random.uniform(0.0005, 0.002)

        initial_pressure = np.random.uniform(2800, 3500)

        water_initial = np.random.uniform(0.10, 0.30)
        water_growth = np.random.uniform(0.0002, 0.001)

        for day, date in enumerate(dates):

            oil = qi * np.exp(-decline * day)

            noise = np.random.normal(0, oil * 0.025)
            oil += noise

            oil = max(oil, 50)

            water_cut = water_initial + water_growth * day

            # Create abnormal behavior in one well
            if i == 3 and 220 <= day <= 300:
                water_cut += 0.20
                oil *= 0.82

            water_cut = min(water_cut, 0.90)

            water = oil * water_cut / max(1 - water_cut, 0.05)

            gas = oil * np.random.uniform(500, 900)

            pressure = (
                initial_pressure
                - day * np.random.uniform(0.8, 1.5)
            )

            if i == 3 and 220 <= day <= 300:
                pressure -= 150

            choke = np.random.uniform(35, 75)

            whp = pressure * np.random.uniform(0.15, 0.30)

            bhp = pressure * np.random.uniform(0.80, 0.95)

            rows.append({
                "date": date,
                "well": well,
                "oil_rate": oil,
                "water_rate": water,
                "gas_rate": gas,
                "pressure": pressure,
                "choke": choke,
                "whp": whp,
                "bhp": bhp
            })

    return pd.DataFrame(rows)
