import numpy as np


def calculate_trend(series):

    if series is None or len(series) < 2:
        return 0

    x = np.arange(len(series))

    slope = np.polyfit(
        x,
        series.values,
        1
    )[0]

    return slope


def investigate_well(df):

    results = []

    if len(df) < 5:
        return results

    df = df.sort_values("date")

    # ---------------------------------------------------------
    # PRODUCTION DECLINE
    # ---------------------------------------------------------

    if "oil_rate" in df.columns:

        first = df["oil_rate"].iloc[0]
        last = df["oil_rate"].iloc[-1]

        if first > 0:

            decline = (
                (first - last) /
                first
            ) * 100

            if decline > 10:

                results.append({
                    "hypothesis":
                        "Production decline",

                    "evidence":
                        f"Oil rate declined "
                        f"{decline:.1f}% over the "
                        f"analysis period.",

                    "severity":
                        "High" if decline > 20
                        else "Medium",

                    "confidence":
                        min(95, 50 + decline)
                })

    # ---------------------------------------------------------
    # WATER BREAKTHROUGH
    # ---------------------------------------------------------

    if "water_cut" in df.columns:

        first_wc = df["water_cut"].iloc[0]
        last_wc = df["water_cut"].iloc[-1]

        increase = (
            last_wc - first_wc
        ) * 100

        if increase > 5:

            results.append({
                "hypothesis":
                    "Possible water breakthrough",

                "evidence":
                    f"Water cut increased "
                    f"by {increase:.1f} percentage "
                    f"points.",

                "severity":
                    "High" if increase > 15
                    else "Medium",

                "confidence":
                    min(95, 55 + increase)
            })

    # ---------------------------------------------------------
    # PRESSURE DECLINE
    # ---------------------------------------------------------

    if "pressure" in df.columns:

        first_p = df["pressure"].iloc[0]
        last_p = df["pressure"].iloc[-1]

        if first_p > 0:

            decline = (
                (first_p - last_p) /
                first_p
            ) * 100

            if decline > 5:

                results.append({
                    "hypothesis":
                        "Reservoir pressure decline",

                    "evidence":
                        f"Pressure decreased "
                        f"by {decline:.1f}% "
                        f"over the analysis period.",

                    "severity":
                        "Medium",

                    "confidence":
                        min(90, 50 + decline)
                })

    # ---------------------------------------------------------
    # SURFACE CONSTRAINT
    # ---------------------------------------------------------

    if (
        "whp" in df.columns and
        "oil_rate" in df.columns
    ):

        whp_slope = calculate_trend(
            df["whp"].dropna()
        )

        oil_slope = calculate_trend(
            df["oil_rate"].dropna()
        )

        if whp_slope > 0 and oil_slope < 0:

            results.append({
                "hypothesis":
                    "Possible surface/backpressure constraint",

                "evidence":
                    "Wellhead pressure is increasing "
                    "while oil rate is decreasing.",

                "severity":
                    "Medium",

                "confidence":
                    65
            })

    # ---------------------------------------------------------
    # ARTIFICIAL LIFT
    # ---------------------------------------------------------

    if "choke" in df.columns:

        choke_range = (
            df["choke"].max() -
            df["choke"].min()
        )

        if choke_range > 30:

            results.append({
                "hypothesis":
                    "Operating-condition change",

                "evidence":
                    "Large choke variation detected "
                    "during the analysis period.",

                "severity":
                    "Low/Medium",

                "confidence":
                    55
            })

    return results
