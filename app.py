import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    clean_column_names,
    generate_sample_data
)

from data_quality import (
    analyze_data_quality,
    remove_duplicates,
    fill_numeric_missing
)

from physics import (
    add_engineering_columns
)

from forecast import (
    dca_forecast,
    calculate_prediction_range
)

from anomaly import (
    statistical_anomalies,
    isolation_forest_anomalies,
    combined_anomaly_score,
    summarize_anomalies
)

from root_cause import (
    investigate_well
)

from scenario import (
    intervention_scenario
)

from economics import (
    economic_summary
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXUS | Energy Engineering Command Center",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.nexus-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
}

.nexus-subtitle {
    font-size: 17px;
    opacity: 0.7;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #333;
    background: rgba(255,255,255,0.03);
}

.alert-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #555;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="nexus-title">🛢️ NEXUS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nexus-subtitle">'
    'Autonomous Energy Engineering Command Center'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("NEXUS CONTROL")

mode = st.sidebar.radio(
    "Data Source",
    [
        "Synthetic Field",
        "Upload CSV/XLSX"
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

if mode == "Upload CSV/XLSX":

    uploaded = st.sidebar.file_uploader(
        "Upload field data",
        type=["csv", "xlsx"]
    )

    if uploaded is None:

        st.info(
            "Upload a CSV/XLSX file or switch "
            "to Synthetic Field."
        )

        st.stop()

    try:

        if uploaded.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded)

        else:

            df = pd.read_excel(uploaded)

    except Exception as e:

        st.error(
            f"Could not read file: {e}"
        )

        st.stop()

else:

    df = generate_sample_data()


# ============================================================
# DATA NORMALIZATION
# ============================================================

df = clean_column_names(df)

df = remove_duplicates(df)

df = fill_numeric_missing(df)

df = add_engineering_columns(df)


# ============================================================
# DATE
# ============================================================

if "date" in df.columns:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR WELL SELECTION
# ============================================================

if "well" not in df.columns:

    st.error(
        "Dataset requires a 'well' column."
    )

    st.stop()

wells = sorted(
    df["well"].dropna().unique()
)

selected_well = st.sidebar.selectbox(
    "Select Well",
    wells
)

well_df = df[
    df["well"] == selected_well
].copy()

well_df = well_df.sort_values(
    "date"
)


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "NEXUS Modules",
    [
        "Command Center",
        "Data Quality",
        "Well Investigation",
        "Anomaly Detection",
        "Production Forecast",
        "DCA",
        "Scenario Engine",
        "Engineering Data"
    ]
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.header("Field Command Center")

    latest = (
        df.sort_values("date")
        .groupby("well")
        .tail(1)
    )

    total_oil = latest[
        "oil_rate"
    ].sum() if "oil_rate" in latest else 0

    total_water = latest[
        "water_rate"
    ].sum() if "water_rate" in latest else 0

    avg_pressure = latest[
        "pressure"
    ].mean() if "pressure" in latest else 0

    avg_wc = latest[
        "water_cut"
    ].mean() * 100 if "water_cut" in latest else 0

    # ----------------------------
    # KPI ROW
    # ----------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "CURRENT OIL",
        f"{total_oil:,.0f} BOPD"
    )

    c2.metric(
        "WATER",
        f"{total_water:,.0f} BWPD"
    )

    c3.metric(
        "AVG PRESSURE",
        f"{avg_pressure:,.0f} psi"
    )

    c4.metric(
        "AVG WATER CUT",
        f"{avg_wc:.1f}%"
    )

    c5.metric(
        "ACTIVE WELLS",
        f"{len(latest)}"
    )

    st.divider()

    # ----------------------------
    # PRODUCTION TREND
    # ----------------------------

    if "date" in df.columns:

        production = (
            df.groupby("date")[
                "oil_rate"
            ].sum()
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=production["date"],
                y=production["oil_rate"],
                mode="lines",
                name="Oil Production"
            )
        )

        fig.update_layout(
            title="Field Oil Production",
            xaxis_title="Date",
            yaxis_title="BOPD",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # WELL WATCHLIST
    # ----------------------------

    st.subheader("Well Watchlist")

    watchlist = []

    for well in wells:

        temp = df[
            df["well"] == well
        ].sort_values("date")

        if len(temp) < 5:
            continue

        first_oil = temp[
            "oil_rate"
        ].iloc[0]

        last_oil = temp[
            "oil_rate"
        ].iloc[-1]

        decline = (
            (first_oil - last_oil) /
            first_oil * 100
        )

        current_wc = (
            temp["water_cut"].iloc[-1]
            * 100
        )

        if decline > 20 or current_wc > 60:
            risk = "HIGH"

        elif decline > 10 or current_wc > 40:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        health = max(
            0,
            100 - decline * 1.5
        )

        watchlist.append({
            "Well": well,
            "Oil Rate":
                round(last_oil, 0),

            "Water Cut":
                f"{current_wc:.1f}%",

            "Decline":
                f"{decline:.1f}%",

            "Health":
                round(health, 0),

            "Risk":
                risk
        })

    watchlist_df = pd.DataFrame(
        watchlist
    )

    st.dataframe(
        watchlist_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.header("Data Quality Engine")

    report = analyze_data_quality(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{report['rows']:,}"
    )

    c2.metric(
        "Columns",
        report["columns"]
    )

    c3.metric(
        "Missing",
        f"{report['missing_percentage']:.2f}%"
    )

    c4.metric(
        "Quality Score",
        f"{report['quality_score']:.0f}/100"
    )

    st.divider()

    st.subheader("Data Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader(
        "Missing Values by Column"
    )

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(missing)


# ============================================================
# WELL INVESTIGATION
# ============================================================

elif page == "Well Investigation":

    st.header(
        f"🔍 Well Investigation — {selected_well}"
    )

    if len(well_df) < 5:

        st.warning(
            "Not enough data for investigation."
        )

        st.stop()

    first = well_df.iloc[0]
    last = well_df.iloc[-1]

    # Production change

    oil_change = (
        (last["oil_rate"] -
         first["oil_rate"])
        / first["oil_rate"]
    ) * 100

    water_change = (
        (last["water_cut"] -
         first["water_cut"])
    ) * 100

    pressure_change = (
        (last["pressure"] -
         first["pressure"])
        / first["pressure"]
    ) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Oil Change",
        f"{oil_change:+.1f}%"
    )

    c2.metric(
        "Water Cut Change",
        f"{water_change:+.1f} points"
    )

    c3.metric(
        "Pressure Change",
        f"{pressure_change:+.1f}%"
    )

    st.divider()

    # ----------------------------
    # PRODUCTION
    # ----------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=well_df["date"],
            y=well_df["oil_rate"],
            mode="lines",
            name="Oil Rate"
        )
    )

    fig.update_layout(
        title="Oil Production",
        yaxis_title="BOPD"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------
    # WATER CUT
    # ----------------------------

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=well_df["date"],
            y=well_df["water_cut"] * 100,
            mode="lines",
            name="Water Cut"
        )
    )

    fig2.update_layout(
        title="Water Cut",
        yaxis_title="Water Cut (%)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ----------------------------
    # ROOT CAUSE
    # ----------------------------

    st.subheader(
        "Root-Cause Hypotheses"
    )

    hypotheses = investigate_well(
        well_df
    )

    if not hypotheses:

        st.success(
            "No major rule-based hypothesis "
            "was detected."
        )

    for item in hypotheses:

        st.markdown(
            f"""
            ### {item['hypothesis']}

            **Severity:** {item['severity']}

            **Confidence:** {item['confidence']:.0f}%

            **Evidence:** {item['evidence']}
            """
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header(
        f"🚨 Anomaly Detection — {selected_well}"
    )

    features = [
        "oil_rate",
        "water_rate",
        "gas_rate",
        "pressure",
        "water_cut",
        "choke",
        "whp",
        "bhp"
    ]

    available = [
        x for x in features
        if x in well_df.columns
    ]

    if "oil_rate" in well_df.columns:

        well_df = statistical_anomalies(
            well_df,
            "oil_rate"
        )

    well_df = isolation_forest_anomalies(
        well_df,
        available
    )

    well_df = combined_anomaly_score(
        well_df
    )

    summary = summarize_anomalies(
        well_df
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        summary["total_records"]
    )

    c2.metric(
        "Anomalies",
        summary["anomalies"]
    )

    c3.metric(
        "Anomaly Rate",
        f"{summary['percentage']:.1f}%"
    )

    # ----------------------------
    # GRAPH
    # ----------------------------

    fig = go.Figure()

    normal = well_df[
        ~well_df["anomaly_flag"]
    ]

    anomalous = well_df[
        well_df["anomaly_flag"]
    ]

    fig.add_trace(
        go.Scatter(
            x=normal["date"],
            y=normal["oil_rate"],
            mode="lines",
            name="Normal"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalous["date"],
            y=anomalous["oil_rate"],
            mode="markers",
            name="Anomaly"
        )
    )

    fig.update_layout(
        title="Production Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Oil Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Events")

    st.dataframe(
        anomalous,
        use_container_width=True
    )


# ============================================================
# PRODUCTION FORECAST
# ============================================================

elif page == "Production Forecast":

    st.header(
        f"📈 Production Forecast — {selected_well}"
    )

    result = dca_forecast(
        well_df,
        forecast_days=90
    )

    if result is None:

        st.error(
            "Not enough valid production data."
        )

        st.stop()

    forecast = result["forecast"]

    historical_std = (
        well_df["oil_rate"]
        .std()
    )

    forecast = calculate_prediction_range(
        forecast,
        historical_std
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=well_df["date"],
            y=well_df["oil_rate"],
            mode="lines",
            name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p50"],
            mode="lines",
            name="P50 Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p90"],
            mode="lines",
            name="P90",
            line=dict(
                dash="dot"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p10"],
            mode="lines",
            name="P10",
            line=dict(
                dash="dot"
            )
        )
    )

    fig.update_layout(
        title="90-Day Production Forecast",
        yaxis_title="BOPD",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        result["model"]
    )

    c2.metric(
        "Decline Rate",
        f"{result['decline_rate']:.5f}/day"
    )

    c3.metric(
        "R²",
        f"{result['r2']:.3f}"
    )

    st.subheader("Forecast Data")

    st.dataframe(
        forecast,
        use_container_width=True
    )


# ============================================================
# DCA
# ============================================================

elif page == "DCA":

    st.header(
        f"📉 Decline Curve Analysis — {selected_well}"
    )

    result = dca_forecast(
        well_df,
        forecast_days=180
    )

    if result is None:

        st.error(
            "Insufficient data for DCA."
        )

        st.stop()

    st.write(
        "NEXUS currently uses an exponential "
        "decline baseline."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Initial Rate",
        f"{result['qi']:,.0f} BOPD"
    )

    c2.metric(
        "Decline",
        f"{result['decline_rate']:.5f}/day"
    )

    c3.metric(
        "R²",
        f"{result['r2']:.3f}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=well_df["date"],
            y=well_df["oil_rate"],
            mode="markers",
            name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result["forecast"]["date"],
            y=result["forecast"]["oil_forecast"],
            mode="lines",
            name="DCA"
        )
    )

    fig.update_layout(
        title="Decline Curve",
        xaxis_title="Date",
        yaxis_title="Oil Rate (BOPD)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SCENARIO ENGINE
# ============================================================

elif page == "Scenario Engine":

    st.header(
        f"🎯 Scenario Engine — {selected_well}"
    )

    current_rate = float(
        well_df["oil_rate"].iloc[-1]
    )

    st.write(
        f"Current production: "
        f"**{current_rate:,.0f} BOPD**"
    )

    expected_gain = st.slider(
        "Expected production improvement (%)",
        min_value=-50,
        max_value=100,
        value=10
    )

    oil_price = st.number_input(
        "Oil price ($/bbl)",
        min_value=1.0,
        value=75.0,
        step=1.0
    )

    result = intervention_scenario(
        current_rate,
        expected_gain
    )

    economics = economic_summary(
        current_rate,
        result["expected_rate"],
        oil_price
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current Rate",
        f"{current_rate:,.0f} BOPD"
    )

    c2.metric(
        "Scenario Rate",
        f"{result['expected_rate']:,.0f} BOPD"
    )

    c3.metric(
        "Incremental",
        f"{result['gain']:+,.0f} BOPD"
    )

    st.divider()

    st.subheader(
        "Economic Impact"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Daily Impact",
        f"${economics['daily_impact']:,.0f}"
    )

    c2.metric(
        "Annualized Impact",
        f"${economics['annualized_impact']:,.0f}"
    )

    st.warning(
        "Scenario values are estimates based on "
        "user-defined assumptions. They are not "
        "guaranteed production or financial outcomes."
    )


# ============================================================
# ENGINEERING DATA
# ============================================================

elif page == "Engineering Data":

    st.header("Engineering Data Explorer")

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Processed Dataset",
        data=csv,
        file_name="nexus_processed_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NEXUS — Autonomous Energy Engineering "
    "Command Center | Decision-support prototype"
)
