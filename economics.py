def production_loss(
    rate_difference,
    oil_price,
    days
):

    return (
        rate_difference *
        oil_price *
        days
    )


def annualized_impact(
    daily_loss
):

    return daily_loss * 365


def scenario_revenue(
    production_rate,
    oil_price,
    days
):

    return (
        production_rate *
        oil_price *
        days
    )


def economic_summary(
    current_rate,
    scenario_rate,
    oil_price
):

    incremental_rate = (
        scenario_rate -
        current_rate
    )

    daily_impact = (
        incremental_rate *
        oil_price
    )

    annual_impact = (
        daily_impact *
        365
    )

    return {
        "incremental_rate":
            incremental_rate,

        "daily_impact":
            daily_impact,

        "annualized_impact":
            annual_impact
    }
