# This module calculates projected investment growth over 1-5 years.
# based on the user's risk appetite and monthly investment amount.
# It shows three scenarios conservative, moderate, aggressive so users
# can see a realistic range of outcomes, not just one number.


SCENARIO_RETURNS = {
    "conservative": 0.06,   # ~6% annual - similar to debt funds, RDs
    "moderate": 0.10,       # ~10% annual - hybrid/balanced funds
    "aggressive": 0.14,     # ~14% annual - equity mutual funds (long-term average)
}



# This gives a realistic starting point if the user hasn't specified an amount

DEFAULT_MONTHLY_AMOUNT = {
    "Conservative": 500,
    "Moderate": 1000,
    "Aggressive": 2000,
}


def calculate_sip_growth(monthly_amount, annual_return_rate, years):
    """
    Calculates the future value of a monthly SIP (Systematic Investment Plan)
    using the standard SIP future value formula:

    FV = P × [(1 + r)^n - 1] / r × (1 + r)

    where:
    P = monthly investment amount
    r = monthly rate of return (annual rate / 12)
    n = total number of months invested
    """
    monthly_rate = annual_return_rate / 12
    months = years * 12

    if monthly_rate == 0:
        future_value = monthly_amount * months
    else:
        future_value = monthly_amount * (
            ((1 + monthly_rate) ** months - 1) / monthly_rate
        ) * (1 + monthly_rate)

    total_invested = monthly_amount * months
    total_growth = future_value - total_invested

    return {
        "total_invested": round(total_invested, 2),
        "projected_value": round(future_value, 2),
        "growth_amount": round(total_growth, 2),
    }


def generate_growth_projection(risk_appetite, monthly_amount=None):
    """
    Generates a full 1-5 year projection across all three scenarios
    (conservative, moderate, aggressive), regardless of the user's own
    risk appetite - this lets them compare outcomes side by side.

    If no monthly_amount is given, uses a sensible default based on
    their risk appetite category.
    """
    if monthly_amount is None:
        monthly_amount = DEFAULT_MONTHLY_AMOUNT.get(risk_appetite, 1000)

    projection = {
        "risk_appetite": risk_appetite,
        "monthly_investment": monthly_amount,
        "projections_by_year": {}
    }

    for year in [1, 2, 3, 4, 5]:
        year_data = {}
        for scenario_name, rate in SCENARIO_RETURNS.items():
            result = calculate_sip_growth(monthly_amount, rate, year)
            year_data[scenario_name] = result

        projection["projections_by_year"][f"year_{year}"] = year_data

    return projection


def print_projection_summary(projection):
    """Pretty-prints the projection in a readable way for terminal testing."""
    print(f"\n=== Growth Projection ===")
    print(f"Risk Appetite: {projection['risk_appetite']}")
    print(f"Monthly Investment: ₹{projection['monthly_investment']}\n")

    for year_key, scenarios in projection["projections_by_year"].items():
        year_num = year_key.split("_")[1]
        print(f"--- After {year_num} year(s) ---")
        for scenario_name, values in scenarios.items():
            print(
                f"  {scenario_name.capitalize():13} | "
                f"Invested: ₹{values['total_invested']:,.0f} | "
                f"Projected: ₹{values['projected_value']:,.0f} | "
                f"Growth: ₹{values['growth_amount']:,.0f}"
            )
        print()


# --- Quick test when running this file directly ---
if __name__ == "__main__":
    # Testing with a Moderate risk appetite, ₹1000/month
    test_projection = generate_growth_projection("Moderate", monthly_amount=1000)
    print_projection_summary(test_projection)

    # Testing with default amount for Aggressive
    test_projection_2 = generate_growth_projection("Aggressive")
    print_projection_summary(test_projection_2)

    