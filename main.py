import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
#####pual 
DATA_DIR = Path(__file__).parent
GLOBAL_FILE = DATA_DIR / "global_factors.xlsx"
SPX_FILE = DATA_DIR / "spx_factors.xlsx"
WITHDRAWALS_FILE = DATA_DIR / "withdrawals.csv"
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
# Allocation label -> (global column, spx column)
ALLOCATION_CHOICES = [
    ("100% equity", "LBM 100E", "spx100e"),
    ("90% equity / 10% fixed income", "LBM 90E", "spx90e"),
    ("80% equity / 20% fixed income", "LBM 80E", "spx80e"),
    ("70% equity / 30% fixed income", "LBM 70E", "spx70e"),
    ("60% equity / 40% fixed income", "LBM 60E", "spx60e"),
    ("50% equity / 50% fixed income", "LBM 50E", "spx50e"),
    ("40% equity / 60% fixed income", "LBM 40E", "spx40e"),
    ("30% equity / 70% fixed income", "LBM 30E", "spx30e"),
    ("20% equity / 80% fixed income", "LBM 20E", "spx20e"),
    ("10% equity / 90% fixed income", "LBM 10E", "spx10e"),
    ("100% fixed income", "LBM 100F", "spx0e"),
]


@st.cache_data(show_spinner=False)
def load_factors(path: Path) -> pd.DataFrame:
    """Read the factor workbook once and reuse the cached dataframe."""
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_withdrawal_rates(path: Path) -> pd.DataFrame:
    """Return withdrawal rates indexed by years."""
    df = pd.read_csv(path)
    return df.set_index("Years")


def rolling_balances(
    factors: pd.Series, years: int, annual_contribution: float
) -> tuple[list[float], list[object]]:
    """Compute balances for every rolling window of `years` length."""
    numeric = pd.to_numeric(factors, errors="coerce").dropna()
    values = numeric.to_numpy(dtype=float)
    if years < 1:
        raise ValueError("Years must be at least 1.")
    required_rows = (years - 1) * MONTHS_PER_YEAR + 1
    if len(values) < required_rows:
        raise ValueError("Not enough factor history for the selected horizon.")

    balances: list[float] = []
    starts: list[object] = []
    stride = MONTHS_PER_YEAR
    for start in range(0, len(values) - (years - 1) * stride):
        balance = 0.0
        for year_idx in range(years):
            factor = values[start + year_idx * stride]
            balance = (balance + annual_contribution) * factor  # contribute then compound
        balances.append(balance)
        starts.append(numeric.index[start])
    return balances, starts


def summarize_outcomes(series: pd.Series, years: int, annual_contribution: float) -> dict[str, object]:
    """Return the full ending balance distribution plus min/median stats."""
    balances, starts = rolling_balances(series, years, annual_contribution)
    if not balances:
        empty_series = pd.Series(dtype=float, name="ending_balance")
        empty_table = pd.DataFrame(columns=["window_start_row", "ending_balance"])
        return {"values": empty_series, "table": empty_table, "min": 0.0, "median": 0.0}

    ending_series = pd.Series(balances, index=starts, name="ending_balance")
    sorted_table = (
        ending_series.sort_values()
        .reset_index()
        .rename(columns={"index": "window_start_row"})
    )
    return {
        "values": ending_series,
        "table": sorted_table,
        "min": float(ending_series.min()),
        "median": float(ending_series.median()),
    }


st.set_page_config(page_title="Habit Savings Investment Calculator", layout="wide")
st.title("Habit Savings Investment Calculator")

st.write(
    "Compare the impact of swapping a daily habit for a lower-cost option and investing the savings."
)

with st.sidebar:
    st.header("Inputs")
    costly_spend = st.slider("Daily habit cost ($)", 0.0, 50.0, 8.0, 0.25)
    frugal_spend = st.slider("Frugal alternative cost ($)", 0.0, costly_spend, 0.50, 0.25)
    days_per_week = st.slider("Days per week for the habit", 1, 7, 5)
    years = st.slider("Years saving and investing", 1, 50, 20)
    retirement_years = st.slider("Years in retirement", 1, 60, 30)
    allocation_label = st.selectbox(
        "Portfolio allocation",
        options=[label for label, _, _ in ALLOCATION_CHOICES],
        index=5,
    )

selected_global_col = next(
    global_col for label, global_col, _ in ALLOCATION_CHOICES if label == allocation_label
)
selected_spx_col = next(
    spx_col for label, _, spx_col in ALLOCATION_CHOICES if label == allocation_label
)

daily_savings = max(costly_spend - frugal_spend, 0.0)
annual_contribution = daily_savings * days_per_week * WEEKS_PER_YEAR
total_contributions = annual_contribution * years

if daily_savings <= 0:
    st.warning("The frugal option needs to cost less than the habit for savings to accumulate.")
else:
    st.subheader("Annual Savings")
    st.write(f"Daily savings: **${daily_savings:,.2f}**")
    st.write(f"Annual contribution: **${annual_contribution:,.2f}**")
    st.write(f"Total contributed over {years} years: **${total_contributions:,.2f}**")

    try:
        global_df = load_factors(GLOBAL_FILE)
        spx_df = load_factors(SPX_FILE)
        withdrawals = load_withdrawal_rates(WITHDRAWALS_FILE)

        global_outcomes = summarize_outcomes(global_df[selected_global_col], years, annual_contribution)
        spx_outcomes = summarize_outcomes(spx_df[selected_spx_col], years, annual_contribution)

        st.subheader("Investment Outcomes")
        cols = st.columns(2)

        with cols[0]:
            st.markdown(f"**Global Factors ({selected_global_col})**")
            st.metric("Median ending balance", f"${global_outcomes['median']:,.0f}")

        with cols[1]:
            st.markdown(f"**S&P 500 Factors ({selected_spx_col})**")
            st.metric("Median ending balance", f"${spx_outcomes['median']:,.0f}")

        st.caption(
            "Returns use rolling windows of the selected allocation's annual factors."
        )

        if years >= 20 and not withdrawals.empty:
            withdrawal_row = withdrawals.reindex([years]).ffill().bfill().iloc[0]
            median_rate = float(withdrawal_row.get("Median", 0.0))

            st.subheader("Sustainable Withdrawals (60% Stock Allocation)")
            st.write(
                f"Using a median withdrawal rate of **{median_rate:.1%}** from the 60% stock allocation study."
            )

            withdraw_cols = st.columns(2)

            median_withdrawal_global = global_outcomes["median"] * median_rate
            total_income_global = median_withdrawal_global * retirement_years
            median_withdrawal_spx = spx_outcomes["median"] * median_rate
            total_income_spx = median_withdrawal_spx * retirement_years

            with withdraw_cols[0]:
                st.markdown("**Global Portfolio**")
                st.metric(
                    "Median annual withdrawal",
                    f"${median_withdrawal_global:,.0f}",
                    help="Median ending balance × median withdrawal rate",
                )
                st.metric(
                    f"Total over {retirement_years} years",
                    f"${total_income_global:,.0f}",
                    help="Median annual withdrawal × years in retirement",
                )

            with withdraw_cols[1]:
                st.markdown("**S&P 500 Portfolio**")
                st.metric(
                    "Median annual withdrawal",
                    f"${median_withdrawal_spx:,.0f}",
                    help="Median ending balance × median withdrawal rate",
                )
                st.metric(
                    f"Total over {retirement_years} years",
                    f"${total_income_spx:,.0f}",
                    help="Median annual withdrawal × years in retirement",
                )
    except FileNotFoundError as err:
        st.error(f"Missing factor file: {err}")
    except ValueError as err:
        st.error(str(err))
