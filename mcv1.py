import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def monte_carlo_simulation(start_price, time_period, step_simulation, vol, drift, num_paths=100):
    """
    Monte Carlo Simulation of Equity Prices.

    Parameters:
    - start_price: Initial price of the equity
    - time_period: Time period for the simulation (in years)
    - step_simulation: Frequency of steps ('daily' or 'monthly')
    - vol: Volatility of the equity (e.g., 0.2 for 20%)
    - drift: Drift rate (risk-free rate, e.g., 0.05 for 5%)
    - num_paths: Number of simulation paths (default: 100)

    Returns:
    - A DataFrame containing simulated price paths.
    """
    if step_simulation == "daily":
        dt = 1 / 252  # Daily steps
        steps = int(time_period * 252)
    elif step_simulation == "monthly":
        dt = 1 / 12  # Monthly steps
        steps = int(time_period * 12)
    else:
        raise ValueError("step_simulation must be 'daily' or 'monthly'")

    price_matrix = np.zeros((steps + 1, num_paths))
    price_matrix[0] = start_price

    for t in range(1, steps + 1):
        z = np.random.normal(size=num_paths)
        price_matrix[t] = price_matrix[t - 1] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z)

    return pd.DataFrame(price_matrix)

def calculate_percentiles(data, steps):
    """
    Calculates the 2.5th, 25th, 50th, 75th, and 97.5th percentiles for the given steps.

    Parameters:
    - data: DataFrame containing simulation paths.
    - steps: List of steps to calculate percentiles for.

    Returns:
    - A DataFrame with percentiles at specified steps.
    """
    percentiles = {step: data.iloc[step].quantile([0.025, 0.25, 0.50, 0.75, 0.975]) for step in steps}
    return pd.DataFrame(percentiles).T

def plot_comparison_percentiles(percentiles_A, percentiles_B, asset_A_2_5_percent, asset_B_2_5_percent):
    """
    Plots the percentiles for two assets over time and adds horizontal lines for 2.5% worst-case values.

    Parameters:
    - percentiles_A: DataFrame containing percentiles for asset A.
    - percentiles_B: DataFrame containing percentiles for asset B.
    - asset_A_2_5_percent: The 2.5% worst-case value for Asset A.
    - asset_B_2_5_percent: The 2.5% worst-case value for Asset B.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Asset A Percentiles
    ax.plot(percentiles_A.index, percentiles_A[0.025], label="Asset A - 2.5% Worst Case", linestyle="-", color="blue")

    # Plot Asset B Percentiles
    ax.plot(percentiles_B.index, percentiles_B[0.025], label="Asset B - 2.5% Worst Case", linestyle="-", color="red")

    # Add horizontal lines for 2.5% worst-case values
    ax.axhline(y=asset_A_2_5_percent, color="blue", linestyle="--", alpha=0.7)
    ax.axhline(y=asset_B_2_5_percent, color="red", linestyle="--", alpha=0.7)

    # Annotate horizontal lines with larger, bold font
    ax.text(0, asset_A_2_5_percent, f"{asset_A_2_5_percent:.2f}", color="blue", fontsize=12, fontweight="bold", verticalalignment='bottom')
    ax.text(0, asset_B_2_5_percent, f"{asset_B_2_5_percent:.2f}", color="red", fontsize=12, fontweight="bold", verticalalignment='bottom')

    ax.set_title("Comparison of Monte Carlo Simulation Worst Case for Asset A and Asset B")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)  # Ensure the Y-axis always includes zero
    st.pyplot(fig)

# Streamlit App
st.title("Monte Carlo Simulation: Asset A vs Asset B")

# User inputs for volatility
vol_A = st.slider("Volatility for Asset A", 0.1, 1.0, 0.2, 0.01)
vol_B = st.slider("Volatility for Asset B", 0.1, 1.0, 0.5, 0.01)

# Fixed Parameters for Simulation
start_price = 10000
time_period = 1
step_simulation = "daily"
drift = 0.05
num_paths = 10000

# Simulate Asset A
simulation_data_A = monte_carlo_simulation(
    start_price=start_price,
    time_period=time_period,
    step_simulation=step_simulation,
    vol=vol_A,
    drift=drift,
    num_paths=num_paths
)

# Simulate Asset B
simulation_data_B = monte_carlo_simulation(
    start_price=start_price,
    time_period=time_period,
    step_simulation=step_simulation,
    vol=vol_B,
    drift=drift,
    num_paths=num_paths
)

# Calculate percentiles for both assets
steps_to_check = list(range(0, simulation_data_A.shape[0], 10))
percentiles_A = calculate_percentiles(simulation_data_A, steps_to_check)
percentiles_B = calculate_percentiles(simulation_data_B, steps_to_check)

# Get the 2.5% percentile values at the final step
final_step = steps_to_check[-1]
asset_A_2_5_percent = percentiles_A.loc[final_step, 0.025]
asset_B_2_5_percent = percentiles_B.loc[final_step, 0.025]

# Plot comparison of percentiles with horizontal lines
plot_comparison_percentiles(percentiles_A, percentiles_B, asset_A_2_5_percent, asset_B_2_5_percent)