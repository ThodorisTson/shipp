"""
SHIPP Example 2 Integration with PyWake
========================================

This script demonstrates how to replace renewable.ninja wind data
with PyWake wind farm modeling for SHIPP optimization.

Key improvements:
- State-of-the-art wake modeling
- Realistic power production estimates
- Flexible wind farm configuration
- Better control over wind farm parameters

Author: Thodoris
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pywake_windfarm import WindFarmModel, create_reference_offshore_windfarm, generate_synthetic_wind_data


def prepare_wind_data_for_shipp(power_ts, price_data=None):
    """
    Prepare PyWake power output for SHIPP optimization.
    
    This function takes the wind farm power time series from PyWake
    and formats it for use in SHIPP, similar to how renewable.ninja
    data was processed in example 2.
    
    Parameters
    ----------
    power_ts : pandas.Series
        Wind farm power output in MW (from PyWake)
    price_data : pandas.Series, optional
        Electricity price time series (€/MWh)
        If None, synthetic prices are generated
        
    Returns
    -------
    shipp_input : dict
        Dictionary with data ready for SHIPP:
        - 'power_mw': Wind power in MW
        - 'price_eur_mwh': Electricity price
        - 'timestamps': Time index
    """
    # Ensure we have a datetime index
    if not isinstance(power_ts.index, pd.DatetimeIndex):
        power_ts.index = pd.date_range('2024-01-01', periods=len(power_ts), freq='h')
    
    # Generate synthetic price data if not provided
    if price_data is None:
        price_data = generate_synthetic_prices(len(power_ts), power_ts.index)
    
    # Align price data with power data
    if len(price_data) != len(power_ts):
        # Resample or align as needed
        price_data = price_data.reindex(power_ts.index, method='nearest')
    
    shipp_input = {
        'power_mw': power_ts.values,
        'price_eur_mwh': price_data.values,
        'timestamps': power_ts.index,
        'total_capacity_mw': power_ts.max()  # Useful for normalization
    }
    
    return shipp_input


def generate_synthetic_prices(n_hours, timestamps=None):
    """
    Generate synthetic electricity prices with realistic patterns.
    
    Includes:
    - Daily cycles (higher during day, lower at night)
    - Weekly cycles (lower on weekends)
    - Seasonal trends
    - Stochastic variations
    
    Parameters
    ----------
    n_hours : int
        Number of hours
    timestamps : DatetimeIndex, optional
        Timestamps for the prices
        
    Returns
    -------
    prices : pandas.Series
        Electricity prices in €/MWh
    """
    if timestamps is None:
        timestamps = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    
    # Base price
    base_price = 50.0  # €/MWh
    
    # Daily cycle (amplitude: 20 €/MWh)
    hour_of_day = timestamps.hour
    daily_cycle = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Weekly cycle (lower on weekends)
    day_of_week = timestamps.dayofweek
    weekend_reduction = np.where(day_of_week >= 5, -15, 0)
    
    # Seasonal trend (higher in winter)
    day_of_year = timestamps.dayofyear
    seasonal = 15 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    
    # Stochastic component (AR(1) process)
    rng = np.random.RandomState(42)
    stoch = np.zeros(n_hours)
    stoch[0] = rng.normal(0, 5)
    
    for i in range(1, n_hours):
        stoch[i] = 0.9 * stoch[i-1] + rng.normal(0, 5)
    
    # Combine all components
    prices = base_price + daily_cycle + weekend_reduction + seasonal + stoch
    
    # Ensure non-negative prices
    prices = np.maximum(prices, 10.0)
    
    return pd.Series(prices, index=timestamps, name='price_eur_mwh')


def calculate_revenue_potential(power_mw, price_eur_mwh):
    """
    Calculate revenue potential from wind generation without storage.
    
    Parameters
    ----------
    power_mw : array
        Wind power generation in MW
    price_eur_mwh : array
        Electricity prices in €/MWh
        
    Returns
    -------
    stats : dict
        Revenue statistics
    """
    # Revenue per hour (assuming 1-hour timesteps)
    hourly_revenue = power_mw * price_eur_mwh
    
    stats = {
        'total_revenue_eur': hourly_revenue.sum(),
        'mean_hourly_revenue_eur': hourly_revenue.mean(),
        'revenue_per_mwh': hourly_revenue.sum() / power_mw.sum() if power_mw.sum() > 0 else 0,
        'total_energy_mwh': power_mw.sum(),
        'mean_power_mw': power_mw.mean(),
        'mean_price_eur_mwh': price_eur_mwh.mean()
    }
    
    return stats


def identify_arbitrage_opportunities(power_mw, price_eur_mwh, window_hours=24):
    """
    Identify potential arbitrage opportunities for battery storage.
    
    This shows periods where storage could benefit from price differences.
    
    Parameters
    ----------
    power_mw : array
        Wind power generation
    price_eur_mwh : array
        Electricity prices
    window_hours : int
        Rolling window for analysis
        
    Returns
    -------
    opportunities : pandas.DataFrame
        DataFrame with arbitrage metrics
    """
    df = pd.DataFrame({
        'power': power_mw,
        'price': price_eur_mwh
    })
    
    # Calculate rolling statistics
    df['price_min'] = df['price'].rolling(window_hours, center=True).min()
    df['price_max'] = df['price'].rolling(window_hours, center=True).max()
    df['price_spread'] = df['price_max'] - df['price_min']
    
    # Price percentile within window (0 = cheapest, 1 = most expensive)
    df['price_percentile'] = df['price'].rolling(window_hours, center=True).apply(
        lambda x: (x.iloc[len(x)//2] - x.min()) / (x.max() - x.min() + 1e-6)
    )
    
    # High value period: high price AND high spread (good for discharge)
    df['high_value_period'] = (df['price'] > df['price'].quantile(0.75)) & \
                               (df['price_spread'] > df['price_spread'].median())
    
    # Low value period: low price AND high spread (good for charge)
    df['low_value_period'] = (df['price'] < df['price'].quantile(0.25)) & \
                              (df['price_spread'] > df['price_spread'].median())
    
    return df


def plot_wind_and_prices(shipp_input, hours_to_plot=168):
    """
    Visualize wind power and electricity prices.
    
    Parameters
    ----------
    shipp_input : dict
        SHIPP input data dictionary
    hours_to_plot : int
        Number of hours to plot (default: 1 week)
    """
    n = min(hours_to_plot, len(shipp_input['power_mw']))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    times = shipp_input['timestamps'][:n]
    power = shipp_input['power_mw'][:n]
    prices = shipp_input['price_eur_mwh'][:n]
    
    # Plot 1: Wind power
    axes[0].plot(times, power, linewidth=1.5, color='steelblue')
    axes[0].fill_between(times, 0, power, alpha=0.3, color='steelblue')
    axes[0].set_ylabel('Power [MW]', fontsize=11)
    axes[0].set_title('Wind Farm Power Production', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Electricity prices
    axes[1].plot(times, prices, linewidth=1.5, color='orangered')
    axes[1].axhline(prices.mean(), color='black', linestyle='--', 
                    label=f'Mean: {prices.mean():.1f} €/MWh', alpha=0.5)
    axes[1].set_ylabel('Price [€/MWh]', fontsize=11)
    axes[1].set_title('Electricity Price', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Revenue potential (power × price)
    revenue = power * prices
    axes[2].bar(times, revenue, width=1/24, color='green', alpha=0.6)
    axes[2].set_ylabel('Revenue [€/h]', fontsize=11)
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_title('Hourly Revenue Potential', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_arbitrage_analysis(power_mw, price_eur_mwh, timestamps, hours_to_plot=168):
    """
    Plot arbitrage opportunity analysis.
    
    Parameters
    ----------
    power_mw : array
        Wind power
    price_eur_mwh : array
        Electricity prices
    timestamps : DatetimeIndex
        Time index
    hours_to_plot : int
        Hours to display
    """
    # Calculate opportunities
    df = identify_arbitrage_opportunities(power_mw, price_eur_mwh)
    
    n = min(hours_to_plot, len(df))
    times = timestamps[:n]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Price with spread
    axes[0].plot(times, df['price'][:n], label='Price', color='blue', linewidth=1.5)
    axes[0].fill_between(times, df['price_min'][:n], df['price_max'][:n], 
                         alpha=0.2, color='gray', label='24h Range')
    axes[0].set_ylabel('Price [€/MWh]', fontsize=11)
    axes[0].set_title('Price Dynamics and Volatility', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Opportunities
    axes[1].plot(times, df['price'][:n], color='gray', linewidth=1, alpha=0.5, label='Price')
    
    # Highlight periods
    high_val = df['high_value_period'][:n]
    low_val = df['low_value_period'][:n]
    
    axes[1].scatter(times[high_val], df['price'][:n][high_val], 
                    color='red', marker='o', s=30, label='Discharge opportunity', alpha=0.7)
    axes[1].scatter(times[low_val], df['price'][:n][low_val], 
                    color='green', marker='o', s=30, label='Charge opportunity', alpha=0.7)
    
    axes[1].set_ylabel('Price [€/MWh]', fontsize=11)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_title('Energy Arbitrage Opportunities', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    """
    Complete workflow: PyWake → SHIPP integration
    """
    print("=" * 70)
    print("PyWake → SHIPP Integration Example")
    print("=" * 70)
    
    # Step 1: Create wind farm
    print("\n[1] Creating offshore wind farm model...")
    wf = create_reference_offshore_windfarm(n_turbines=10, turbine_mw=5.0)
    print(f"    Total capacity: {wf.n_turbines * wf.turbine_rating_mw:.0f} MW")
    
    # Step 2: Generate wind data
    print("\n[2] Generating wind data (1 week for demonstration)...")
    wind_data = generate_synthetic_wind_data(hours=168, mean_ws=9.0)
    print(f"    Mean wind speed: {wind_data['wind_speed'].mean():.2f} m/s")
    
    # Step 3: Calculate power production
    print("\n[3] Calculating wind farm power output with wake effects...")
    power_ts = wf.generate_timeseries(
        wind_data['wind_speed'],
        wind_data['wind_direction'],
        ti=0.06
    )
    cf = power_ts.mean() / (wf.n_turbines * wf.turbine_rating_mw) * 100
    print(f"    Mean power: {power_ts.mean():.2f} MW")
    print(f"    Capacity factor: {cf:.1f}%")
    
    # Step 4: Prepare for SHIPP
    print("\n[4] Preparing data for SHIPP optimization...")
    shipp_input = prepare_wind_data_for_shipp(power_ts)
    
    # Calculate baseline revenue (no storage)
    stats = calculate_revenue_potential(
        shipp_input['power_mw'],
        shipp_input['price_eur_mwh']
    )
    
    print(f"    Total energy: {stats['total_energy_mwh']:.0f} MWh")
    print(f"    Revenue (no storage): {stats['total_revenue_eur']:.0f} €")
    print(f"    Mean price: {stats['mean_price_eur_mwh']:.2f} €/MWh")
    
    # Step 5: Visualize
    print("\n[5] Creating visualizations...")
    
    # Wind and prices
    fig1 = plot_wind_and_prices(shipp_input, hours_to_plot=168)
    
    # Arbitrage analysis
    fig2 = plot_arbitrage_analysis(
        shipp_input['power_mw'],
        shipp_input['price_eur_mwh'],
        shipp_input['timestamps'],
        hours_to_plot=168
    )
    
    plt.show()
    
    print("\n[6] Next steps for SHIPP integration:")
    print("    - Use shipp_input['power_mw'] as renewable generation in SHIPP")
    print("    - Use shipp_input['price_eur_mwh'] for arbitrage optimization")
    print("    - Configure battery parameters (E_capacity, P_capacity, efficiency)")
    print("    - Run SHIPP optimization to find optimal storage operation")
    
    print("\n" + "=" * 70)
    print("Integration example complete!")
    print("\nData ready for SHIPP:")
    print(f"  - Power time series: {len(shipp_input['power_mw'])} hours")
    print(f"  - Price time series: {len(shipp_input['price_eur_mwh'])} hours")
    print(f"  - Wind farm capacity: {shipp_input['total_capacity_mw']:.0f} MW")
    print("=" * 70)