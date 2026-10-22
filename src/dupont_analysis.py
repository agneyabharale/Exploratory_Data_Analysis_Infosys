import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dupont_3stage(df_income: pd.DataFrame, df_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 3-Stage DuPont Decomposition:
    ROE = Net Profit Margin * Asset Turnover * Equity Multiplier (Financial Leverage)
    """
    # Standardize column names
    inc = df_income[['year', 'revenue', 'net_profit']].copy()
    bal = df_balance[['year', 'total_assets', 'total_equity']].copy()
    
    # Ensure year is integer
    inc['year'] = inc['year'].astype(int)
    bal['year'] = bal['year'].astype(int)
    
    merged = pd.merge(inc, bal, on='year').sort_values('year').reset_index(drop=True)
    
    # DuPont Component Formulations
    merged['net_profit_margin'] = merged['net_profit'] / merged['revenue']
    merged['asset_turnover'] = merged['revenue'] / merged['total_assets']
    merged['equity_multiplier'] = merged['total_assets'] / merged['total_equity']
    
    # Reconstructed ROE via DuPont
    merged['dupont_roe'] = merged['net_profit_margin'] * merged['asset_turnover'] * merged['equity_multiplier']
    
    # Format percentages for clean display
    merged['net_margin_pct'] = merged['net_profit_margin'] * 100
    merged['dupont_roe_pct'] = merged['dupont_roe'] * 100
    
    return merged

def plot_dupont_analysis(dupont_df: pd.DataFrame, save_path: str = None):
    """
    Generate an executive 2-panel DuPont visualization showing the 3 core engines
    of Return on Equity (ROE).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    years = dupont_df['year'].astype(str)
    
    # Panel 1: The 3 Core Drivers
    ax1.plot(years, dupont_df['net_margin_pct'], marker='o', linewidth=2.5, color='#2563eb', label='1. Net Profit Margin (%)')
    ax1.plot(years, dupont_df['asset_turnover'] * 10, marker='s', linewidth=2.5, color='#16a34a', linestyle='--', label='2. Asset Turnover (x10 scale)')
    ax1.plot(years, dupont_df['equity_multiplier'] * 10, marker='^', linewidth=2.5, color='#d97706', linestyle='-.', label='3. Equity Multiplier / Leverage (x10 scale)')
    
    ax1.set_title("DuPont Analysis: The 3 Underlying Engines of Profitability", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Driver Trajectory", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)
    
    # Annotate specific key milestones on panel 1
    for i, txt in enumerate(dupont_df['net_margin_pct']):
        ax1.annotate(f"{txt:.1f}%", (years[i], dupont_df['net_margin_pct'].iloc[i] + 0.8), fontsize=9, ha='center')
    for i, em in enumerate(dupont_df['equity_multiplier']):
        ax1.annotate(f"{em:.2f}x", (years[i], (em * 10) + 0.8), fontsize=9, ha='center', color='#d97706')

    # Panel 2: The Resulting ROE Trajectory
    bars = ax2.bar(years, dupont_df['dupont_roe_pct'], color='#4f46e5', width=0.45, alpha=0.85, label='Resulting ROE (%)')
    ax2.plot(years, dupont_df['dupont_roe_pct'], color='#ef4444', marker='o', linewidth=2)
    
    ax2.set_title("Resulting Return on Equity (ROE) Surge: 40% → 79%", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Financial Year", fontsize=11)
    ax2.set_ylabel("ROE (%)", fontsize=11)
    ax2.set_ylim(0, 95)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1e1b4b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def explain_dupont_drivers(dupont_df: pd.DataFrame) -> str:
    """
    Generate an executive narrative explaining why ROE surged and what it signifies for business strategy.
    """
    start = dupont_df.iloc[0]
    end = dupont_df.iloc[-1]
    
    margin_diff = end['net_margin_pct'] - start['net_margin_pct']
    turnover_diff = end['asset_turnover'] - start['asset_turnover']
    leverage_diff = end['equity_multiplier'] - start['equity_multiplier']
    roe_diff = end['dupont_roe_pct'] - start['dupont_roe_pct']
    
    narrative = [
        "==================================================================",
        "          DUPONT 3-STAGE ROE DECOMPOSITION INSIGHTS               ",
        "==================================================================",
        f"Period Analyzed: FY{int(start['year'])} to FY{int(end['year'])}",
        f"Overall ROE Expansion: {start['dupont_roe_pct']:.1f}%  --->  {end['dupont_roe_pct']:.1f}% (+{roe_diff:.1f}% pts)\n",
        "DECONSTRUCTING THE 3 ENGINES:",
        f"1. Profit Margin Engine:  {start['net_margin_pct']:.1f}% -> {end['net_margin_pct']:.1f}% (Shift: {margin_diff:+.1f}% pts)",
        "   Significance: Margins remained steady around 17-18%. Pricing power was stable,",
        "   meaning higher ROE was NOT caused by price gouging or extreme cost cuts.\n",
        f"2. Asset Turnover Engine: {start['asset_turnover']:.2f}x -> {end['asset_turnover']:.2f}x (Shift: {turnover_diff:+.2f}x)",
        "   Significance: Asset efficiency improved. Every ₹1 of assets generates ~₹1.65 of revenue,",
        "   reflecting Infosys's asset-light IT services operational model.\n",
        f"3. Leverage Engine (Equity Multiplier): {start['equity_multiplier']:.2f}x -> {end['equity_multiplier']:.2f}x (Shift: {leverage_diff:+.2f}x)",
        "   Significance: THIS WAS THE PRIMARY ENGINE (+61% increase in leverage ratio)!",
        "   Why? Infosys returned massive excess cash to shareholders via dividends and buybacks,",
        "   deliberately slimming its equity base while taking on zero risky debt. This magnified",
        "   the return on every remaining rupee of shareholders' equity.",
        "=================================================================="
    ]
    return "\n".join(narrative)
