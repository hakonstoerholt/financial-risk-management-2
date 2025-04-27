import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm
import os
import sys

# --- Configuration and Constants ---
# Style and Random Seed
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("deep")

# Model Parameters
N_SIMULATIONS = 10000
N_MONTHS = 60
LOAN_RECOVERY_RATE = 0.40
CPTY_RECOVERY_RATE = 0.60
CPTY_LGD = 1.0 - CPTY_RECOVERY_RATE
CORRELATION_FACTOR_DEFAULTS = 0.39
DERIVATIVE_THRESHOLD = 0.15
DERIVATIVE_PAYOUT_FACTOR = 0.95
DERIVATIVE_HAIRCUT = 1.0 - DERIVATIVE_PAYOUT_FACTOR
FUNDING_LOSS_THRESHOLD = 0.15
CAPITAL_COST_RATE = 0.08
RISK_FREE_RATE_5Y_ANNUAL = 0.044006

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
DATA_DIR = os.path.join(SCRIPT_DIR, 'Data')
LOAN_PORTFOLIO_FILE = os.path.join(DATA_DIR, 'Loan_Portfolio_data.csv')
CDS_FILE = os.path.join(DATA_DIR, 'CDS.csv')
INTEREST_RATE_FACTOR_SIMS_FILE = os.path.join(DATA_DIR, 'Interest_Rate_Factor_Sims.csv')
INTEREST_RATE_FORECAST_SIMS_FILE = os.path.join(DATA_DIR, 'Interest_Rate_Forecast_Sims.csv')
BID_ASK_SPREADS_FILE = os.path.join(DATA_DIR, 'Bid-Ask_Spreads.csv')

# Output Directories
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'Results_Final')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'Plots')
DATA_OUT_DIR = os.path.join(RESULTS_DIR, 'Data')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'Reports')


# --- Utility Functions ---

def create_output_directories():
    """Create directories for organizing different types of outputs"""
    directories = [
        RESULTS_DIR,
        PLOTS_DIR,
        DATA_OUT_DIR,
        REPORTS_DIR
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def get_monthly_discount_rate(annual_rate):
    """Convert annual rate to effective monthly rate"""
    return (1 + annual_rate)**(1/12) - 1

def discount_factor(monthly_rate, num_periods):
    """
    Calculate discount factor for a number of periods.
    Handles scalar or array num_periods inputs.
    """
    if np.isscalar(num_periods):
        if num_periods <= 0: return 1.0
        else: return (1 / (1 + monthly_rate)) ** num_periods
    else:
        df = np.ones_like(num_periods, dtype=float)
        positive_periods_mask = (num_periods > 0)
        if np.any(positive_periods_mask):
            df[positive_periods_mask] = (1 / (1 + monthly_rate)) ** num_periods[positive_periods_mask]
        return df

def present_value(cash_flow, monthly_rate, num_periods):
    return cash_flow * discount_factor(monthly_rate, num_periods)

def calculate_annuity_pv(payment, monthly_rate, num_periods):
    if num_periods <= 0:
        return 0.0 if np.isscalar(monthly_rate) else np.zeros_like(monthly_rate)
    if np.isscalar(monthly_rate):
        rate_safe = monthly_rate + 1e-12
        if abs(rate_safe) < 1e-12: return payment * num_periods
        else: return payment * (1 - (1 + rate_safe)**(-num_periods)) / rate_safe
    else:
        pv = np.zeros_like(monthly_rate, dtype=float)
        rate_safe = monthly_rate + 1e-12
        zero_mask = np.abs(rate_safe) < 1e-12
        non_zero_mask = ~zero_mask
        if np.any(non_zero_mask):
            pv[non_zero_mask] = payment * (1 - (1 + rate_safe[non_zero_mask])**(-num_periods)) / rate_safe[non_zero_mask]
        if np.any(zero_mask):
            pv[zero_mask] = payment * num_periods
        return pv


# --- Data Loading and Initial Processing ---

def load_data():
    """Load all relevant data files for analysis, halt on critical errors."""
    print("=== LOADING DATA ===")
    data = {}
    critical_error = False
    try:
        loan_portfolio = pd.read_csv(LOAN_PORTFOLIO_FILE, index_col=0)
        required_cols = ['Principal', 'Monthly Payment', 'Monthly Prob. Default', 'Yield']
        if not all(col in loan_portfolio.columns for col in required_cols): raise ValueError(f"Loan portfolio missing required columns: {required_cols}")
        loan_portfolio['Monthly Yield'] = (1 + loan_portfolio['Yield'])**(1/12) - 1
        data['loan_portfolio'] = loan_portfolio
        print(f"Loaded Loan Portfolio: {loan_portfolio.shape}")
    except FileNotFoundError: print(f"CRITICAL ERROR: Loan portfolio file not found at {LOAN_PORTFOLIO_FILE}"); critical_error = True
    except Exception as e: print(f"CRITICAL ERROR loading Loan Portfolio: {e}"); critical_error = True

    try:
        interest_rate_forecasts_df = pd.read_csv(INTEREST_RATE_FORECAST_SIMS_FILE, index_col=0)
        if interest_rate_forecasts_df.columns[0].lower().replace(" ", "_") == 'remaining_maturity':
            interest_rate_forecasts = interest_rate_forecasts_df.iloc[:, 1:]
        else:
            interest_rate_forecasts = interest_rate_forecasts_df
        if interest_rate_forecasts.shape != (N_MONTHS, N_SIMULATIONS): raise ValueError(f"Forecast shape mismatch. Expected ({N_MONTHS}, {N_SIMULATIONS}), got {interest_rate_forecasts.shape}")
        data['interest_rate_forecasts'] = interest_rate_forecasts.apply(pd.to_numeric)
        print(f"Loaded Interest Rate Forecasts: {interest_rate_forecasts.shape}")
    except FileNotFoundError: print(f"CRITICAL ERROR: Interest Rate Forecasts file not found at {INTEREST_RATE_FORECAST_SIMS_FILE}"); critical_error = True
    except Exception as e: print(f"CRITICAL ERROR loading Interest Rate Forecasts: {e}"); critical_error = True

    try:
        interest_rate_factors_df = pd.read_csv(INTEREST_RATE_FACTOR_SIMS_FILE, index_col=0)
        if interest_rate_factors_df.columns[0].lower().replace(" ", "_") == 'remaining_maturity':
            interest_rate_factors = interest_rate_factors_df.iloc[:, 1:]
        else:
            interest_rate_factors = interest_rate_factors_df
        if interest_rate_factors.shape != (N_MONTHS, N_SIMULATIONS): raise ValueError(f"Factor shape mismatch. Expected ({N_MONTHS}, {N_SIMULATIONS}), got {interest_rate_factors.shape}")
        data['interest_rate_factors'] = interest_rate_factors.apply(pd.to_numeric)
        print(f"Loaded Interest Rate Factors: {interest_rate_factors.shape}")
    except FileNotFoundError: print(f"CRITICAL ERROR: Interest Rate Factors file not found at {INTEREST_RATE_FACTOR_SIMS_FILE}"); critical_error = True
    except Exception as e: print(f"CRITICAL ERROR loading Interest Rate Factors: {e}"); critical_error = True

    try:
        cds_data = pd.read_csv(CDS_FILE)
        if 'CDS Spread' not in cds_data.columns: raise ValueError("CDS data missing 'CDS Spread' column.")
        cds_data['CDS Spread Decimal'] = cds_data['CDS Spread'].str.rstrip('%').astype('float') / 100.0
        data['cds_data'] = cds_data
        print(f"Loaded CDS Data: {cds_data.shape}")
    except FileNotFoundError: print(f"CRITICAL ERROR: CDS file not found at {CDS_FILE}"); critical_error = True
    except Exception as e: print(f"CRITICAL ERROR loading CDS Data: {e}"); critical_error = True

    try:
        bid_ask_spreads = pd.read_csv(BID_ASK_SPREADS_FILE)
        if 'Proportional_Spread (%)' not in bid_ask_spreads.columns: raise ValueError("Bid-Ask data missing 'Proportional_Spread (%)' column.")
        bid_ask_spreads = bid_ask_spreads.dropna(subset=['Proportional_Spread (%)']) # Drop rows where spread is NaN
        bid_ask_spreads['Proportional_Spread_Decimal'] = pd.to_numeric(bid_ask_spreads['Proportional_Spread (%)']) / 100.0
        data['bid_ask_spreads'] = bid_ask_spreads
        print(f"Loaded Bid-Ask Spreads: {bid_ask_spreads.shape}")
    except FileNotFoundError: print(f"CRITICAL ERROR: Bid-Ask Spreads file not found at {BID_ASK_SPREADS_FILE}"); critical_error = True
    except Exception as e: print(f"CRITICAL ERROR loading Bid-Ask Spreads: {e}"); critical_error = True

    if critical_error:
        print("Halting execution due to critical data loading errors.")
        sys.exit(1)

    print("Data loading and initial validation complete!")
    return data


# --- Core Calculation Functions ---

def calculate_outstanding_principal(loan_portfolio):
    """Calculate month-by-month outstanding principal for each loan."""
    print("Calculating outstanding principal trajectories...")
    loans = loan_portfolio
    num_loans = len(loans)
    outstanding_principal = np.zeros((num_loans, N_MONTHS + 1))
    outstanding_principal[:, 0] = loans['Principal'].values
    monthly_payments = loans['Monthly Payment'].values
    monthly_yields = loans['Monthly Yield'].values

    for m in range(N_MONTHS):
        interest_due = outstanding_principal[:, m] * monthly_yields
        payment_m = np.minimum(monthly_payments, outstanding_principal[:, m] * (1 + monthly_yields))
        principal_paid = payment_m - interest_due
        principal_paid = np.minimum(principal_paid, outstanding_principal[:, m])
        outstanding_principal[:, m+1] = outstanding_principal[:, m] - principal_paid
        outstanding_principal[:, m+1] = np.maximum(0, outstanding_principal[:, m+1])

    print("Outstanding principal calculation complete.")
    return outstanding_principal

def calculate_portfolio_exposure(loan_portfolio, interest_rate_forecasts):
    """Q1: Calculate portfolio exposure (Default-Free Value) based on simulated rates."""
    print("\n=== Q1: CALCULATING PORTFOLIO EXPOSURE ===")
    loans = loan_portfolio
    num_loans = len(loans)
    monthly_payments = loans['Monthly Payment'].values
    rate_sims_annual_np = interest_rate_forecasts.values
    n_sims = rate_sims_annual_np.shape[1]

    # Convert annual rate simulations to monthly
    monthly_rate_sims_np = (1 + rate_sims_annual_np)**(1/12) - 1

    portfolio_value_ts = np.zeros((N_MONTHS, n_sims))

    for t in tqdm(range(N_MONTHS), desc="Q1: Calculating portfolio exposure"):
        remaining_periods = N_MONTHS - t
        if remaining_periods <= 0:
            continue

        # Get the monthly rates for the current month across all simulations
        rates_t = monthly_rate_sims_np[t, :]

        pv_loans_t = np.zeros((num_loans, n_sims))
        for i in range(num_loans):
            pv_loans_t[i, :] = calculate_annuity_pv(monthly_payments[i], rates_t, remaining_periods)

        portfolio_value_ts[t, :] = pv_loans_t.sum(axis=0)

    # Analysis and Output
    avg_portfolio_value = portfolio_value_ts.mean(axis=1)
    print(f"Average portfolio value at t=0 (Start Month 1): ${avg_portfolio_value[0]:,.2f}")
    print(f"Average portfolio value at t={N_MONTHS-1} (Start Month {N_MONTHS}): ${avg_portfolio_value[N_MONTHS-1]:,.2f}")

    plt.figure()
    plt.plot(range(1, N_MONTHS + 1), avg_portfolio_value)
    plt.title('Average Portfolio Value Over Time (Default-Free)')
    plt.xlabel('Month')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'q1_portfolio_value_over_time.png'))
    plt.close()

    # Save detailed statistics
    value_stats_df = pd.DataFrame({
        'Month': range(1, N_MONTHS + 1),
        'Average_Value': avg_portfolio_value,
        'Min_Value': portfolio_value_ts.min(axis=1),
        'Max_Value': portfolio_value_ts.max(axis=1),
        'Std_Value': portfolio_value_ts.std(axis=1),
        'P01_Value': np.percentile(portfolio_value_ts, 1, axis=1),
        'P99_Value': np.percentile(portfolio_value_ts, 99, axis=1)
    })
    value_stats_df.to_csv(os.path.join(DATA_OUT_DIR, 'q1_portfolio_value_stats.csv'), index=False)

    # Write summary report
    with open(os.path.join(REPORTS_DIR, 'q1_portfolio_exposure_summary.txt'), 'w') as f:
        f.write("=== Q1: PORTFOLIO EXPOSURE SUMMARY ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(" - Value is PV of remaining fixed payments.\n")
        f.write(" - Discount rate is the simulated monthly rate for the period.\n\n")
        f.write(f"Number of loans: {num_loans}\n")
        f.write(f"Number of simulations: {n_sims}\n")
        f.write(f"Average portfolio value at start Month 1: ${avg_portfolio_value[0]:,.2f}\n")
        f.write(f"Average portfolio value at start Month {N_MONTHS}: ${avg_portfolio_value[N_MONTHS-1]:,.2f}\n")
        f.write(f"99th Percentile value at start Month {N_MONTHS}: ${value_stats_df['P99_Value'].iloc[-1]:,.2f}\n")
        f.write(f"1st Percentile value at start Month {N_MONTHS}: ${value_stats_df['P01_Value'].iloc[-1]:,.2f}\n")

    return portfolio_value_ts

def simulate_defaults_and_losses(loan_portfolio, outstanding_principal, interest_rate_factors, portfolio_value_ts):
    """Q2: Simulate defaults and calculate losses based on outstanding principal."""
    print("\n=== Q2: SIMULATING DEFAULTS AND CALCULATING LOSSES ===")
    loans = loan_portfolio
    num_loans = len(loans)
    monthly_pd = loans['Monthly Prob. Default'].values
    factor_sims_np = interest_rate_factors.values
    n_sims = factor_sims_np.shape[1]

    # Standardize the factor (Z)
    factor_mean = np.mean(factor_sims_np)
    factor_std = np.std(factor_sims_np)
    factor_std = factor_std if factor_std > 0 else 1.0
    standardized_factor_sims = (factor_sims_np - factor_mean) / factor_std

    # Initialize arrays
    loan_default_status = np.zeros((num_loans, N_MONTHS + 1, n_sims), dtype=int) 
    credit_loss_ts = np.zeros((N_MONTHS, n_sims))

    # Generate idiosyncratic shocks (once for efficiency)
    idiosyncratic_shocks = norm.rvs(size=(num_loans, N_MONTHS, n_sims))

    # Calculate default thresholds (once for efficiency)
    default_thresholds = norm.ppf(np.clip(monthly_pd, 1e-10, 1-1e-10))

    # Gaussian Copula parameters
    rho = CORRELATION_FACTOR_DEFAULTS # desired Corr(M, Z)
    sqrt_one_minus_rho_sq = np.sqrt(1 - rho**2)

    for t in tqdm(range(N_MONTHS), desc="Q2: Simulating defaults"):
        # Systematic factor for month t, all simulations
        factor_t = standardized_factor_sims[t, :]

        # Calculate latent variable M_its for month t
        factor_component = rho * factor_t[np.newaxis, :]
        idio_component = sqrt_one_minus_rho_sq * idiosyncratic_shocks[:, t, :]
        M_its_t = factor_component + idio_component

        # Determine potential defaults in month t
        thresholds_broadcast = default_thresholds[:, np.newaxis]
        potential_defaults_t = M_its_t < thresholds_broadcast

        # Identify *new* defaults (only if not already defaulted)
        already_defaulted = (loan_default_status[:, t, :] == 1)
        new_defaults_t = potential_defaults_t & (~already_defaulted)

        # Update default status for the *next* month
        loan_default_status[:, t+1, :] = np.where(new_defaults_t | already_defaulted, 1, 0)

        # Calculate losses for *this* month (t) based on *new* defaults
        exposure_at_default = outstanding_principal[:, t]
        loss_given_default = exposure_at_default * (1.0 - LOAN_RECOVERY_RATE)

        # Calculate total loss for month t across all simulations
        loss_given_default_expanded = loss_given_default[:, np.newaxis]
        losses_this_month = new_defaults_t * loss_given_default_expanded
        credit_loss_ts[t, :] = losses_this_month.sum(axis=0)

    # Calculate accumulated losses and defaults
    accumulated_losses_ts = credit_loss_ts.cumsum(axis=0)
    total_defaults_ts = loan_default_status[:, 1:, :].sum(axis=0) # Sum defaults across loans for each month/sim

    # Q2.c Analysis: VaR, Expected Loss, Total Value
    results_q2c = []
    for t in range(N_MONTHS):
        losses_t_acc = accumulated_losses_ts[t, :]
        value_t_dfv = portfolio_value_ts[t, :] # Default-Free Value at start of month t+1 (end of month t)
        var_99 = np.percentile(losses_t_acc, 99)
        expected_loss = losses_t_acc.mean()
        avg_portfolio_value = value_t_dfv.mean()
        results_q2c.append({
            'Month': t + 1,
            'Avg Portfolio Value (DFV)': avg_portfolio_value,
            'Expected Loss (EL)': expected_loss,
            '99% VaR (Accumulated Loss)': var_99
        })
    results_q2c_df = pd.DataFrame(results_q2c)
    print("\nQ2.c: Summary Statistics (Losses based on Outstanding Principal EAD)")
    print(results_q2c_df.tail())

    # Plotting
    plt.figure()
    plt.plot(range(1, N_MONTHS + 1), total_defaults_ts.mean(axis=1))
    plt.title('Average Accumulated Defaults Over Time')
    plt.xlabel('Month'); plt.ylabel('Number of Defaults')
    plt.grid(True); plt.savefig(os.path.join(PLOTS_DIR, 'q2_accumulated_defaults.png')); plt.close()

    plt.figure()
    plt.plot(range(1, N_MONTHS + 1), accumulated_losses_ts.mean(axis=1))
    plt.title('Average Accumulated Losses Over Time (Correct EAD)')
    plt.xlabel('Month'); plt.ylabel('Loss Amount ($)')
    plt.grid(True); plt.savefig(os.path.join(PLOTS_DIR, 'q2_accumulated_losses.png')); plt.close()

    plt.figure()
    plt.plot(results_q2c_df['Month'], results_q2c_df['99% VaR (Accumulated Loss)'], 'r-', label='99% VaR')
    plt.plot(results_q2c_df['Month'], results_q2c_df['Expected Loss (EL)'], 'b-', label='Expected Loss')
    plt.title('VaR and Expected Loss Over Time (Correct EAD)')
    plt.xlabel('Month'); plt.ylabel('Amount ($)')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(PLOTS_DIR, 'q2_var_el_over_time.png')); plt.close()

    # Save Q2c results
    results_q2c_df.to_csv(os.path.join(DATA_OUT_DIR, 'q2_credit_risk_stats.csv'), index=False)

    # Q2.d: Correlation between monthly loss and factor (Right/Wrong Way Risk)
    correlations = []
    monthly_losses = credit_loss_ts
    monthly_factors = standardized_factor_sims # Use the standardized factor
    for t in range(N_MONTHS):
        # Calculate correlation only if there's variation in both series
        if np.std(monthly_losses[t, :]) > 0 and np.std(monthly_factors[t, :]) > 0:
            corr = np.corrcoef(monthly_losses[t, :], monthly_factors[t, :])[0, 1]
        else:
            corr = np.nan # Assign NaN if no variation
        correlations.append(corr)

    avg_corr = np.nanmean(correlations) # Use nanmean to ignore potential NaNs
    print(f"\nQ2.d: Average Monthly Correlation (Loss vs Factor): {avg_corr:.4f}")

    # Add correlation to results and save again
    results_q2c_df['Monthly_Loss_Factor_Corr'] = correlations
    results_q2c_df.to_csv(os.path.join(DATA_OUT_DIR, 'q2_credit_risk_stats.csv'), index=False)

    # Plot correlation over time
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_MONTHS + 1), correlations)
    plt.xlabel('Month'); plt.ylabel('Correlation')
    plt.title('Monthly Correlation between Credit Loss and Interest Rate Factor')
    plt.grid(True); plt.savefig(os.path.join(PLOTS_DIR, 'q2_loss_factor_correlation.png')); plt.close()

    # Write summary report for Q2
    with open(os.path.join(REPORTS_DIR, 'q2_default_simulation_summary.txt'), 'w') as f:
        f.write("=== Q2: DEFAULT SIMULATION SUMMARY ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(" - Gaussian copula model for defaults.\n")
        f.write(f" - Correlation (Factor-Default): {CORRELATION_FACTOR_DEFAULTS:.2f}.\n")
        f.write(" - EAD = Outstanding Principal.\n")
        f.write(f" - Recovery Rate: {LOAN_RECOVERY_RATE:.0%}.\n")
        f.write(" - Amortization is deterministic.\n\n")

        f.write("Q2.a: Accumulated Defaults\n")
        f.write(f"Average accumulated defaults at end: {total_defaults_ts[-1, :].mean():.2f}\n")
        f.write(f"Max accumulated defaults at end: {total_defaults_ts[-1, :].max()}\n\n")

        f.write("Q2.b: Accumulated Credit Losses\n")
        f.write(f"Average accumulated loss at end: ${accumulated_losses_ts[-1, :].mean():,.2f}\n")
        f.write(f"Max accumulated loss at end: ${accumulated_losses_ts[-1, :].max():,.2f}\n\n")

        f.write("Q2.c: VaR, Expected Loss, Total Value\n")
        f.write("See q2_credit_risk_stats.csv for detailed monthly data\n\n")

        f.write("Q2.d: Right-Way vs Wrong-Way Risk\n")
        f.write(f"Average Monthly Correlation (Loss vs Factor): {avg_corr:.4f}\n")
        if avg_corr > 0.05:
            f.write("Positive correlation suggests potential WRONG-WAY risk (higher factor associated with higher losses).\n")
        elif avg_corr < -0.05:
            f.write("Negative correlation suggests potential RIGHT-WAY risk (higher factor associated with lower losses).\n")
        else:
            f.write("Correlation close to zero suggests limited direct WWR/RWR from this factor.\n")

    return accumulated_losses_ts, total_defaults_ts, results_q2c_df, standardized_factor_sims

def calculate_derivative_value(portfolio_value_ts, accumulated_losses_ts):
    """Q3: Calculate the value of the derivative (single trigger)."""
    print("\n=== Q3: DERIVATIVE VALUATION (Single Trigger) ===")
    n_sims = portfolio_value_ts.shape[1]
    monthly_discount_rate = get_monthly_discount_rate(RISK_FREE_RATE_5Y_ANNUAL)

    # Determine trigger condition for each month and simulation
    # Trigger: Accumulated Losses >= Threshold * Default-Free Value (DFV)
    threshold_values = DERIVATIVE_THRESHOLD * portfolio_value_ts
    # Ensure loss is > 0 to avoid triggering at month 0 if threshold is 0
    trigger_hit_ts = (accumulated_losses_ts > 0) & (accumulated_losses_ts >= threshold_values) # Boolean, Shape (N_MONTHS, n_sims)

    # Find the *first* month the trigger is hit for each simulation
    # np.argmax returns the index of the first True value along axis 0
    # If no True value, it returns 0. We need to handle this.
    first_trigger_month_idx_s = np.full(n_sims, -1, dtype=int) # Initialize with -1 (not triggered)
    triggered_sims = np.any(trigger_hit_ts, axis=0) # Check if trigger ever hit for each sim

    if np.any(triggered_sims):
        # For sims that triggered, find the first index (month)
        first_trigger_month_idx_s[triggered_sims] = np.argmax(trigger_hit_ts[:, triggered_sims], axis=0)

    # Calculate Payoffs and PVs for each simulation
    cost_payoff_pv_s = np.zeros(n_sims) # PV of Cost leg (Firm pays DFV)
    benefit_payoff_pv_s = np.zeros(n_sims) # PV of Benefit leg (Dealer pays Factor * DFV)

    for s in range(n_sims):
        trigger_idx = first_trigger_month_idx_s[s]
        if trigger_idx != -1: # If the trigger was hit
            # Payoff occurs at the END of the trigger month (trigger_idx)
            # Discount from the end of month trigger_idx+1 back to time 0
            num_periods = trigger_idx + 1
            df = discount_factor(monthly_discount_rate, num_periods)

            # Cost: Firm pays DFV at the time of trigger
            cost_payoff_amount = portfolio_value_ts[trigger_idx, s]
            cost_payoff_pv_s[s] = cost_payoff_amount * df

            # Benefit: Dealer pays Factor * DFV at the time of trigger
            benefit_payoff_amount = DERIVATIVE_PAYOUT_FACTOR * portfolio_value_ts[trigger_idx, s]
            benefit_payoff_pv_s[s] = benefit_payoff_amount * df

    # Calculate Expected PVs by averaging across simulations
    expected_cost_pv = cost_payoff_pv_s.mean()
    expected_derivative_benefit_pv = benefit_payoff_pv_s.mean()
    expected_net_cost_pv = expected_cost_pv - expected_derivative_benefit_pv # Net Cost = Cost - Benefit

    print(f"Q3.a: Expected Cost of Derivative (PV): ${expected_cost_pv:,.2f}")
    print(f"Q3.b: Expected Benefit of Derivative (PV): ${expected_derivative_benefit_pv:,.2f}")
    print(f"   Net Cost (Cost - Benefit, PV, excl CVA): ${expected_net_cost_pv:,.2f}")

    # Analysis of trigger frequency and timing
    plt.figure()
    trigger_months_actual = first_trigger_month_idx_s[first_trigger_month_idx_s != -1] + 1 # Convert 0-based index to 1-based month
    trigger_freq = len(trigger_months_actual) / n_sims
    avg_trigger_month = trigger_months_actual.mean() if len(trigger_months_actual) > 0 else np.nan

    if len(trigger_months_actual) > 0:
        plt.hist(trigger_months_actual, bins=range(1, N_MONTHS + 2), alpha=0.7, align='left')
        plt.title(f'Month when Trigger Condition is First Met ({trigger_freq:.1%} of Sims)')
        plt.xlabel('Month (1-60)'); plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, 'q3_trigger_month_frequency.png'))
    else:
        plt.title('Trigger Condition Never Met in Simulations')
        plt.text(0.5, 0.5, 'Trigger Never Hit', horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(PLOTS_DIR, 'q3_trigger_month_frequency_none.png'))
    plt.close()

    # Save derivative data
    derivative_data = pd.DataFrame({
        'Simulation': range(n_sims),
        'First_Trigger_Month_Idx': first_trigger_month_idx_s, # 0-based index, -1 if not triggered
        'Cost_Payoff_PV': cost_payoff_pv_s,
        'Benefit_Payoff_PV': benefit_payoff_pv_s
    })
    derivative_data.to_csv(os.path.join(DATA_OUT_DIR, 'q3_derivative_valuation.csv'), index=False)

    # Write summary report
    with open(os.path.join(REPORTS_DIR, 'q3_derivative_valuation_summary.txt'), 'w') as f:
        f.write("=== Q3: DERIVATIVE VALUATION SUMMARY (Single Trigger) ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(" - Payoff occurs once at first trigger month.\n")
        f.write(" - Cost = DFV at trigger.\n")
        f.write(f" - Benefit = {DERIVATIVE_PAYOUT_FACTOR:.0%} * DFV at trigger.\n")
        f.write(" - Discounting uses 5Y risk-free rate.\n\n")
        f.write(f"Q3.a: Expected Cost of Derivative (PV): ${expected_cost_pv:,.2f}\n")
        f.write(f"Q3.b: Expected Benefit of Derivative (PV): ${expected_derivative_benefit_pv:,.2f}\n")
        f.write(f"   Net Cost (Cost - Benefit, PV, excl CVA): ${expected_net_cost_pv:,.2f}\n")
        f.write(f"Percentage of simulations where trigger is hit: {trigger_freq:.2%}\n")
        if not np.isnan(avg_trigger_month):
            f.write(f"Average month (1-60) when trigger is first hit: {avg_trigger_month:.2f}\n")
        if trigger_freq == 1.0:
            f.write("\nNote: Trigger frequency is 100% because accumulated losses generally increase while the portfolio value (DFV) amortizes towards zero, making it inevitable that the loss threshold (relative to current DFV) is eventually crossed.\n")

    return expected_cost_pv, expected_derivative_benefit_pv, expected_net_cost_pv, first_trigger_month_idx_s

def bootstrap_cds(cds_data):
    """Q4 Helper: Bootstrap marginal PDs from CDS spreads (simplified)."""
    print("Bootstrapping CDS spreads (simplified)...")
    cds_df = cds_data.copy()

    if len(cds_df) < N_MONTHS:
        print(f"Warning: CDS data only available for {len(cds_df)} months. Extrapolating last spread.")
        last_spread = cds_df['CDS Spread Decimal'].iloc[-1]
        extra_rows = pd.DataFrame({'CDS Spread Decimal': [last_spread] * (N_MONTHS - len(cds_df))}, index=range(len(cds_df), N_MONTHS))
        cds_df = pd.concat([cds_df[['CDS Spread Decimal']], extra_rows], ignore_index=True)
    else:
        cds_df = cds_df[['CDS Spread Decimal']].iloc[:N_MONTHS]

    cds_spreads_monthly = cds_df['CDS Spread Decimal'].values

    hazard_rates_annual = cds_spreads_monthly / CPTY_LGD
    monthly_hazard_rates = hazard_rates_annual / 12.0
    monthly_hazard_rates = np.maximum(0, monthly_hazard_rates)

    cumulative_hazard = np.cumsum(monthly_hazard_rates)
    survival_prob = np.exp(-cumulative_hazard)
    survival_prob = np.insert(survival_prob, 0, 1.0)

    marginal_pd = -np.diff(survival_prob)
    marginal_pd = np.maximum(0, marginal_pd)

    if marginal_pd.sum() > 1.0:
        marginal_pd /= marginal_pd.sum()

    print("CDS bootstrapping complete.")
    return marginal_pd # Returns array of length N_MONTHS

def calculate_cva(cds_data, first_trigger_month_idx_s, portfolio_value_ts, expected_net_cost_pv):
    """Q4: Calculate CVA using path-dependent EPE."""
    print("\n=== Q4: CREDIT VALUATION ADJUSTMENT (Path-Dependent EPE) ===")
    n_sims = portfolio_value_ts.shape[1]
    monthly_discount_rate = get_monthly_discount_rate(RISK_FREE_RATE_5Y_ANNUAL)

    marginal_pd = bootstrap_cds(cds_data)

    epe_profile = np.zeros(N_MONTHS)
    print("Calculating path-dependent EPE profile...")

    for t in tqdm(range(N_MONTHS), desc="Q4: Calculating EPE(t)"):
        exposure_t_s = np.zeros(n_sims)

        active_mask = (first_trigger_month_idx_s >= t)

        if np.any(active_mask):
            s_active = np.where(active_mask)[0]
            k_active = first_trigger_month_idx_s[s_active]

            payoff_amount = DERIVATIVE_PAYOUT_FACTOR * portfolio_value_ts[k_active, s_active]

            discount_periods = k_active - t
            pv_at_t = present_value(payoff_amount, monthly_discount_rate, discount_periods)

            exposure_t_s[s_active] = np.maximum(0, pv_at_t)

        epe_profile[t] = exposure_t_s.mean()

    print(f"Path-dependent EPE profile calculated. Avg EPE at t=0: {epe_profile[0]:,.2f}, t={N_MONTHS//2}: {epe_profile[N_MONTHS//2]:,.2f}")

    discount_factors_cva = discount_factor(monthly_discount_rate, np.arange(1, N_MONTHS + 1))
    cva_terms = epe_profile * marginal_pd * CPTY_LGD * discount_factors_cva
    cva = cva_terms.sum()

    print(f"\nQ4.b: Calculated CVA (using path-dependent EPE): ${cva:,.2f}")

    total_cost_with_cva = expected_net_cost_pv + cva
    print(f"   Total Net Cost of Derivative (Net Cost + CVA): ${total_cost_with_cva:,.2f}")

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    ax1.plot(range(1, N_MONTHS + 1), epe_profile, 'b-')
    ax1.set_title('Path-Dependent Expected Positive Exposure (EPE) Profile (Benefit Leg)')
    ax1.set_ylabel('Exposure ($)')
    ax1.grid(True)

    ax2 = plt.subplot(212)
    ax2.plot(range(1, N_MONTHS + 1), marginal_pd * 100, 'r-')
    ax2.set_title('Counterparty Marginal Default Probability per Month (from CDS)')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Probability (%)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'q4_cva_components.png'))
    plt.close()

    cva_data = pd.DataFrame({
        'Month': range(1, N_MONTHS + 1),
        'EPE_Profile_PathDependent': epe_profile,
        'Marginal_PD': marginal_pd,
        'Discount_Factor_CVA': discount_factors_cva,
        'CVA_Contribution': cva_terms
    })
    cva_data.to_csv(os.path.join(DATA_OUT_DIR, 'q4_cva_stats.csv'), index=False)

    with open(os.path.join(REPORTS_DIR, 'q4_cva_summary.txt'), 'w') as f:
        f.write("=== Q4: CREDIT VALUATION ADJUSTMENT (CVA) SUMMARY ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(" - Path-dependent EPE calculated for the derivative benefit leg.\n")
        f.write(" - Simplified CDS bootstrapping (Spread/LGD, ignores accrual & rate effects).\n")
        f.write(f" - Counterparty Recovery Rate: {CPTY_RECOVERY_RATE:.0%}.\n")
        f.write(" - Discounting uses 5Y risk-free rate.\n\n")
        f.write("Q4.a: CVA Explanation\n")
        f.write(" - CVA represents the market price of counterparty credit risk associated with the derivative.\n")
        f.write(" - It is the expected loss due to the counterparty defaulting before fulfilling its obligations.\n")
        f.write(" - Calculated as Sum[ EPE(t) * PD(t) * LGD * DF(t) ], where EPE is Expected Positive Exposure.\n\n")
        f.write("Q4.b: CVA Calculation\n")
        f.write(f"   Calculated CVA (using path-dependent EPE): ${cva:,.2f}\n")
        f.write(f"   Total Net Cost of Derivative (Net Cost + CVA): ${total_cost_with_cva:,.2f}\n")

    return cva, total_cost_with_cva

def analyze_funding_liquidity_risk(portfolio_value_ts, accumulated_losses_ts, total_cost_with_cva):
    """Q5: Analyze funding liquidity risk."""
    print("\n=== Q5: FUNDING LIQUIDITY RISK ===")
    monthly_discount_rate = get_monthly_discount_rate(RISK_FREE_RATE_5Y_ANNUAL)
    discount_factors = discount_factor(monthly_discount_rate, np.arange(1, N_MONTHS + 1))

    funding_trigger_level = FUNDING_LOSS_THRESHOLD * portfolio_value_ts
    excess_loss_ts = np.maximum(0, accumulated_losses_ts - funding_trigger_level)
    funding_cost_sims_ts = excess_loss_ts * CAPITAL_COST_RATE

    expected_funding_cost_monthly = funding_cost_sims_ts.mean(axis=1)
    wc_funding_cost_99_monthly = np.percentile(funding_cost_sims_ts, 99, axis=1)

    results_q5a_df = pd.DataFrame({
        'Month': range(1, N_MONTHS + 1),
        'Expected_Add_Funding_Cost': expected_funding_cost_monthly,
        '99%_WC_Add_Funding_Cost': wc_funding_cost_99_monthly
    })
    print("Q5.a: Expected and 99% Worst Case Additional Funding Cost per Month")
    print(results_q5a_df.tail())

    total_expected_funding_cost_pv = (expected_funding_cost_monthly * discount_factors).sum()
    print(f"\nTotal Expected Additional Funding Cost (PV): ${total_expected_funding_cost_pv:,.2f}")

    plt.figure()
    plt.plot(results_q5a_df['Month'], results_q5a_df['Expected_Add_Funding_Cost'], 'b-', label='Expected Cost (Flow)')
    plt.plot(results_q5a_df['Month'], results_q5a_df['99%_WC_Add_Funding_Cost'], 'r-', label='99% Worst Case Cost (Flow)')
    plt.title('Additional Funding Costs Over Time (Without Derivative)')
    plt.xlabel('Month'); plt.ylabel('Cost ($)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'q5_funding_costs.png'))
    plt.close()

    results_q5a_df.to_csv(os.path.join(DATA_OUT_DIR, 'q5_funding_liquidity.csv'), index=False)

    print("\nQ5.b: Discussion on Derivative Purchase")
    print(f" - Total Net Cost of Derivative (incl. CVA): ${total_cost_with_cva:,.2f}")
    print(f" - Total Expected Funding Cost (PV) without derivative: ${total_expected_funding_cost_pv:,.2f}")

    with open(os.path.join(REPORTS_DIR, 'q5_funding_liquidity_risk_summary.txt'), 'w') as f:
        f.write("=== Q5: FUNDING LIQUIDITY RISK SUMMARY ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(f" - Funding cost triggered if AccumLoss > {FUNDING_LOSS_THRESHOLD:.0%} * DFV.\n")
        f.write(f" - Cost = {CAPITAL_COST_RATE:.0%} * Excess Loss (applied monthly).\n")
        f.write(" - Discounting uses 5Y risk-free rate.\n\n")

        f.write("Q5.a: Expected and Worst Case Additional Funding Cost\n")
        f.write(f"Total Expected Additional Funding Cost (PV): ${total_expected_funding_cost_pv:,.2f}\n")
        f.write("See q5_funding_liquidity.csv for monthly flow details.\n\n")

        f.write("Q5.b: Discussion on Derivative Purchase\n")
        f.write(f" - Total Net Cost of Derivative (incl. CVA): ${total_cost_with_cva:,.2f}\n")
        f.write(f" - Total Expected Funding Cost (PV) without derivative: ${total_expected_funding_cost_pv:,.2f}\n")
        if total_cost_with_cva < total_expected_funding_cost_pv:
            f.write(" - The derivative appears cost-effective from an expected value perspective, as its net cost is lower than the expected funding costs it avoids.\n")
            f.write(" - Additionally, the derivative eliminates the tail risk of very high funding costs (represented by the 99% WC cost).\n")
            f.write(" - Recommendation: Purchase the derivative, as it provides a positive expected value and hedges against funding stress.\n")
        else:
            f.write(" - The derivative's net cost (including CVA) is higher than the expected funding cost without it.\n")
            f.write(" - This implies paying a premium to eliminate the tail risk of high funding costs in stress scenarios (represented by the 99% WC cost).\n")
            f.write(" - Recommendation: Evaluate the trade-off. If the firm is highly risk-averse or faces severe consequences from funding stress, the premium might be justified. Otherwise, retaining the funding risk might be preferred based purely on expected cost.\n")

    return results_q5a_df, total_expected_funding_cost_pv

def analyze_market_liquidity_risk(portfolio_value_ts, accumulated_losses_ts, bid_ask_spreads):
    """Q6: Analyze market liquidity risk."""
    print("\n=== Q6: MARKET LIQUIDITY RISK ===")
    n_sims = portfolio_value_ts.shape[1]

    if 'Proportional_Spread_Decimal' not in bid_ask_spreads.columns or bid_ask_spreads['Proportional_Spread_Decimal'].isnull().all():
        print("Error: Cannot find valid 'Proportional_Spread_Decimal' column in bid-ask data. Skipping Market Liquidity Analysis.")
        return pd.DataFrame()

    spreads = bid_ask_spreads['Proportional_Spread_Decimal'].values
    avg_spread = np.mean(spreads)
    spread_99_percentile = np.percentile(spreads, 99)
    print(f"Average historical proportional bid-ask spread (proxy): {avg_spread:.4%}")
    print(f"99th percentile historical proportional bid-ask spread (proxy): {spread_99_percentile:.4%}")

    remaining_value_ts = np.maximum(0, portfolio_value_ts - accumulated_losses_ts)

    expected_liquidation_cost_monthly = remaining_value_ts.mean(axis=1) * avg_spread

    stressed_liquidation_cost_monthly = remaining_value_ts.mean(axis=1) * spread_99_percentile
    remaining_value_99p = np.percentile(remaining_value_ts, 99, axis=1)
    stressed_wc_liquidation_cost_monthly = remaining_value_99p * spread_99_percentile

    results_q6_df = pd.DataFrame({
        'Month': np.arange(1, N_MONTHS + 1),
        'Avg_Remaining_Value': remaining_value_ts.mean(axis=1),
        'Expected_Liquidation_Cost': expected_liquidation_cost_monthly,
        'Stressed_Liquidation_Cost_AvgVal': stressed_liquidation_cost_monthly,
        'Stressed_Liquidation_Cost_99Val': stressed_wc_liquidation_cost_monthly
    })
    print("\nQ6.a & Q6.b: Liquidation Costs per Month")
    print(results_q6_df.tail())

    plt.figure()
    plt.plot(results_q6_df['Month'], results_q6_df['Expected_Liquidation_Cost'], 'b-', label='Expected Cost (Avg Spread)')
    plt.plot(results_q6_df['Month'], results_q6_df['Stressed_Liquidation_Cost_AvgVal'], 'r-', label='Stressed Cost (99% Spread, Avg Val)')
    plt.plot(results_q6_df['Month'], results_q6_df['Stressed_Liquidation_Cost_99Val'], 'r--', label='Stressed Cost (99% Spread, 99% Val)')
    plt.title('Estimated Portfolio Liquidation Costs Over Time')
    plt.xlabel('Month'); plt.ylabel('Cost ($)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'q6_liquidation_costs.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(bid_ask_spreads['Proportional_Spread_Decimal'] * 100, bins=30, alpha=0.7, density=True)
    plt.axvline(avg_spread * 100, color='b', linestyle='--', label=f'Mean: {avg_spread:.2%}')
    plt.axvline(spread_99_percentile * 100, color='r', linestyle='--', label=f'99th %ile: {spread_99_percentile:.2%}')
    plt.title('Distribution of Historical Bid-Ask Spreads (Proxy)')
    plt.xlabel('Proportional Spread (%)'); plt.ylabel('Density')
    plt.legend(); plt.grid(axis='y')

    plt.subplot(1, 2, 2)
    final_liq_costs = remaining_value_ts[-1, :] * avg_spread
    sns.histplot(final_liq_costs, bins=30, kde=True)
    plt.axvline(expected_liquidation_cost_monthly[-1], color='b', linestyle='--', label=f'Mean Cost: ${expected_liquidation_cost_monthly[-1]:,.0f}')
    plt.title(f'Distribution of Liquidation Cost (Month {N_MONTHS})')
    plt.xlabel('Liquidation Cost ($)'); plt.ylabel('Frequency')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'q6_spread_cost_distributions.png'))
    plt.close()

    results_q6_df.to_csv(os.path.join(DATA_OUT_DIR, 'q6_market_liquidity.csv'), index=False)

    with open(os.path.join(REPORTS_DIR, 'q6_market_liquidity_risk_summary.txt'), 'w') as f:
        f.write("=== Q6: MARKET LIQUIDITY RISK SUMMARY ===\n\n")
        f.write("Key Assumptions:\n")
        f.write(" - Historical ABS spreads used as proxy for loan portfolio liquidation costs.\n")
        f.write(" - Liquidation cost = Remaining Value - Accumulated Credit Losses.\n")
        f.write(" - Remaining Value = Default-Free Value - Accumulated Credit Losses.\n\n")

        f.write("Q6.a & Q6.b: Liquidation Costs\n")
        f.write(f"Average historical bid-ask spread (proxy): {avg_spread:.4%}\n")
        f.write(f"99th percentile historical bid-ask spread (proxy): {spread_99_percentile:.4%}\n")
        f.write(f"Average expected liquidation cost at end (Month {N_MONTHS}): ${expected_liquidation_cost_monthly[-1]:,.2f}\n")
        f.write(f"Average stressed liquidation cost (99% spread, avg value) at end: ${stressed_liquidation_cost_monthly[-1]:,.2f}\n")
        f.write(f"99th percentile stressed liquidation cost (99% spread & 99% value) at end: ${stressed_wc_liquidation_cost_monthly[-1]:,.2f}\n")
        f.write("See q6_market_liquidity.csv for monthly details.\n\n")

        f.write("Q6.c: Discussion on Correlation Modeling (Value vs Spread)\n")
        f.write(" - The current model assumes spreads are independent of the portfolio's value/losses.\n")
        f.write(" - In reality, market stress often causes both portfolio value to decrease (due to higher defaults/discount rates) AND spreads to widen.\n")
        f.write(" - This positive correlation (lower value, higher spread) represents wrong-way liquidity risk, amplifying liquidation costs in stress scenarios.\n")
        f.write(" - Modeling this correlation is important for accurate stressed cost estimation. Methods include:\n")
        f.write("   - Historical analysis: Correlate historical ABS index values/returns with spread changes.\n")
        f.write("   - Copulas: Model the joint distribution of portfolio value drivers (e.g., rates, default factors) and spread drivers.\n")
        f.write("   - Factor models: Link both portfolio value and spreads to common macroeconomic factors.\n\n")

        f.write("Q6.d: Discussion on Appropriateness of ABS Spreads as Proxy\n")
        f.write(" - Pros: Related asset class (securitized loans), represents secondary market transaction costs, historical data may be available.\n")
        f.write(" - Cons:\n")
        f.write("   - Diversification: ABS pools are typically more diversified than a single portfolio.\n")
        f.write("   - Tranching: ABS spreads vary by tranche seniority; using an index might not match the specific risk profile.\n")
        f.write("   - Liquidity Mismatch: ABS are generally more liquid than whole loan portfolios, especially individual loans.\n")
        f.write("   - Portfolio Specifics: The actual portfolio's characteristics (e.g., subprime auto) might have significantly different liquidity.\n")
        f.write(" - Conclusion: ABS spreads are an imperfect proxy. They likely *underestimate* the true liquidation cost, especially under stressed market conditions, as the portfolio is less liquid than traded ABS. The actual cost could be substantially higher.\n")

    return results_q6_df

def provide_risk_management_recommendations(total_cost_with_cva, total_expected_funding_cost_pv, results_q2c_df, first_trigger_month_idx_s, results_q6_df):
    """Q7: Provide overall recommendations."""
    print("\n=== Q7: RISK MANAGEMENT RECOMMENDATIONS ===")
    n_sims = N_SIMULATIONS

    if results_q6_df.empty:
        print("Warning: Market liquidity results are empty. Using NaN for related recommendations.")
        final_exp_liq_cost = np.nan
        final_stress_liq_cost = np.nan
    else:
        final_exp_liq_cost = results_q6_df['Expected_Liquidation_Cost'].iloc[-1]
        final_stress_liq_cost = results_q6_df['Stressed_Liquidation_Cost_99Val'].iloc[-1]

    final_el = results_q2c_df['Expected Loss (EL)'].iloc[-1]
    final_var = results_q2c_df['99% VaR (Accumulated Loss)'].iloc[-1]

    with open(os.path.join(REPORTS_DIR, 'q7_risk_management_recommendations.txt'), 'w') as f:
        f.write("=== Q7: RISK MANAGEMENT RECOMMENDATIONS ===\n\n")
        f.write("Key Assumptions Summary:\n")
        f.write(f" - CIR Rates (Simulated), Gaussian Copula Defaults (Corr={CORRELATION_FACTOR_DEFAULTS:.2f}), Recovery Rates (Loan={LOAN_RECOVERY_RATE:.0%}, Cpty={CPTY_RECOVERY_RATE:.0%}), Simplified CDS Bootstrap, ABS Spread Proxy, 5Y RF Discounting.\n\n")

        f.write("Q7.a: Credit Risk Hedging and Liquidity Risks Faced\n")
        f.write(f" - Credit Risk Exposure: Significant. The portfolio faces substantial potential credit losses (Final EL ~${final_el:,.0f}, 99% VaR ~${final_var:,.0f}).\n")

        if total_cost_with_cva < total_expected_funding_cost_pv:
            f.write(f" - Derivative Hedge: Recommended. The net cost including CVA (${total_cost_with_cva:,.2f}) is less than the expected present value of additional funding costs (${total_expected_funding_cost_pv:,.2f}) that would be incurred without the hedge. It provides a positive expected value and eliminates tail funding risk.\n")
        else:
            f.write(f" - Derivative Hedge: Evaluate Cost vs. Benefit. The net cost including CVA (${total_cost_with_cva:,.2f}) exceeds the expected present value of avoided funding costs (${total_expected_funding_cost_pv:,.2f}). The firm would pay a premium to hedge tail funding risk. Consider purchase based on risk appetite and the potential severity of funding stress.\n")

        f.write(" - Liquidity Risks Faced:\n")
        f.write("   1. Funding Liquidity Risk: Potential need for costly capital if losses exceed the threshold ({FUNDING_LOSS_THRESHOLD:.0%} of DFV). This risk is mitigated (or eliminated if triggered) by purchasing the derivative.\n".format(FUNDING_LOSS_THRESHOLD=FUNDING_LOSS_THRESHOLD))
        f.write(f"   2. Market Liquidity Risk: Significant cost to sell the portfolio quickly, especially in stress. Estimated final cost: Avg ${final_exp_liq_cost:,.0f}, Stress 99%/99% ${final_stress_liq_cost:,.0f}. The ABS proxy likely underestimates the true cost significantly.\n\n")

        f.write("Q7.b: Other Risks to Consider\n")
        f.write(" - Model Risk: Assumptions regarding interest rate model (CIR), default correlation (Gaussian copula, fixed rho), recovery rates (fixed), CVA calculation (simplified bootstrap, EPE paths), market liquidity proxy (ABS spreads), and constant discount rate may not hold.\n")
        f.write(" - Prepayment Risk: Not modeled. Faster/slower prepayments affect portfolio value and duration.\n")
        f.write(" - Operational Risk: Loan servicing issues, fraud, system failures.\n")
        f.write(" - Systemic Risk / Market Environment: A broad market downturn could simultaneously increase defaults, decrease recovery rates, increase counterparty risk, widen market spreads, and make funding difficult (correlated impacts).\n")
        f.write(" - Regulatory Risk: Changes in capital requirements or consumer lending regulations.\n")
        f.write(" - Concentration Risk: Exposure concentrated in a specific sector (e.g., subprime auto loans), making the portfolio vulnerable to sector-specific downturns.\n\n")

        f.write("Q7.c: Mitigating Counterparty and Liquidity Risks\n")
        f.write(" - Counterparty Risk (Derivative): \n")
        f.write("   - Use collateral agreements (Credit Support Annex - CSA) with margin calls.\n")
        f.write("   - Transact with high-quality counterparties; monitor their creditworthiness (e.g., CDS spreads).\n")
        f.write("   - Diversify counterparty exposure if possible.\n")
        f.write("   - Ensure CVA is accurately priced into the derivative cost.\n")
        f.write(" - Funding Liquidity Risk:\n")
        f.write("   - Maintain adequate capital buffers above regulatory minimums.\n")
        f.write("   - Develop and test Contingency Funding Plans (CFPs) outlining actions during stress.\n")
        f.write("   - Diversify funding sources (deposits, repo, credit lines).\n")
        f.write("   - Conduct regular funding stress tests.\n")
        f.write(" - Market Liquidity Risk:\n")
        f.write("   - Hold a buffer of highly liquid assets.\n")
        f.write("   - Explore potential for securitization to transform illiquid loans into more liquid securities (if feasible and economical).\n")
        f.write("   - Establish committed credit lines that can be drawn if asset sales are too costly.\n")
        f.write("   - Limit portfolio size relative to market depth.\n")
        f.write("   - Conduct stress tests incorporating stressed liquidation costs and potential market freezes.\n")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Risk Management Dashboard Summary", fontsize=16)

    plt.subplot(221)
    plt.plot(results_q2c_df['Month'], results_q2c_df['Expected Loss (EL)'], 'b-', label='Expected Loss')
    plt.plot(results_q2c_df['Month'], results_q2c_df['99% VaR (Accumulated Loss)'], 'r-', label='99% VaR')
    plt.title('Credit Risk Profile')
    plt.xlabel('Month'); plt.ylabel('Amount ($)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(); plt.grid(True)

    plt.subplot(222)
    costs = [abs(total_cost_with_cva), total_expected_funding_cost_pv]
    labels = ['Net Derivative Cost\n(incl CVA)', 'Expected Funding Cost\n(PV, No Derivative)']
    colors = ['orange', 'lightblue']
    
    costs = [c if c > 0 else 0 for c in costs]
    bars = plt.bar(labels, costs, color=colors)
    plt.title('Derivative Cost vs. Funding Cost Avoided')
    plt.ylabel('Present Value ($)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'${yval:,.0f}', va='bottom', ha='center', fontsize=9)

    plt.subplot(223)
    if not results_q6_df.empty:
        plt.plot(results_q6_df['Month'], results_q6_df['Expected_Liquidation_Cost'], 'b-', label='Expected Cost (Avg Spread)')
        plt.plot(results_q6_df['Month'], results_q6_df['Stressed_Liquidation_Cost_99Val'], 'r-', label='Stressed Cost (99% Spread/Val)')
        plt.title('Market Liquidity Risk (Liquidation Cost)')
        plt.xlabel('Month'); plt.ylabel('Cost ($)')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(); plt.grid(True)
    else:
        plt.title('Market Liquidity Risk (Data Missing)')
        plt.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

    plt.subplot(224)
    trigger_months_actual = first_trigger_month_idx_s[first_trigger_month_idx_s != -1] + 1
    trigger_freq = len(trigger_months_actual) / n_sims if n_sims > 0 else 0
    avg_trigger_month = trigger_months_actual.mean() if len(trigger_months_actual) > 0 else np.nan

    if len(trigger_months_actual) > 0:
        plt.hist(trigger_months_actual, bins=range(1, N_MONTHS + 2), alpha=0.7, align='left', color='green')
        plt.title(f'Derivative Trigger Month Freq. ({trigger_freq:.1%} Hit)')
        plt.xlabel('Month (1-60)'); plt.ylabel('Frequency')
        plt.grid(axis='y')
    else:
        plt.title('Derivative Trigger Never Hit')
        plt.text(0.5, 0.5, 'No Triggers', horizontalalignment='center', verticalalignment='center')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, 'q7_risk_management_dashboard.png'))
    plt.close()


# --- Main Execution ---

def main():
    """Main execution function"""
    create_output_directories()
    data = load_data()

    # Unpack data safely
    loan_portfolio = data.get('loan_portfolio')
    cds_data = data.get('cds_data')
    interest_rate_factors = data.get('interest_rate_factors')
    interest_rate_forecasts = data.get('interest_rate_forecasts')
    bid_ask_spreads = data.get('bid_ask_spreads')

    # Check if all necessary data loaded before proceeding
    if any(v is None for v in [loan_portfolio, cds_data, interest_rate_factors, interest_rate_forecasts, bid_ask_spreads]):
        print("ERROR: Not all required dataframes were loaded successfully. Exiting.")
        sys.exit(1)


    # --- Execute Analysis Steps ---

    outstanding_principal = calculate_outstanding_principal(loan_portfolio)

    portfolio_value_ts = calculate_portfolio_exposure(loan_portfolio, interest_rate_forecasts)

    accumulated_losses_ts, total_defaults_ts, results_q2c_df, standardized_factor_sims = simulate_defaults_and_losses(
        loan_portfolio, outstanding_principal, interest_rate_factors, portfolio_value_ts
    )

    expected_cost_pv, expected_derivative_benefit_pv, expected_net_cost_pv, first_trigger_month_idx_s = calculate_derivative_value(
        portfolio_value_ts, accumulated_losses_ts
    )

    cva, total_cost_with_cva = calculate_cva(
        cds_data, first_trigger_month_idx_s, portfolio_value_ts, expected_net_cost_pv
    )

    results_q5a_df, total_expected_funding_cost_pv = analyze_funding_liquidity_risk(
        portfolio_value_ts, accumulated_losses_ts, total_cost_with_cva
    )

    results_q6_df = analyze_market_liquidity_risk(
        portfolio_value_ts, accumulated_losses_ts, bid_ask_spreads
    )

    provide_risk_management_recommendations(
        total_cost_with_cva, total_expected_funding_cost_pv, results_q2c_df, first_trigger_month_idx_s, results_q6_df
    )

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
