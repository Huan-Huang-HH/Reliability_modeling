import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2


def calculate_test_frequency(min_temp, max_temp, dwell_time_min, ramp_rate_c_min):
    """
    Calculates the thermal cycling frequency (f_test) in cycles per day.
    
    Inputs:
    - min_temp: Minimum cycle temperature (deg C)
    - max_temp: Maximum cycle temperature (deg C)
    - dwell_time_min: Dwell time at EACH extreme (minutes)
    - ramp_rate_c_min: Ramp rate for both heating and cooling (deg C/min)
    """

    # 1. Calculate the temperature swing (Delta T)
    delta_t = max_temp - min_temp

    # 2. Calculate time spent ramping (2 ramps per cycle: up and down)
    # Time = Distance / Speed
    time_per_ramp = delta_t / ramp_rate_c_min
    total_ramp_time = 2 * time_per_ramp
    
    # 3. Calculate total dwell time (2 dwells per cycle: hot and cold)
    total_dwell_time = 2 * dwell_time_min
    
    # 4. Total cycle period in minutes
    total_cycle_minutes = total_ramp_time + total_dwell_time
    
    # 5. Convert to cycles per day (1440 minutes in a day)
    cycles_per_day = 1440 / total_cycle_minutes
    
    return cycles_per_day


def calculate_norris_landzberg_af(f_test, f_field, dt_test, dt_field, tmax_test_c, tmax_field_c):
    """
    Calculates the Acceleration Factor (AF) using the Norris-Landzberg model.
    
    Inputs:
    - f_test: Test frequency (cycles per day or cycles per hour)
    - f_field: Field frequency (must use same units as f_test)
    - dt_test: Temperature swing in test (deg C)
    - dt_field: Temperature swing in field (deg C)
    - tmax_test_c: Maximum temperature in test (deg C)
    - tmax_field_c: Maximum temperature in field (deg C)
    """
    
    # Convert Max Temperatures to Kelvin for the Arrhenius term
    T_max_test_K = tmax_test_c + 273.15
    T_max_field_K = tmax_field_c + 273.15
    
    # 1. Temperature Swing Term (Exponent n = 1.9)
    af_temp = (dt_test / dt_field)**1.9
    
    # 2. Frequency Term (Exponent = 1/3)
    af_freq = (f_field / f_test)**(1/3)
    
    # 3. Arrhenius/Maximum Temperature Term (Constant = 1414)
    # This accounts for the increased creep rate at higher absolute temperatures
    af_arrhenius = math.exp(1414 * (1/T_max_field_K - 1/T_max_test_K))
    
    # Total Acceleration Factor
    af_total = af_temp * af_freq * af_arrhenius
    
    return af_total


def calculate_n_zero_failure(confidence, reliability):
    """
    Calculates the required sample size for a zero-failure test.
    
    Inputs:
    - confidence: Confidence level (e.g., 0.90 for 90%)
    - reliability: Reliability target (e.g., 0.99 for 99%)
    """
    # Success Run Formula: n = ln(1 - C) / ln(R)
    n = math.log(1 - confidence) / math.log(reliability)
    return math.ceil(n)


def calculate_n_one_failure(confidence, reliability):
    """
    Calculates the required sample size for a test allowing up to 1 failure.
    Uses the Chi-squared approximation method.
    
    Inputs:
    - confidence: Confidence level (e.g., 0.90)
    - reliability: Reliability target (e.g., 0.99)
    """
    # Degrees of freedom for Ac=1 is 2 * (1 + 1) = 4
    df = 4
    # chi2.ppf returns the value of the distribution for a given cumulative probability
    chi_sq_val = chi2.ppf(confidence, df)
    # Formula: n = chi_sq / (2 * (1 - R))
    n = chi_sq_val / (2 * (1 - reliability))
    return math.ceil(n)


def calculate_sample_size_weibull(C, R, L_field, L_test, beta):
    """
    Calculates sample size n for a zero-failure test.
    C: Confidence Level (e.g., 0.90)
    R: Reliability Target (e.g., 0.99)
    L_field: Required life in the field (e.g., 5000 cycles)
    L_test: Equivalent life demonstrated in test (Actual Cycles * AF)
    beta: Weibull shape factor
    """
    numerator = np.log(1 - C)
    # Success Run Formula rearranged for n:
    # n = ln(1-C) / (ln(R) * (L_test/L_field)^beta)
    denominator = np.log(R) * (L_test / L_field)**beta
    return numerator / denominator


def plot_sample_size_vs_test_duration(C, R, L_field, AF, betas):
    '''Plots sample size vs. test duration for different Weibull shape factors
    C: Confidence Level (e.g., 0.90)
    R: Reliability Target (e.g., 0.99)
    L_field: Required life in the field (e.g., 5000 cycles)
    AF: Acceleration Factor (e.g., 2.5)
    betas: List of Weibull shape factors to plot (e.g., [2, 3, 4])
    '''
    # Range of test durations: from 0.5x life to 3x life
    test_durations = np.linspace(L_field * 0.5, L_field * 3, 500)

    plt.figure(figsize=(10, 6))
    for beta in betas:
        n_values = calculate_sample_size_weibull(C, R, L_field, test_durations, beta)
        plt.plot(test_durations/AF, n_values, label=f'β = {beta}', linewidth=2)
    # Reference line for 1x Field Life
    plt.axvline(x=L_field/AF, color='black', linestyle='--', alpha=0.6, label='1x Design Life')

    plt.title(f'Sample Size vs. Test Duration (Confidence={C*100}%, Reliability={R*100:.2f}%)', fontsize=14)
    plt.xlabel('Number of cycles for zero-failure test', fontsize=12)
    plt.ylabel('Required Sample Size', fontsize=12)
    plt.yscale('log') # Log scale helps visualize the massive drop in n
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(title="Weibull Shape Factor (β)")
    plt.tight_layout()
    plt.show()


def main():
    # exmple usage of the functions

    # calculate test frequency
    min_temp = -10
    max_temp = 85
    dwell_time_min = 10
    ramp_rate_c_min = 10
    f_test = calculate_test_frequency(min_temp, max_temp, dwell_time_min, ramp_rate_c_min)
    print(f"Calculated Test Frequency: {f_test:.2f} cycles per day")

    # calculate AF
    f_field = 50
    dt_field = 125
    tmax_field_c = 55 # maximum temperature in the field (deg C)
    dt_test = max_temp - min_temp
    tmax_test_c = max_temp
    AF = calculate_norris_landzberg_af(f_test, f_field, dt_test, dt_field, tmax_test_c, tmax_field_c)
    print(f"Calculated Acceleration Factor (AF): {AF:.2f}")

    # calculate sample size for zero-failure test
    n_zero_failure = calculate_n_zero_failure(0.90, 0.90)
    print(f"Required Sample Size (Zero-Failure): {n_zero_failure}")

    # calculate sample size for one-failure test
    n_one_failure = calculate_n_one_failure(0.90, 0.90)
    print(f"Required Sample Size (Allowing 1 Failure): {n_one_failure}")

    C = 0.90 # target confidence level
    R = 0.90 # target reliability
    L_field = 5000 # required life in the field
    AF = 3 # example acceleration factor
    betas = [2, 2.5, 3, 3.5, 4] # range of shape factors
    plot_sample_size_vs_test_duration(C, R, L_field, AF, betas)


if __name__ == "__main__":
    main()

