import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Weibull_2P
import numpy as np


def get_test_input(test_num):
    """Collect test data from user input"""
    print(f"\n--- Enter data for Test {test_num} ---")
    
    # Get Delta T
    delta_t = float(input(f"Enter Delta T for Test {test_num} (°C): "))
    
    # Get failure data
    failures_input = input(f"Enter failure data for Test {test_num} (comma-separated values): ")
    failures = [float(x.strip()) for x in failures_input.split(",")]
    
    # Get censored data
    censored_input = input(f"Enter censored data for Test {test_num} (comma-separated values): ")
    censored = [float(x.strip()) for x in censored_input.split(",")]
    
    return delta_t, failures, censored


def empirical_unreliability(failures, censored):
    """
    Return (sorted_failures, unreliabilities(between 0-1)) for the data.
    """
    # total number of samples
    n = len(failures) + len(censored)
    sorted_f = sorted(failures)
    # simple rank method
    unreliab = [((i + 1) / n) for i in range(len(sorted_f))]
    return sorted_f, unreliab


def weibull_line(t_val, a, b):
    return b * np.log(t_val) - b * np.log(a)


def plot_weibull_custom(fit_model, delta_t, failures, censored, color, label):
    """Plot failure data and fitted Weibull model with confidence bounds"""
    ax = plt.gca()
    
    # Get fit parameters
    alpha = fit_model.alpha
    beta = fit_model.beta
    alpha_lower = fit_model.alpha_lower
    alpha_upper = fit_model.alpha_upper
    beta_lower = fit_model.beta_lower
    beta_upper = fit_model.beta_upper
    
    # Create time range for the fitted curve
    t_min = min(failures) * 0.8
    t_max = max(failures) * 1.2
    t_range = np.geomspace(t_min, t_max, 1000)

    x = np.log(t_range)
    
    # Calculate ln(-ln(1 - F(t)))$ values for fitted distribution 
    y_fit = weibull_line(t_range, alpha, beta)
    
    # Calculate confidence bounds
    y_1 = weibull_line(t_range, alpha_lower, beta_lower)
    y_2 = weibull_line(t_range, alpha_upper, beta_upper)
    y_3 = weibull_line(t_range, alpha_lower, beta_upper)
    y_4 = weibull_line(t_range, alpha_upper, beta_lower)
    y_lower = np.minimum.reduce([y_1, y_2, y_3, y_4])
    y_upper = np.maximum.reduce([y_1, y_2, y_3, y_4])
    
    # Plot confidence bounds as shaded region
    ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)
    
    # Plot fitted curve
    ax.plot(x, y_fit, color=color, linewidth=2, label=f"Fitted Curve: {label}")
    
    # Plot failure data as scatter points
    sorted_f, unreliab = empirical_unreliability(failures, censored)
    failure_x = np.log(sorted_f)
    failure_y = np.log(-np.log(1 - np.array(unreliab)))
    ax.plot(failure_x, failure_y, 'o', color=color, markersize=3, alpha=0.7, label=f"Test Data: {label}")


def main():
    # Run Weibull analysis for two tests
    plt.figure(figsize=(12, 7))
    
    # Collect and fit Test 1
    delta_t1, failures1, censored1 = get_test_input(1)

    fit1 = Fit_Weibull_2P(
        failures=failures1, 
        right_censored=censored1,
        show_probability_plot=False,
        print_results=True,
        CI=0.95
    )
    plot_weibull_custom(fit1, delta_t1, failures1, censored1, "red", f"ΔT={delta_t1}C")
    
    # Collect and fit Test 2
    delta_t2, failures2, censored2 = get_test_input(2)

    fit2 = Fit_Weibull_2P(
        failures=failures2, 
        right_censored=censored2, 
        show_probability_plot=False,
        print_results=True,
        CI=0.95
    )
    plot_weibull_custom(fit2, delta_t2, failures2, censored2, "blue", f"ΔT={delta_t2}C")
    
    # Format plot
    ax = plt.gca()
    x_ticks_val = np.array([500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticks(np.log(x_ticks_val))
    ax.set_xticklabels([f"{v:,}" for v in x_ticks_val])
    ax.set_xlabel("Time (Cycles)", fontsize=12)

    f_ticks_pct = np.array([1, 5, 10, 20, 40, 63.2, 80, 95, 99])
    # Transform % to Y-coordinate: ln(-ln(1 - F))
    y_ticks_pos = np.log(-np.log(1 - f_ticks_pct/100))
    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels([f"{v}%" for v in f_ticks_pct])

    ax.set_ylim(np.log(-np.log(1 - 1/100)), np.log(-np.log(1 - 95/100)))

    ax.set_ylabel("Unreliability F(t) (%)", fontsize=12)
    ax.set_title("Weibull Probability Plot", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()