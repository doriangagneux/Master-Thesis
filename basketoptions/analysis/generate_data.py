import numpy as np
from typing import Union
import matplotlib.pyplot as plt


def generate_random_spot_prices(num_assets: int):
    return np.random.uniform(50, 150, size=num_assets)

def generate_random_volatilities(num_assets: int):
    return np.random.uniform(0.1, 0.5, size=num_assets)

def generate_random_correlation_matrix(num_assets: int):
    A = np.random.uniform(-1, 1, size=(num_assets, num_assets))
    corr_matrix = np.dot(A, A.transpose())
    D = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / D[:, None]
    corr_matrix = corr_matrix / D[None, :]
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix

def generate_random_weights(num_assets: int):
    weights = np.random.uniform(0, 1, size=num_assets)
    return weights / weights.sum()

def generate_random_parameters(low=0.01, high=0.1, num_assets: int = 3):
    return np.random.uniform(low, high, size=num_assets)

def generate_random_float(low=0.01, high=0.1):
    return np.random.uniform(low, high)

def generate_random_model_inputs(model_type: str, num_assets: int, time_window: float = 1.0):
    inputs = {
        "spot_prices": generate_random_spot_prices(num_assets),
        "weights": generate_random_weights(num_assets),
        "risk_free_rate": generate_random_float(0.03, 0.15),
        "dividend_yields": generate_random_parameters(0.01, 0.05, num_assets),  # Corrected line
        "time_to_maturity": time_window,
        "num_steps": int(252 * time_window),
        "correlation_matrix": generate_random_correlation_matrix(num_assets)
    }

    if model_type == "BlackScholesModel":
        inputs["volatilities"] = generate_random_volatilities(num_assets)
    elif model_type == "JumpDiffusionProcess":
        inputs["drift"] = generate_random_float(0.01, 0.1)
        inputs["volatilities"] = generate_random_volatilities(num_assets)
        inputs["jump_intensity"] = generate_random_float(0.1, 1.0)
        inputs["jump_mean"] = generate_random_parameters(0.01, 0.05, num_assets)
        inputs["jump_std"] = generate_random_parameters(0.01, 0.05, num_assets)
    elif model_type == "HestonModel":
        inputs["initial_variances"] = generate_random_volatilities(num_assets)
        inputs["kappa"] = generate_random_parameters(0.1, 2.0, num_assets)
        inputs["theta"] = generate_random_parameters(0.01, 0.2, num_assets)
        inputs["sigma"] = generate_random_parameters(0.1, 0.5, num_assets)
        inputs["rho"] = generate_random_parameters(-0.9, 0.9, num_assets)
    elif model_type == "VarianceGammaProcess":
        inputs["volatilities"] = generate_random_volatilities(num_assets)
        inputs["theta"] = generate_random_parameters(-0.1, 0.1, num_assets)
        inputs["nu"] = generate_random_parameters(0.01, 0.1, num_assets)
    elif model_type == "SABRModel":
        inputs["initial_volatilities"] = generate_random_volatilities(num_assets)
        inputs["alpha"] = generate_random_parameters(0.1, 0.5, num_assets)
        inputs["beta"] = generate_random_parameters(0.3, 0.7, num_assets)
        inputs["rho"] = generate_random_parameters(-0.9, 0.9, num_assets)
        inputs["nu"] = generate_random_parameters(0.1, 0.5, num_assets)

    return inputs


def generate_market_situation(model_class, **model_params):
    # Create an instance of the model
    model = model_class(**model_params)
    
    # Generate one path (num_paths=1) over one year (time_to_maturity should be set in model_params)
    paths = model.generate_paths(num_paths=1)
    
    # Return the entire price history
    return paths[0]

# Plot the market situations
def plot_market_situation(market_situation, title):
    num_steps = market_situation.shape[0]
    time = np.linspace(0, 1, num_steps)
    for i in range(market_situation.shape[1]):
        plt.plot(time, market_situation[:, i], label=f'Asset {i+1}')
    plt.title(title)
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()