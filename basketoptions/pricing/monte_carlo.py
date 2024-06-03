import numpy as np
import sobol_seq
from scipy.stats import norm, qmc

def monte_carlo_pricing(model, strike_price, option_type='call', num_paths=10000):
    """
    Prices the option using Monte Carlo simulation.

    Parameters:
        model: The model instance with a `generate_paths` method.
        strike_price: The strike price of the option.
        option_type: The type of the option ('call' or 'put'). Default is 'call'.
        num_paths: The number of paths to simulate. Default is 10000.

    Returns:
        The price of the option.
    """
    price_paths = model.generate_paths(num_paths)
    basket_prices = np.dot(price_paths[:, -1, :], model.weights)

    if option_type == 'call':
        payoffs = np.maximum(basket_prices - strike_price, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike_price - basket_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = np.exp(-model.risk_free_rate * model.time_to_maturity) * payoffs
    return np.mean(discounted_payoffs)

def qmc_pricing(model, strike_price, option_type='call', num_paths=50000):
    """
    Prices a basket option using a Quasi-Monte Carlo method with Halton sequences.

    Parameters:
        model: The pricing model instance.
        strike_price: The strike price of the option.
        option_type: The type of the option ('call' or 'put').
        num_paths: The number of QMC samples to use.

    Returns:
        The price of the option.
    """

    S0 = np.array(model.spot_prices)
    T = model.time_to_maturity
    r = model.risk_free_rate
    sigma = np.array(model.volatilities)
    weights = np.array(model.weights)
    correlation_matrix = np.array(model.correlation_matrix)
    num_assets = model.num_assets
    num_steps = model.num_steps

    # Generate Halton sequences
    sampler = qmc.Halton(d=num_assets * num_steps, scramble=True)
    halton_seq_samples = sampler.random(n=num_paths)
    halton_seq_samples = norm.ppf(halton_seq_samples)  # Map to standard normal
    halton_seq_samples = halton_seq_samples.reshape(num_paths, num_steps, num_assets)

    # Generate paths using Halton sequence samples
    price_paths = model.generate_paths(num_paths, sobol_seq_samples=halton_seq_samples)
    basket_prices = np.dot(price_paths[:, -1, :], weights)

    # Calculate payoff
    if option_type == 'call':
        payoffs = np.maximum(basket_prices - strike_price, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike_price - basket_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount payoffs back to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoffs)