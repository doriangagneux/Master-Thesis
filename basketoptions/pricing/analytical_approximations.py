import numpy as np
from scipy.stats import norm


def moment_matching_approximation(model, strike_price, option_type='call'):
    """
    Prices the option using moment-matching approximation.

    Parameters:
        model: The pricing model instance.
        strike_price: The strike price of the option.
        option_type: The type of the option ('call' or 'put').

    Returns:
        The approximated price of the option.
    """
    # Calculate the mean of the basket
    mean = np.dot(model.weights, model.spot_prices * np.exp((model.risk_free_rate - model.dividend_yields) * model.time_to_maturity))
    
    # Calculate the variance of the basket using the provided formula
    variance = 0
    # First term: Sum of weighted variances
    for i in range(model.num_assets):
        variance += (model.weights[i] ** 2) * (model.volatilities[i] ** 2)
    
    # Second term: Sum of covariances
    for i in range(model.num_assets):
        for j in range(i + 1, model.num_assets):
            variance += 2 * model.weights[i] * model.weights[j] * model.correlation_matrix[i, j] * model.volatilities[i] * model.volatilities[j]
    
    stddev = np.sqrt(variance) * np.sqrt(model.time_to_maturity)
    
    # Use the Black-Scholes formula to approximate the option price
    d1 = (np.log(mean / strike_price) + 0.5 * stddev**2) / stddev
    d2 = d1 - stddev
    
    if option_type == 'call':
        price = np.exp(-model.risk_free_rate * model.time_to_maturity) * (mean * norm.cdf(d1) - strike_price * norm.cdf(d2))
    elif option_type == 'put':
        price = np.exp(-model.risk_free_rate * model.time_to_maturity) * (strike_price * norm.cdf(-d2) - mean * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price




def taylor_series_approximation(model, strike_price, option_type, spot_prices, order = 1, initial_price = None):
    # Evaluate the option price at the initial asset prices
    if not initial_price:
        initial_price = model.price(strike_price, option_type)

    # Calculate the Greeks (delta and gamma)
    if not model.greekvalues:
        model.greeks(strike_price, option_type)
    
    delta = model.greekvalues['delta']
    gamma = model.greekvalues['gamma']

    # Apply the Taylor series approximation formula
    deviation = spot_prices - model.spot_prices
    if order == 1:
        return initial_price + np.dot(delta, deviation)
    else:
        return initial_price + np.dot(delta, deviation) + 0.5 * np.dot(deviation, np.dot(gamma, deviation))




