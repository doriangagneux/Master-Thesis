
import numpy as np

def gaussian_quadrature_pricing(model, strike_price, option_type='call', num_points=100):
    """
    Prices the option using Gaussian quadrature.

    Parameters:
        model: The model instance with a `generate_paths` method.
        strike_price: The strike price of the option.
        option_type: The type of the option ('call' or 'put'). Default is 'call'.
        num_points: The number of quadrature points. Default is 50.

    Returns:
        The price of the option.
    """
    def payoff_function(x):
        basket_prices = np.dot(x, model.weights)
        if option_type == 'call':
            return np.maximum(basket_prices - strike_price, 0)
        elif option_type == 'put':
            return np.maximum(strike_price - basket_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # Generate quadrature points and weights
    points, weights = np.polynomial.legendre.leggauss(num_points)
    
    # Adjust points and weights for integration interval [0, 1]
    points = 0.5 * points + 0.5
    weights *= 0.5

    # Generate asset price paths
    price_paths = model.generate_paths(num_points)
    basket_prices = price_paths[:, -1, :]

    # Calculate payoffs
    payoffs = np.array([payoff_function(basket_prices[i, :]) for i in range(num_points)])

    # Calculate option price
    option_price = np.dot(weights, payoffs) * np.exp(-model.risk_free_rate * model.time_to_maturity)
    
    return option_price