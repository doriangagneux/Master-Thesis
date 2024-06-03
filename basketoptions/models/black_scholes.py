import numpy as np
from scipy.stats import norm
from basketoptions.models.base_model import BaseModel
import matplotlib.pyplot as plt


class BlackScholesModel(BaseModel):

    def __init__(self, spot_prices, weights, volatilities, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, num_steps=252):
        super().__init__(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, num_steps)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)
        self.num_steps = num_steps

    def generate_paths(self, num_paths, sobol_seq_samples=None):
        """
        Generates price paths using the Black-Scholes model and Cholesky decomposition.

        Parameters:
            num_paths (int): The number of price paths to generate.
            sobol_seq_samples (numpy.ndarray): Optional Sobol sequence samples for QMC.

        Returns:
            numpy.ndarray: An array of shape (num_paths, num_steps+1, num_assets) representing the generated price paths.
        """
        chol_matrix = np.linalg.cholesky(self.correlation_matrix)
        dt = self.time_to_maturity / self.num_steps
        price_paths = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        price_paths[:, 0, :] = self.spot_prices

        if sobol_seq_samples is not None:
            random_normals = sobol_seq_samples
        else:
            random_normals = norm.rvs(size=(num_paths, self.num_steps, self.num_assets))

        for t in range(1, self.num_steps + 1):
            if sobol_seq_samples is not None:
                correlated_normals = np.dot(random_normals[:, t-1, :], chol_matrix.T)
            else:
                correlated_normals = np.dot(random_normals[:, t-1, :], chol_matrix.T)
            drifts = (self.risk_free_rate - self.dividend_yields - 0.5 * self.volatilities**2) * dt
            diffusions = self.volatilities * np.sqrt(dt) * correlated_normals
            price_paths[:, t, :] = price_paths[:, t-1, :] * np.exp(drifts + diffusions)

        return price_paths




def test_black_scholes_pricing(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,  # 1 year
    strike_price = 100,   # Example strike price
    option_type = 'call',   # Example option type
    num_steps = 252):  # Number of steps for finite difference method):

    # Instantiate the Black-Scholes model
    bs_model = BlackScholesModel(
        spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=risk_free_rate,
        dividend_yields=dividend_yields,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps
    )

    # Calculate the option price using Monte Carlo method
    option_price_mc = bs_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Option Price (Monte Carlo): {option_price_mc}")

    # Calculate the option price using Monte Carlo method
    option_price_qmc = bs_model.price(strike_price, option_type, method='QMC', num_paths=10000)
    print(f"Option Price (QMC): {option_price_qmc}")

    # Calculate the option price using Gaussian Quadrature method
    option_price_quad = bs_model.price(strike_price, option_type, method='quadrature', num_points=3000)
    print(f"Option Price (Quadrature): {option_price_quad}")

    # Calculate the option price using Finite Difference method
    #option_price_fd = bs_model.price(strike_price, option_type, method='finite_difference', num_steps=num_steps*2, num_grid_points=100)
    #print(f"Option Price (Finite Difference): {option_price_fd}")


def test_taylor_series_approximation(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    strike_price = 100,   # Example strike price
    option_type = 'call',   # Example option type
    new_spot_prices = np.array([102, 107]), # New spot prices for the test
    num_steps = 252, ): 

    # Instantiate the Black-Scholes model
    bs_model = BlackScholesModel(
        spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=risk_free_rate,
        dividend_yields=dividend_yields,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps
    )

    
    

    # Calculate the option price using Monte Carlo method as a reference
    reference_price = bs_model.price(strike_price, option_type, method='QMC', num_paths=10000)
    print(f"Reference Option Price (QMC): {reference_price}")

    # Calculate the option price using Taylor Series approximation
    approximated_price = bs_model.price(strike_price, option_type, method = "taylor_series", new_prices = new_spot_prices)
    print(f"Approximated Option Price (Taylor Series): {approximated_price}")


def test_moment_matching_approximation(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    strike_price = 100,  
    option_type = 'call', 
    num_steps = 252):  

    # Instantiate the Black-Scholes model
    bs_model = BlackScholesModel(
        spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=risk_free_rate,
        dividend_yields=dividend_yields,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps
    )

    # Calculate the option price using Monte Carlo method as a reference
    reference_price = bs_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Reference Option Price (Monte Carlo): {reference_price}")

    # Calculate the option price using Moment Matching approximation
    approximated_price = bs_model.price(strike_price, option_type, method='moment_matching')
    print(f"Approximated Option Price (Moment Matching): {approximated_price}")


def test_bessel_process_solution(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0, 
    strike_price = 100, 
    option_type = 'call', 
    num_steps = 252): 

    # Instantiate the Black-Scholes model
    bs_model = BlackScholesModel(
        spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=risk_free_rate,
        dividend_yields=dividend_yields,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps
    )

    # Calculate the option price using the Bessel process-based solution
    price = bs_model.price(strike_price, option_type, method="bessel")
    print(f"Option Price (Bessel Process): {price}")


# Test function to validate QMC method
def test_qmc_pricing(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0, 
    strike_price = 100, 
    option_type = 'call', 
    num_steps = 252):

    # Instantiate the Black-Scholes model
    bs_model = BlackScholesModel(spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        risk_free_rate=risk_free_rate,
        dividend_yields=dividend_yields,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps
    )

    # Calculate the option price using the QMC method
    price = bs_model.price(strike_price, option_type, method = "QMC")
    print(f"Option Price (QMC): {price}")

