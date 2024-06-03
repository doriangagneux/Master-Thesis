import numpy as np
from abc import ABC, abstractmethod
from basketoptions.models.base_model import BaseModel
from scipy.stats import norm
import matplotlib.pyplot as plt


class StochasticVolatilityModel(BaseModel, ABC):
    def __init__(self, spot_prices: np.ndarray, weights: np.ndarray, risk_free_rate: float, dividend_yields: np.ndarray, time_to_maturity: float, num_steps: int = 252):
        super().__init__(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, num_steps)

    @abstractmethod
    def generate_paths(self, num_paths: int) -> np.ndarray:
        pass



class HestonModel(StochasticVolatilityModel):
    def __init__(self, spot_prices: np.ndarray, weights: np.ndarray, initial_variances: np.ndarray, correlation_matrix: np.ndarray, risk_free_rate: float, dividend_yields: np.ndarray, time_to_maturity: float, kappa: np.ndarray, theta: np.ndarray, sigma: np.ndarray, rho: np.ndarray, num_steps: int = 252):
        super().__init__(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, num_steps)
        self.initial_variances = initial_variances
        self.correlation_matrix = correlation_matrix
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def generate_paths(self, num_paths: int) -> np.ndarray:
        dt = self.time_to_maturity / self.num_steps
        prices = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        variances = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        prices[:, 0, :] = self.spot_prices
        variances[:, 0, :] = self.initial_variances

        for t in range(1, self.num_steps + 1):
            z1 = norm.rvs(size=(num_paths, self.num_assets))
            z2 = norm.rvs(size=(num_paths, self.num_assets))
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            variances[:, t, :] = np.abs(variances[:, t-1, :] + self.kappa * (self.theta - variances[:, t-1, :]) * dt + self.sigma * np.sqrt(variances[:, t-1, :]) * np.sqrt(dt) * w1)
            prices[:, t, :] = prices[:, t-1, :] * np.exp((self.risk_free_rate - 0.5 * variances[:, t-1, :]) * dt + np.sqrt(variances[:, t-1, :]) * np.sqrt(dt) * w2)

        return prices, variances
    


class SABRModel(StochasticVolatilityModel):
    def __init__(self, spot_prices: np.ndarray, weights: np.ndarray, initial_volatilities: np.ndarray, correlation_matrix: np.ndarray, risk_free_rate: float, dividend_yields: np.ndarray, time_to_maturity: float, alpha: np.ndarray, beta: np.ndarray, rho: np.ndarray, nu: np.ndarray, num_steps: int = 252):
        super().__init__(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity)
        self.initial_volatilities = initial_volatilities
        self.correlation_matrix = correlation_matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.num_steps = num_steps

    def generate_paths(self, num_paths: int) -> np.ndarray:
        dt = self.time_to_maturity / self.num_steps
        prices = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        volatilities = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        prices[:, 0, :] = self.spot_prices
        volatilities[:, 0, :] = self.initial_volatilities

        for t in range(1, self.num_steps + 1):
            z1 = norm.rvs(size=(num_paths, self.num_assets))
            z2 = norm.rvs(size=(num_paths, self.num_assets))
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            volatilities[:, t, :] = np.abs(volatilities[:, t-1, :] + self.alpha * np.sqrt(dt) * w2 + 0.5 * self.nu**2 * (dt * (volatilities[:, t-1, :] ** 2)))
            prices[:, t, :] = prices[:, t-1, :] * np.exp((self.risk_free_rate - self.dividend_yields - 0.5 * volatilities[:, t-1, :]**2) * dt + volatilities[:, t-1, :] * np.sqrt(dt) * w1)

        return prices, volatilities
    



def test_heston_model(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    initial_variances = np.array([0.04, 0.05]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    kappa = np.array([2.0, 1.5]), 
    theta = np.array([0.04, 0.05]), 
    sigma = np.array([0.2, 0.25]), 
    rho = np.array([0.5, -0.5]), 
    num_steps = 252):
    

    # Instantiate the model
    heston_model = HestonModel(spot_prices, weights, initial_variances, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, kappa, theta, sigma, rho, num_steps)

    # Set the option parameters
    strike_price = 100  # Example strike price
    option_type = 'call'  # Example option type

    # Calculate the option price using Monte Carlo
    option_price_mc = heston_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Option Price (Monte Carlo): {option_price_mc}")

    # Calculate the option price using Quadrature
    option_price_quad = heston_model.price(strike_price, option_type, method='quadrature', num_points=5000)
    print(f"Option Price (Quadrature): {option_price_quad}")

    # Generate paths for visualization (optional)
    num_paths = 10  # Generate 10 paths for visualization
    paths = heston_model.generate_paths(num_paths)

    # Plot paths (optional)
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[2]):
        for j in range(num_paths):
            plt.plot(paths[j, :, i], label=f'Asset {i+1}' if j == 0 else "", alpha=0.6)
    plt.title('Heston Model Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def test_sabr_model(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    initial_volatilities = np.array([0.2, 0.25]), 
    correlation_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    alpha = np.array([0.1, 0.15]), 
    beta = np.array([0.5, 0.5]), 
    rho = np.array([0.5, -0.5]), 
    nu = np.array([0.2, 0.3]), 
    num_steps = 252):
    # Parameters for SABR Model with two underlying assets
    

    # Instantiate the model
    sabr_model = SABRModel(spot_prices, weights, initial_volatilities, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, alpha, beta, rho, nu, num_steps)

    # Set the option parameters
    strike_price = 100  # Example strike price
    option_type = 'call'  # Example option type

    # Calculate the option price using Monte Carlo
    option_price_mc = sabr_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Option Price (Monte Carlo): {option_price_mc}")

    # Calculate the option price using Quadrature
    option_price_quad = sabr_model.price(strike_price, option_type, method='quadrature', num_points=5000)
    print(f"Option Price (Quadrature): {option_price_quad}")

    # Generate paths for visualization (optional)
    num_paths = 10  # Generate 10 paths for visualization
    paths = sabr_model.generate_paths(num_paths)

    # Plot paths (optional)
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[2]):
        for j in range(num_paths):
            plt.plot(paths[j, :, i], label=f'Asset {i+1}' if j == 0 else "", alpha=0.6)
    plt.title('SABR Model Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Run the test
# test_sabr_model()
# test_heston_model()