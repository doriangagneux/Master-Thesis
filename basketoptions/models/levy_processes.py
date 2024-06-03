from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm, poisson, gamma
import matplotlib.pyplot as plt
from basketoptions.models.base_model import BaseModel


class LevyProcess(BaseModel):
    def __init__(self, spot_prices, weights, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, num_steps=252):
        super().__init__(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, num_steps)
        self.correlation_matrix = np.array(correlation_matrix)

    def generate_paths(self, num_paths: int) -> np.ndarray:
        pass

class JumpDiffusionProcess(LevyProcess):
    def __init__(self, spot_prices, weights, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, drift, volatilities, jump_intensity, jump_mean, jump_std, num_steps=252):
        super().__init__(spot_prices, weights, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, num_steps)
        self.drift = drift
        self.volatilities = volatilities
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def generate_paths(self, num_paths: int) -> np.ndarray:
        dt = self.time_to_maturity / self.num_steps
        paths = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        paths[:, 0, :] = self.spot_prices

        chol_matrix = np.linalg.cholesky(self.correlation_matrix)

        for t in range(1, self.num_steps + 1):
            z = norm.rvs(size=(num_paths, self.num_assets))
            jumps = poisson.rvs(self.jump_intensity * dt, size=(num_paths, self.num_assets))
            jump_sizes = np.zeros((num_paths, self.num_assets))
            for i in range(self.num_assets):
                jump_sizes[:, i] = np.sum(norm.rvs(self.jump_mean[i], self.jump_std[i], size=(num_paths, jumps[:, i].max())), axis=1)
            correlated_normals = np.dot(z, chol_matrix.T)
            paths[:, t, :] = paths[:, t-1, :] * np.exp((self.drift - 0.5 * self.volatilities**2) * dt + self.volatilities * np.sqrt(dt) * correlated_normals + jump_sizes)

        return paths

 
    
class VarianceGammaProcess(LevyProcess):
    def __init__(self, spot_prices, weights, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, volatilities, theta, nu, num_steps=252):
        super().__init__(spot_prices, weights, correlation_matrix, risk_free_rate, dividend_yields, time_to_maturity, num_steps)
        self.volatilities = volatilities
        self.theta = theta
        self.nu = nu

    def generate_paths(self, num_paths: int) -> np.ndarray:
        dt = self.time_to_maturity / self.num_steps
        paths = np.zeros((num_paths, self.num_steps + 1, self.num_assets))
        paths[:, 0, :] = self.spot_prices

        chol_matrix = np.linalg.cholesky(self.correlation_matrix)

        for t in range(1, self.num_steps + 1):
            z = norm.rvs(size=(num_paths, self.num_assets))
            gamma_increments = gamma.rvs(dt / self.nu, scale=self.nu, size=(num_paths, self.num_assets))
            correlated_normals = np.dot(z, chol_matrix.T)
            paths[:, t, :] = paths[:, t-1, :] * np.exp(self.theta * gamma_increments + self.volatilities * np.sqrt(gamma_increments) * correlated_normals)

        return paths


def test_jump_diffusion(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    drift = np.array([0.02, 0.03]), 
    volatility = np.array([0.2, 0.25]), 
    jump_intensity = np.array([0.5, 0.6]), 
    jump_mean = np.array([0.01, 0.02]), 
    jump_std = np.array([0.02, 0.03]), 
    num_steps = 252):
    

    # Instantiate the model
    jump_diffusion_model = JumpDiffusionProcess(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, drift, volatility, jump_intensity, jump_mean, jump_std, num_steps)

    # Set the option parameters
    strike_price = 100  # Example strike price
    option_type = 'call'  # Example option type

    # Calculate the option price using Monte Carlo
    option_price = jump_diffusion_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Option Price (Jump Diffusion): {option_price}")

    # Generate paths for visualization (optional)
    num_paths = 10  # Generate 10 paths for visualization
    paths = jump_diffusion_model.generate_paths(num_paths)

    # Plot paths (optional)
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[2]):
        for j in range(num_paths):
            plt.plot(paths[j, :, i], label=f'Asset {i+1}' if j == 0 else "", alpha=0.6)
    plt.title('Jump Diffusion Process Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def test_variance_gamma(spot_prices = np.array([100, 105]), 
    weights = np.array([0.5, 0.5]), 
    risk_free_rate = 0.05, 
    dividend_yields = np.array([0.02, 0.03]), 
    time_to_maturity = 1.0,   # 1 year
    volatility = np.array([0.2, 0.25]), 
    theta = np.array([0.1, 0.15]), 
    nu = np.array([0.2, 0.3]), 
    num_steps = 252):
    

    # Instantiate the model
    variance_gamma_model = VarianceGammaProcess(spot_prices, weights, risk_free_rate, dividend_yields, time_to_maturity, volatility, theta, nu, num_steps)

    # Set the option parameters
    strike_price = 100  # Example strike price
    option_type = 'call'  # Example option type

    # Calculate the option price using Monte Carlo
    option_price = variance_gamma_model.price(strike_price, option_type, method='monte_carlo', num_paths=10000)
    print(f"Option Price (Variance Gamma): {option_price}")

    # Generate paths for visualization (optional)
    num_paths = 10  # Generate 10 paths for visualization
    paths = variance_gamma_model.generate_paths(num_paths)

    # Plot paths (optional)
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[2]):
        for j in range(num_paths):
            plt.plot(paths[j, :, i], label=f'Asset {i+1}' if j == 0 else "", alpha=0.6)
    plt.title('Variance Gamma Process Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Run tests
#test_jump_diffusion()
#test_variance_gamma()