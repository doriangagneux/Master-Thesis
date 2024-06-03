from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict
import numpy as np
from basketoptions.pricing.monte_carlo import monte_carlo_pricing, qmc_pricing
from basketoptions.pricing.quadrature import gaussian_quadrature_pricing
from basketoptions.pricing.analytical_approximations import taylor_series_approximation, moment_matching_approximation




class BaseModel(ABC):
    """
    An abstract base class for all pricing models.
    """

    @abstractmethod
    def __init__(self, spot_prices: Union[float, np.ndarray], weights: np.ndarray,
                 risk_free_rate: float, dividend_yields: Union[float, np.ndarray],
                 time_to_maturity: float, num_steps: int):
        """
        Initializes the base model with common parameters.

        :param spot_prices: The current spot prices of the underlying assets.
        :param weights: The weights of the assets in the basket.
        :param risk_free_rate: The risk-free interest rate.
        :param dividend_yields: The dividend yields of the underlying assets.
        :param time_to_maturity: The time to maturity of the option (in years).
        """
        self.spot_prices = np.array(spot_prices) if isinstance(spot_prices, list) else spot_prices
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = np.array(dividend_yields) if isinstance(dividend_yields, list) else dividend_yields
        self.time_to_maturity = time_to_maturity
        self.num_assets = len(self.spot_prices)
        self.num_steps = num_steps
        self.greekvalues = None # will contain greek values as soon as greeks method as been run one time

        if len(self.spot_prices) != len(self.weights):
            raise ValueError("The number of spot prices must match the number of weights.")
        if isinstance(self.dividend_yields, np.ndarray) and len(self.dividend_yields) != len(self.spot_prices):
            raise ValueError("The number of dividend yields must match the number of spot prices.")

    @abstractmethod
    def generate_paths(self, num_paths: int) -> np.ndarray:
        pass

    """
    def monte_carlo_simulation(self, num_paths: int, strike_price: float, option_type: str, price_paths: np.ndarray) -> float:
        basket_prices = np.dot(price_paths[:, -1, :], self.weights)

        if option_type == 'call':
            payoffs = np.maximum(basket_prices - strike_price, 0)
        elif option_type == 'put':
            payoffs = np.maximum(strike_price - basket_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_payoffs = np.exp(-self.risk_free_rate * self.time_to_maturity) * payoffs
        return np.mean(discounted_payoffs)
    """

    """
    def price(self, strike_price: float, option_type: str = 'call') -> float:
        #num_paths = 10000  # Number of price paths to simulate
        #price_paths = self.generate_paths(num_paths)
        #return self.monte_carlo_simulation(num_paths, strike_price, option_type, price_paths)
    """

    def price(self, strike_price: float, option_type: str = 'call', new_prices: np.ndarray = None, method='QMC', **kwargs) -> float:
        if method == 'monte_carlo':
            return monte_carlo_pricing(self, strike_price, option_type, **kwargs)
        elif method == 'quadrature':
            return gaussian_quadrature_pricing(self, strike_price, option_type, **kwargs)
        elif method == 'taylor_series':
            if new_prices.all:
                return taylor_series_approximation(self, strike_price, option_type, new_prices, **kwargs)
            else:
                raise ValueError(f"Taylor series method needs new spot prices to calculate the option value on")
        elif method == 'moment_matching':
            return moment_matching_approximation(self, strike_price, option_type)
        elif method == "QMC":
            return qmc_pricing(self, strike_price, option_type, **kwargs)
        else:
            raise ValueError(f"Unknown pricing method: {method}")
        
    
    def deltas(self, strike_price: float, option_type: str = 'call'):

        epsilon = 1  # Small change for finite difference

        # Base price
        base_price = self.price(strike_price, option_type, method = 'QMC')

        # Delta
        deltas = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            self.spot_prices[i] += epsilon
            new_price = self.price(strike_price, option_type, method = 'QMC')
            deltas[i] = (new_price - base_price) / epsilon
            self.spot_prices[i] -= epsilon   

        return deltas



    def greeks(self, strike_price: float, option_type: str = 'call') -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculates the Greeks (delta, gamma, vega, theta, rho) for the basket option.

        Parameters:
            strike_price (float): The strike price of the option.
            option_type (str): The type of the option ('call' or 'put'). Default is 'call'.

        Returns:
            dict: A dictionary containing the values of the Greeks.
        """
        epsilon = 1  # Small change for finite difference

        # Base price
        base_price = self.price(strike_price, option_type, method = 'QMC')

        # Delta
        deltas = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            self.spot_prices[i] += epsilon
            new_price = self.price(strike_price, option_type, method = 'QMC')
            deltas[i] = (new_price - base_price) / epsilon
            self.spot_prices[i] -= epsilon            

        # Gamma
        gammas = np.zeros((self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i == j:
                    self.spot_prices[i] += epsilon
                    price_up = self.price(strike_price, option_type, method = 'QMC')
                    self.spot_prices[i] -= 2 * epsilon
                    price_down = self.price(strike_price, option_type, method = 'QMC')
                    self.spot_prices[i] += epsilon
                    gammas[i, j] = (price_up - 2 * base_price + price_down) / (epsilon**2)
                else:
                    self.spot_prices[i] += epsilon
                    price_up_i = self.price(strike_price, option_type, method = 'QMC')
                    self.spot_prices[i] -= epsilon
                    self.spot_prices[j] += epsilon
                    price_up_j = self.price(strike_price, option_type, method = 'QMC')
                    self.spot_prices[i] += epsilon
                    price_up_ij = self.price(strike_price, option_type, method = 'QMC')
                    self.spot_prices[i] -= epsilon
                    self.spot_prices[j] -= epsilon
                    gammas[i, j] = (price_up_ij - price_up_i - price_up_j + base_price) / (epsilon**2)

        # Vega
        vegas = np.zeros(self.num_assets)
        epsilon = 0.005
        for i in range(self.num_assets):
            self.volatilities[i] += epsilon
            new_price = self.price(strike_price, option_type)
            vegas[i] = (new_price - base_price) / epsilon
            self.volatilities[i] -= epsilon

        # Theta
        epsilon = 1
        self.time_to_maturity -= epsilon
        new_price = self.price(strike_price, option_type)
        theta = (new_price - base_price) / epsilon
        self.time_to_maturity += epsilon

        # Rho
        epsilon = 0.005
        self.risk_free_rate += epsilon
        new_price = self.price(strike_price, option_type)
        rho = (new_price - base_price) / epsilon
        self.risk_free_rate -= epsilon

        self.greekvalues = {
            'delta': deltas,
            'gamma': gammas,
            'vega': vegas,
            'theta': theta,
            'rho': rho
        }


