import numpy as np

def finite_difference_multidim_bs_pricing(model, strike_price, option_type='call', num_steps=100, num_grid_points=50):
    # Transformations and grid setup
    S_min = np.array([model.spot_prices[i] * 0.5 for i in range(model.num_assets)])
    S_max = np.array([model.spot_prices[i] * 1.5 for i in range(model.num_assets)])
    h = (S_max - S_min) / (num_grid_points - 1)
    tau_final = model.time_to_maturity
    dt = tau_final / num_steps

    grid_points = np.meshgrid(*[np.linspace(S_min[i], S_max[i], num_grid_points) for i in range(model.num_assets)])
    grid_points = np.array(grid_points)
    option_values = np.zeros_like(grid_points[0])

    # Initial condition
    if option_type == 'call':
        payoff = lambda x: np.maximum(np.dot(x.T, model.weights) - strike_price, 0)
    elif option_type == 'put':
        payoff = lambda x: np.maximum(strike_price - np.dot(x.T, model.weights), 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Calcul du payoff initial sur la grille
    option_values = payoff(grid_points)

    # Smoothing the initial condition
    option_values = smooth_initial_condition(option_values, h)

    # Time stepping
    for _ in range(num_steps):
        option_values = time_step(model, option_values, h, dt)

    # Interpolate the option value at the spot prices
    index = tuple(np.argmin(np.abs(grid_points[i] - model.spot_prices[i])) for i in range(model.num_assets))
    return option_values

def time_step(model, option_values, h, dt):
    num_assets = model.num_assets
    num_grid_points = option_values.shape[0]
    option_values_next = np.zeros_like(option_values)

    # Coefficients
    gamma = 0.25
    rho = model.correlation_matrix[0, 1]
    sigma_1, sigma_2 = model.volatilities
    r = model.risk_free_rate

    K_coeff = lambda i1, i2: (
        -2 * gamma**2 * rho**2 / (3 * h[0] * h[1]) + 5 * gamma**2 / (3 * h[0]**2) + ((sigma_1/2 - r/sigma_1)**2 + (sigma_2/2 - r/sigma_2)**2) / 3
        if i1 == i2 == 0 else
        gamma**2 * rho**2 / (3 * h[0] * h[1]) + (gamma * (sigma_1/2 - r/sigma_1) / (3 * h[0]) if i1 != 0 else 0) - gamma * (sigma_2/2 - r/sigma_2) * rho / (3 * h[1]) - (sigma_1/2 - r/sigma_1)**2 / 6 - gamma**2 / (3 * h[0]**2)
        if i1 != 0 and i2 == 0 else
        gamma**2 * rho**2 / (3 * h[0] * h[1]) + (gamma * (sigma_2/2 - r/sigma_2) / (3 * h[1]) if i2 != 0 else 0) - gamma * (sigma_1/2 - r/sigma_1) * rho / (3 * h[0]) - (sigma_2/2 - r/sigma_2)**2 / 6 - gamma**2 / (3 * h[1]**2)
        if i1 == 0 and i2 != 0 else
        (sigma_2/2 - r/sigma_2) * (sigma_1/2 - r/sigma_1) / 12 - gamma * (sigma_2/2 - r/sigma_2) / (12 * h[1]) - gamma * (sigma_1/2 - r/sigma_1) / (12 * h[0]) - gamma * (sigma_1/2 - r/sigma_1) * rho / (6 * h[1]) - gamma * (sigma_2/2 - r/sigma_2) * rho / (6 * h[0]) - gamma**2 / (12 * h[0] * h[1]) - gamma**2 * rho**2 / (6 * h[0] * h[1])
        if i1 == -1 and i2 == 1 else
        gamma * (sigma_2/2 - r/sigma_2) / (12 * h[1]) - (sigma_2/2 - r/sigma_2) * (sigma_1/2 - r/sigma_1) / 12 - gamma * (sigma_1/2 - r/sigma_1) / (12 * h[0]) + gamma * (sigma_1/2 - r/sigma_1) * rho / (6 * h[1]) - gamma * (sigma_2/2 - r/sigma_2) * rho / (6 * h[0]) - gamma**2 / (12 * h[0] * h[1]) - gamma**2 * rho**2 / (6 * h[0] * h[1])
        if i1 == 1 and i2 == -1 else 0
    )

    M_coeff = lambda i1, i2: (
        2/3 if i1 == i2 == 0 else
        rho / 24 if (i1 == 1 and i2 == -1) or (i1 == -1 and i2 == 1) else
        1/12 - h[0] * (sigma_1/2 - r/sigma_1) / (12 * gamma) if i1 != 0 and i2 == 0 else
        1/12 - h[1] * (sigma_2/2 - r/sigma_2) / (12 * gamma) if i1 == 0 and i2 != 0 else 0
    )

    # Time stepping
    for i in range(1, num_grid_points - 1):
        for j in range(1, num_grid_points - 1):
            stencil_values = option_values[i-1:i+2, j-1:j+2]
            stencil_coeffs_K = np.array([[K_coeff(i1, i2) for i2 in range(-1, 2)] for i1 in range(-1, 2)])
            stencil_coeffs_M = np.array([[M_coeff(i1, i2) for i2 in range(-1, 2)] for i1 in range(-1, 2)])

            option_values_next[i, j] = (
                np.sum(stencil_coeffs_M * stencil_values) -
                0.5 * dt * np.sum(stencil_coeffs_K * stencil_values)
            ) / (1 - 0.5 * dt * stencil_coeffs_K[1, 1])

    return option_values_next


def smooth_initial_condition(option_values, h):
    num_grid_points = option_values.shape[0]
    smoothed_values = np.zeros_like(option_values)

    # Fourier transform of the smoothing operator
    def phi_4_hat(omega):
        return (np.sin(omega/2) / (omega/2))**4 * (1 + 2/3 * np.sin(omega/2)**2)

    # Smoothing in each direction
    for i in range(num_grid_points):
        for j in range(num_grid_points):
            x, y = i * h[0], j * h[1]
            integral_x = lambda x_hat: phi_4_hat(x_hat/h[0]) * phi_4_hat(y/h[1]) * option_values[max(0, min(i - int(x_hat/h[0]), num_grid_points - 1)), j]
            integral_y = lambda y_hat: phi_4_hat(x/h[0]) * phi_4_hat(y_hat/h[1]) * option_values[i, max(0, min(j - int(y_hat/h[1]), num_grid_points - 1))]
            smoothed_values[i, j] = 1/h[0] * 1/h[1] * (integral_x(-3*h[0]) + integral_x(3*h[0])) * (integral_y(-3*h[1]) + integral_y(3*h[1]))

    return smoothed_values