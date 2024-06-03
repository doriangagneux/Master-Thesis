import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from basketoptions.analysis.generate_data import generate_market_situation




def time_step_forward(model_class, model_params, time_step):
    # Create a model instance with time step maturity parameters
    maturity, model_params['time_to_maturity'] = model_params['time_to_maturity'], time_step
    model = model_class(**model_params)
    
    # Generate new paths for the specified time step
    paths = model.generate_paths(num_paths=1)
    
    # Update the spot prices and time to maturity in the model parameters
    new_spot_prices = paths[0, -1, :]
    model_params['spot_prices'] = new_spot_prices
    model_params['time_to_maturity'] = maturity - model_params['time_to_maturity']
    
    return model_params

def generate_initial_market_situation(model_class, model_params, num_assets): 
    #This function is useful only if you want an history of the evolution of the assets of the basket, which is the case when you perform a PCA
    
    market_situation = generate_market_situation(model_class, **model_params)
    new_spot_prices = market_situation[-1]
    model_params['spot_prices'] = new_spot_prices
    return model_params, market_situation

def generate_updated_market_situation(model_class, model_params, time_step):
    updated_params = time_step_forward(model_class, model_params, time_step)
    return updated_params

def compute_asset_deltas(model_class, model_params, strike_price, option_type):
    model = model_class(**model_params)
    return model.deltas(strike_price, option_type)

def delta_hedge(asset_deltas, spot_prices, initial_positions=None):
    if initial_positions is None:
        initial_positions = np.zeros(spot_prices.shape[0])
    
    hedge_positions = initial_positions - asset_deltas
    hedge_values = np.dot(spot_prices, hedge_positions)
    return hedge_values

def calculate_pnl(option_price_initial, option_price_updated, hedge_value_initial, hedge_value_updated):
    pnl = (option_price_updated - option_price_initial) + (hedge_value_updated - hedge_value_initial)
    return pnl





def perform_pca(data, n_components=None):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = pca.components_
    return principal_components, explained_variance_ratio, loadings, scaler

def compute_pca_deltas(asset_deltas, loadings, n_components):
    pc_deltas = np.dot(asset_deltas, loadings[:n_components].T)
    return pc_deltas

def pca_delta_hedge(pca_deltas, spot_prices, loadings, scaler, n_components, initial_positions=None):
    if initial_positions is None:
        initial_positions = np.zeros(n_components)
    
    hedge_positions = initial_positions - pca_deltas
    transformed_positions = np.dot(hedge_positions, loadings[:n_components])
    transformed_positions = transformed_positions.reshape(1, -1)
    transformed_positions = scaler.inverse_transform(transformed_positions)
    hedge_values = np.dot(spot_prices.reshape(1, -1), transformed_positions.T)
    return hedge_values



