"""
Paquete src – Sistema de Recomendación de Películas.

Importa las funciones principales para usarlas directamente:
    from src import load_data, train_knn_model, evaluate_model
"""

from .data_preprocessing import load_data, clean_data, create_user_item_matrix
from .recommender_knn import train_knn_model, get_knn_recommendations
from .recommender_svd import train_svd_model, get_svd_recommendations
from .evaluation import evaluate_model, cross_validate_model
from .utils import save_model, load_model, plot_rmse_comparison

__all__ = [
    "load_data",
    "clean_data",
    "create_user_item_matrix",
    "train_knn_model",
    "get_knn_recommendations",
    "train_svd_model",
    "get_svd_recommendations",
    "evaluate_model",
    "cross_validate_model",
    "save_model",
    "load_model",
    "plot_rmse_comparison",
]
