"""
evaluation.py
-------------
Evaluación de los modelos de recomendación.

Funciones:
    evaluate_model()       – calcula RMSE y MAE sobre un testset de Surprise.
    cross_validate_model() – validación cruzada k-fold con Surprise.
"""

import numpy as np
import pandas as pd
from surprise import accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, KFold


def evaluate_model(model, testset: list, model_name: str = "Modelo") -> dict:
    """Calcula RMSE y MAE de un modelo Surprise sobre su testset.

    Parámetros
    ----------
    model : algoritmo de Surprise
        Modelo ya entrenado (SVD, KNNBasic, etc.).
    testset : list
        Lista de tuplas (uid, iid, r_ui) generada por Surprise.
    model_name : str
        Nombre del modelo para el mensaje de salida.

    Retorna
    -------
    dict con claves 'rmse' y 'mae'.

    Ejemplo
    -------
    >>> metrics = evaluate_model(svd_model, testset, "SVD")
    >>> print(metrics)
    {'rmse': 0.87, 'mae': 0.68}
    """
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    print(f"{model_name} → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "mae": mae}


def cross_validate_model(
    algorithm,
    ratings_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Validación cruzada k-fold usando Surprise.

    Parámetros
    ----------
    algorithm : algoritmo de Surprise
        Instancia del algoritmo (p. ej. SVD(n_factors=100)).
    ratings_df : pd.DataFrame
        DataFrame con columnas userId, movieId, rating.
    n_splits : int
        Número de folds (default 5).
    random_state : int
        Semilla para reproducibilidad (default 42).

    Retorna
    -------
    dict con 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std'.

    Ejemplo
    -------
    >>> results = cross_validate_model(SVD(), ratings_clean)
    >>> print(results)
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    cv_results = cross_validate(algorithm, data, measures=["RMSE", "MAE"], cv=kf, verbose=False)

    rmse_mean = float(np.mean(cv_results["test_rmse"]))
    rmse_std = float(np.std(cv_results["test_rmse"]))
    mae_mean = float(np.mean(cv_results["test_mae"]))
    mae_std = float(np.std(cv_results["test_mae"]))

    print(
        f"Validación cruzada ({n_splits}-fold) → "
        f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} | "
        f"MAE: {mae_mean:.4f} ± {mae_std:.4f}"
    )
    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std,
            "mae_mean": mae_mean, "mae_std": mae_std}
