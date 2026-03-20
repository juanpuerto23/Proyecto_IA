"""
recommender_knn.py
------------------
Sistema de recomendación basado en K-Nearest Neighbors (KNN).
Utiliza filtrado colaborativo basado en usuarios.

Funciones:
    train_knn_model()        – entrena el modelo KNN.
    get_knn_recommendations()– genera recomendaciones para un usuario.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer


def train_knn_model(
    user_item_matrix: pd.DataFrame,
    n_neighbors: int = 20,
    metric: str = "cosine",
) -> tuple:
    """Entrena un modelo KNN sobre la matriz usuario-película.

    Los valores NaN se imputan con la media de cada película
    antes del entrenamiento.

    Parámetros
    ----------
    user_item_matrix : pd.DataFrame
        Matriz usuario-película (salida de create_user_item_matrix).
    n_neighbors : int
        Número de vecinos más cercanos (default 20).
    metric : str
        Métrica de distancia: 'cosine', 'euclidean' (default 'cosine').

    Retorna
    -------
    (knn_model, matrix_imputed, imputer) : tuple
        - knn_model      : modelo NearestNeighbors entrenado.
        - matrix_imputed : array numpy con NaN imputados.
        - imputer        : objeto SimpleImputer ajustado.

    Ejemplo
    -------
    >>> knn, mat, imp = train_knn_model(matrix)
    """
    imputer = SimpleImputer(strategy="mean")
    matrix_imputed = imputer.fit_transform(user_item_matrix.values)

    # +1 porque kneighbors incluye al propio usuario
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        algorithm="brute",
    )
    knn_model.fit(matrix_imputed)

    print(f"KNN entrenado | vecinos={n_neighbors} | métrica={metric}")
    return knn_model, matrix_imputed, imputer


def get_knn_recommendations(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    knn_model: NearestNeighbors,
    matrix_imputed: np.ndarray,
    movies_df: pd.DataFrame,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    """Genera las top-N recomendaciones de películas para un usuario usando KNN.

    La valoración predicha se calcula como el promedio ponderado de los
    ratings de los vecinos más cercanos (peso = 1 - distancia coseno).

    Parámetros
    ----------
    user_id : int
        ID del usuario objetivo.
    user_item_matrix : pd.DataFrame
        Matriz usuario-película original.
    knn_model : NearestNeighbors
        Modelo entrenado con train_knn_model.
    matrix_imputed : np.ndarray
        Matriz con NaN imputados (salida de train_knn_model).
    movies_df : pd.DataFrame
        DataFrame con columnas movieId y title.
    n_recommendations : int
        Número de películas a recomendar (default 10).

    Retorna
    -------
    pd.DataFrame
        Columnas: movieId, title, predicted_rating (ordenado desc).

    Ejemplo
    -------
    >>> recs = get_knn_recommendations(user_id=1, ...)
    >>> print(recs)
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"El usuario {user_id} no está en la matriz.")

    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = matrix_imputed[user_idx].reshape(1, -1)

    distances, indices = knn_model.kneighbors(user_vector)
    # Excluir al propio usuario (primer resultado)
    neighbor_distances = distances[0][1:]
    neighbor_indices = indices[0][1:]

    # Películas ya vistas por el usuario
    seen = set(user_item_matrix.columns[user_item_matrix.loc[user_id].notna()])

    predictions = {}
    for col_idx, movie_id in enumerate(user_item_matrix.columns):
        if movie_id in seen:
            continue
        weights, ratings = [], []
        for dist, n_idx in zip(neighbor_distances, neighbor_indices):
            r = user_item_matrix.iloc[n_idx, col_idx]
            if pd.notna(r):
                weights.append(1.0 - dist)
                ratings.append(r)
        if weights:
            predictions[movie_id] = float(np.average(ratings, weights=weights))

    if not predictions:
        return pd.DataFrame(columns=["movieId", "title", "predicted_rating"])

    recs = pd.DataFrame(
        predictions.items(), columns=["movieId", "predicted_rating"]
    )
    recs = recs.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
    recs = recs.sort_values("predicted_rating", ascending=False).head(n_recommendations)
    return recs[["movieId", "title", "predicted_rating"]].reset_index(drop=True)
