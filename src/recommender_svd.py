"""
recommender_svd.py
------------------
Sistema de recomendación basado en Singular Value Decomposition (SVD).
Utiliza la librería Surprise para el entrenamiento y la predicción.

Funciones:
    train_svd_model()        – entrena el modelo SVD.
    get_svd_recommendations()– genera recomendaciones para un usuario.
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split


def train_svd_model(
    ratings_df: pd.DataFrame,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Entrena un modelo SVD con la librería Surprise.

    Parámetros
    ----------
    ratings_df : pd.DataFrame
        DataFrame con columnas userId, movieId, rating.
    n_factors : int
        Número de factores latentes (default 100).
    n_epochs : int
        Épocas de entrenamiento SGD (default 20).
    lr_all : float
        Tasa de aprendizaje (default 0.005).
    reg_all : float
        Regularización (default 0.02).
    test_size : float
        Proporción del conjunto de prueba (default 0.2).
    random_state : int
        Semilla para reproducibilidad (default 42).

    Retorna
    -------
    (svd_model, trainset, testset) : tuple
        - svd_model : modelo SVD entrenado.
        - trainset  : conjunto de entrenamiento de Surprise.
        - testset   : lista de tuplas para evaluación.

    Ejemplo
    -------
    >>> svd, trainset, testset = train_svd_model(ratings_clean)
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    trainset, testset = surprise_split(data, test_size=test_size, random_state=random_state)

    svd_model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    svd_model.fit(trainset)

    print(f"SVD entrenado | factores={n_factors} | épocas={n_epochs}")
    return svd_model, trainset, testset


def get_svd_recommendations(
    user_id: int,
    svd_model: SVD,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_recommendations: int = 10,
) -> pd.DataFrame:
    """Genera las top-N recomendaciones para un usuario usando SVD.

    Predice la valoración de todas las películas no vistas y devuelve
    las de mayor puntuación estimada.

    Parámetros
    ----------
    user_id : int
        ID del usuario objetivo.
    svd_model : SVD
        Modelo entrenado con train_svd_model.
    ratings_df : pd.DataFrame
        DataFrame de ratings (para identificar películas ya vistas).
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
    >>> recs = get_svd_recommendations(user_id=1, svd_model=svd, ...)
    >>> print(recs)
    """
    if user_id not in ratings_df["userId"].unique():
        raise ValueError(f"El usuario {user_id} no existe en el dataset.")

    seen = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])

    predictions = []
    for movie_id in movies_df["movieId"]:
        if movie_id not in seen:
            pred = svd_model.predict(str(user_id), str(movie_id))
            predictions.append((movie_id, pred.est))

    recs = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
    recs = recs.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
    recs = recs.sort_values("predicted_rating", ascending=False).head(n_recommendations)
    return recs[["movieId", "title", "predicted_rating"]].reset_index(drop=True)
