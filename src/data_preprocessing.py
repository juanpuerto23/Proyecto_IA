"""
data_preprocessing.py
---------------------
Carga, limpieza y transformación del dataset MovieLens.

Funciones:
    load_data()              – carga los CSV de ratings y películas.
    clean_data()             – elimina duplicados, nulos y outliers.
    create_user_item_matrix()– construye la matriz usuario x película.
"""

import os
import pandas as pd
import numpy as np

# Rutas por defecto (relativas a este archivo)
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")


def load_data(
    ratings_path: str = os.path.join(_DATA_DIR, "ratings.csv"),
    movies_path: str = os.path.join(_DATA_DIR, "movies.csv"),
) -> tuple:
    """Carga los archivos CSV del dataset MovieLens.

    Parámetros
    ----------
    ratings_path : str
        Ruta al archivo ratings.csv (columnas: userId, movieId, rating, timestamp).
    movies_path : str
        Ruta al archivo movies.csv (columnas: movieId, title, genres).

    Retorna
    -------
    (ratings_df, movies_df) : tuple de DataFrames

    Ejemplo
    -------
    >>> ratings, movies = load_data()
    >>> print(ratings.shape)
    """
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"No se encontró: {ratings_path}\n"
            "Descarga MovieLens 100K desde https://grouplens.org/datasets/movielens/100k/"
            " y coloca los CSV en la carpeta data/."
        )
    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"No se encontró: {movies_path}\n"
            "Descarga MovieLens 100K desde https://grouplens.org/datasets/movielens/100k/"
            " y coloca los CSV en la carpeta data/."
        )

    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    print(f"Ratings cargados  : {len(ratings_df):,} filas")
    print(f"Películas cargadas: {len(movies_df):,} filas")

    return ratings_df, movies_df


def clean_data(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings_user: int = 20,
    min_ratings_movie: int = 10,
) -> tuple:
    """Limpia los DataFrames eliminando nulos, duplicados y usuarios/películas
    con pocas valoraciones.

    Parámetros
    ----------
    ratings_df : pd.DataFrame
        DataFrame de valoraciones.
    movies_df : pd.DataFrame
        DataFrame de películas.
    min_ratings_user : int
        Número mínimo de ratings que debe tener un usuario (default 20).
    min_ratings_movie : int
        Número mínimo de ratings que debe tener una película (default 10).

    Retorna
    -------
    (ratings_clean, movies_clean) : tuple de DataFrames limpios

    Ejemplo
    -------
    >>> ratings_c, movies_c = clean_data(ratings, movies)
    """
    # Eliminar nulos y duplicados
    ratings_df = ratings_df.dropna().drop_duplicates()
    movies_df = movies_df.dropna().drop_duplicates()

    # Filtrar ratings fuera del rango válido [0.5, 5.0]
    ratings_df = ratings_df[ratings_df["rating"].between(0.5, 5.0)]

    # Mantener solo usuarios activos
    active_users = (
        ratings_df["userId"].value_counts()[
            lambda s: s >= min_ratings_user
        ].index
    )
    ratings_df = ratings_df[ratings_df["userId"].isin(active_users)]

    # Mantener solo películas con suficientes valoraciones
    active_movies = (
        ratings_df["movieId"].value_counts()[
            lambda s: s >= min_ratings_movie
        ].index
    )
    ratings_df = ratings_df[ratings_df["movieId"].isin(active_movies)]

    # Sincronizar películas
    movies_df = movies_df[movies_df["movieId"].isin(active_movies)]

    ratings_df = ratings_df.reset_index(drop=True)
    movies_df = movies_df.reset_index(drop=True)

    print(
        f"Tras limpieza → {len(ratings_df):,} ratings | "
        f"{ratings_df['userId'].nunique():,} usuarios | "
        f"{ratings_df['movieId'].nunique():,} películas"
    )
    return ratings_df, movies_df


def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Crea la matriz usuario-película (filas = usuarios, columnas = películas).

    Las celdas sin valoración quedan como NaN (matriz dispersa).

    Parámetros
    ----------
    ratings_df : pd.DataFrame
        DataFrame de valoraciones limpio.

    Retorna
    -------
    pd.DataFrame
        Matriz de forma (n_usuarios, n_películas).

    Ejemplo
    -------
    >>> matrix = create_user_item_matrix(ratings_clean)
    >>> print(matrix.shape)
    """
    matrix = ratings_df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    ).astype(np.float32)

    density = matrix.notna().sum().sum() / matrix.size * 100
    print(f"Matriz: {matrix.shape} | Densidad: {density:.2f}%")

    return matrix
