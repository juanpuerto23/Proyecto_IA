# 🎬 Sistema de Recomendación de Películas

Sistema de recomendación de películas en Python usando **filtrado colaborativo** con dos enfoques:

- **KNN** (K-Nearest Neighbors) — basado en usuarios similares.
- **SVD** (Singular Value Decomposition) — factorización de matrices con Surprise.

---

## Descripción

El proyecto predice qué películas le gustarán a un usuario basándose en las valoraciones de usuarios similares.  
Se comparan ambos modelos usando las métricas **RMSE** y **MAE** para determinar cuál ofrece predicciones más precisas.

---

## Tecnologías usadas

| Librería | Uso |
|---|---|
| `pandas` / `numpy` | Carga y manipulación de datos |
| `scikit-learn` | Modelo KNN e imputación |
| `scikit-surprise` | Modelo SVD y validación cruzada |
| `matplotlib` | Visualización de resultados |
| `jupyter` | Notebook interactivo |

---

## Dataset

**MovieLens 100K** — 100 000 valoraciones de 943 usuarios sobre 1 682 películas.

1. Descarga desde: https://grouplens.org/datasets/movielens/100k/
2. Coloca `ratings.csv` y `movies.csv` en la carpeta `data/`.

---

## Estructura del proyecto

```
Proyecto_IA/
├── data/               # Archivos CSV del dataset
├── notebooks/
│   └── movie_recommendation.ipynb   # Notebook principal
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Carga y limpieza de datos
│   ├── recommender_knn.py      # Modelo KNN
│   ├── recommender_svd.py      # Modelo SVD
│   ├── evaluation.py           # RMSE, MAE, validación cruzada
│   └── utils.py                # Guardar/cargar modelos, gráficos
├── models/             # Modelos entrenados (.pkl)
├── reports/figures/    # Gráficos generados
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/juanpuerto23/Proyecto_IA.git
cd Proyecto_IA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset

Descarga **MovieLens 100K** y coloca `ratings.csv` y `movies.csv` en `data/`.

### 4. Abrir el notebook

```bash
jupyter notebook notebooks/movie_recommendation.ipynb
```

O importa el notebook en **Google Colab** directamente.

### 5. Uso desde Python

```python
from src.data_preprocessing import load_data, clean_data, create_user_item_matrix
from src.recommender_knn import train_knn_model, get_knn_recommendations
from src.recommender_svd import train_svd_model, get_svd_recommendations
from src.evaluation import evaluate_model

# Cargar y limpiar datos
ratings, movies = load_data()
ratings_clean, movies_clean = clean_data(ratings, movies)

# Modelo KNN
matrix = create_user_item_matrix(ratings_clean)
knn, mat_imp, imp = train_knn_model(matrix)
recs_knn = get_knn_recommendations(user_id=1, user_item_matrix=matrix,
                                    knn_model=knn, matrix_imputed=mat_imp,
                                    movies_df=movies_clean)
print(recs_knn)

# Modelo SVD
svd, trainset, testset = train_svd_model(ratings_clean)
metrics = evaluate_model(svd, testset, "SVD")
recs_svd = get_svd_recommendations(user_id=1, svd_model=svd,
                                    ratings_df=ratings_clean, movies_df=movies_clean)
print(recs_svd)
```

---

## Resultados

| Modelo | RMSE  | MAE   |
|--------|-------|-------|
| KNN    | ~0.95 | ~0.74 |
| SVD    | ~0.87 | ~0.68 |

SVD obtiene menor error de predicción. KNN es más interpretable.

---

## Autor

**Juan Puerto** — Proyecto universitario de Inteligencia Artificial.
