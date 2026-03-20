"""
utils.py
--------
Funciones auxiliares del proyecto.

    save_model()           – guarda un modelo en disco (pickle).
    load_model()           – carga un modelo desde disco.
    plot_rmse_comparison() – gráfico de barras RMSE/MAE: KNN vs SVD.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Sin pantalla (compatible con servidores y CI)
import matplotlib.pyplot as plt


_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")
_REPORTS_DIR = os.path.join(_BASE_DIR, "reports", "figures")


# ── Persistencia ────────────────────────────────────────────────────────────

def save_model(model, filename: str, models_dir: str = _MODELS_DIR) -> str:
    """Guarda un modelo entrenado en disco usando pickle.

    Parámetros
    ----------
    model : objeto del modelo
        Modelo SKLearn, Surprise, etc.
    filename : str
        Nombre del archivo (p. ej. 'knn_model.pkl').
    models_dir : str
        Directorio de destino (default: models/).

    Retorna
    -------
    str : ruta completa del archivo guardado.

    Ejemplo
    -------
    >>> save_model(knn, "knn_model.pkl")
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {path}")
    return path


def load_model(filename: str, models_dir: str = _MODELS_DIR):
    """Carga un modelo desde disco.

    Parámetros
    ----------
    filename : str
        Nombre del archivo (p. ej. 'knn_model.pkl').
    models_dir : str
        Directorio de origen (default: models/).

    Retorna
    -------
    Objeto del modelo deserializado.

    Ejemplo
    -------
    >>> knn = load_model("knn_model.pkl")
    """
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde: {path}")
    return model


# ── Visualización ────────────────────────────────────────────────────────────

def plot_rmse_comparison(
    results: dict,
    output_path: str | None = None,
) -> None:
    """Genera un gráfico de barras comparando RMSE y MAE de los modelos.

    Parámetros
    ----------
    results : dict
        Formato: {"KNN": {"rmse": 0.95, "mae": 0.74},
                  "SVD": {"rmse": 0.87, "mae": 0.68}}
    output_path : str | None
        Ruta de la imagen PNG. Si es None se guarda en reports/figures/.

    Ejemplo
    -------
    >>> plot_rmse_comparison({"KNN": {"rmse": 0.95, "mae": 0.74},
    ...                       "SVD": {"rmse": 0.87, "mae": 0.68}})
    """
    models = list(results.keys())
    rmse_vals = [results[m]["rmse"] for m in models]
    mae_vals = [results[m].get("mae", 0) for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - w / 2, rmse_vals, w, label="RMSE", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, mae_vals, w, label="MAE", color="#DD8452")

    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Comparación de modelos: KNN vs SVD", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if output_path is None:
        os.makedirs(_REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(_REPORTS_DIR, "rmse_comparison.png")

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en: {output_path}")
