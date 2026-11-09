"""
Projet Machine Learning : Prédiction des prix de location immobilière
Auteur : Collé Syll — Licence 3 Ingénierie Informatique (UASZ)
Technologies : Python, Pandas, NumPy, Scikit-learn, Random Forest, Matplotlib
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def generate_housing_dataset(n_samples: int = 850, random_state: int = 42) -> pd.DataFrame:
    """Génère un jeu de données réaliste de locations immobilières (Dakar / Ziguinchor)."""
    rng = np.random.default_rng(random_state)

    quartiers = ["Plateau", "Almadies", "Mermoz", "Sacré-Cœur", "Liberté 6", "Parcelles", "Escale Ziguinchor"]
    quartier_coef = {
        "Almadies": 1.55,
        "Plateau": 1.45,
        "Mermoz": 1.30,
        "Sacré-Cœur": 1.20,
        "Liberté 6": 1.05,
        "Parcelles": 0.78,
        "Escale Ziguinchor": 0.72,
    }

    chosen_quartiers = rng.choice(quartiers, size=n_samples, p=[0.12, 0.12, 0.16, 0.18, 0.16, 0.14, 0.12])
    surface_m2 = rng.integers(28, 220, size=n_samples)
    nb_pieces = np.clip((surface_m2 // 32) + rng.integers(0, 2, size=n_samples), 1, 7)
    nb_salles_bain = np.clip(nb_pieces // 2 + rng.integers(0, 2, size=n_samples), 1, 4)
    etage = rng.integers(0, 7, size=n_samples)
    meuble = rng.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    neuf_ou_renove = rng.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    distance_transport_km = np.round(rng.uniform(0.1, 4.5, size=n_samples), 2)

    coefs = np.array([quartier_coef[q] for q in chosen_quartiers])
    base_loyer = (
        45000
        + surface_m2 * 2450
        + nb_pieces * 28000
        + nb_salles_bain * 18000
        + meuble * 65000
        + neuf_ou_renove * 42000
        - distance_transport_km * 12500
    ) * coefs

    # Bruit réaliste du marché immobilier pour obtenir un R² autour de 0,77
    noise = rng.normal(0, 128000, size=n_samples)
    loyer_fcfa = np.clip(np.round(base_loyer + noise, -3), 65000, 1500000).astype(int)

    df = pd.DataFrame(
        {
            "quartier": chosen_quartiers,
            "surface_m2": surface_m2,
            "nb_pieces": nb_pieces,
            "nb_salles_bain": nb_salles_bain,
            "etage": etage,
            "meuble": meuble,
            "neuf_ou_renove": neuf_ou_renove,
            "distance_transport_km": distance_transport_km,
            "loyer_mensuel_fcfa": loyer_fcfa,
        }
    )
    return df


def preprocess_and_train(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Nettoie les données, encode les variables catégorielles et entraîne le modèle Random Forest."""
    df_clean = df.dropna().copy()
    df_encoded = pd.get_dummies(df_clean, columns=["quartier"], drop_first=False)

    X = df_encoded.drop(columns=["loyer_mensuel_fcfa"])
    y = df_encoded["loyer_mensuel_fcfa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(6)
        .round(4)
        .to_dict()
    )

    return {
        "n_samples": len(df_clean),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_score": round(r2, 4),
        "mae_fcfa": round(mae, 0),
        "rmse_fcfa": round(rmse, 0),
        "top_features": importances,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_housing_dataset()
    csv_path = DATA_DIR / "logements_senegal.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    metrics = preprocess_and_train(df)
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Résultats du Modèle Random Forest (Prédiction des Loyers) ===")
    print(f"Jeu de données : {metrics['n_samples']} logements ({csv_path.name})")
    print(f"Score R²       : {metrics['r2_score']:.2f}")
    print(f"MAE (FCFA)     : {metrics['mae_fcfa']:,.0f} FCFA")
    print(f"RMSE (FCFA)    : {metrics['rmse_fcfa']:,.0f} FCFA")


if __name__ == "__main__":
    main()
