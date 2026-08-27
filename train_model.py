"""
train_model.py
---------------
Entraîne un modèle de classification (Random Forest) pour prédire la
probabilité de panne d'une machine à partir de ses relevés capteurs.

Usage :
    python generate_synthetic_data.py   # une seule fois, si data/maintenance_data.csv n'existe pas
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = "data/maintenance_data.csv"
MODEL_PATH = "models/model.pkl"

FEATURES_NUM = [
    "temperature_air_K",
    "temperature_process_K",
    "vitesse_rotation_rpm",
    "couple_Nm",
    "usure_outil_min",
]
FEATURES_CAT = ["type_machine"]
TARGET = "panne"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} introuvable. Lance d'abord : python generate_synthetic_data.py "
            "(ou dépose ton propre CSV avec les mêmes colonnes à cet emplacement)."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        )),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Rapport de classification ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "features_num": FEATURES_NUM, "features_cat": FEATURES_CAT},
        MODEL_PATH,
    )
    print(f"\nModèle sauvegardé -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
