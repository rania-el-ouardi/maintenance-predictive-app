"""
adapt_real_dataset.py
----------------------
Convertit le vrai jeu de données Kaggle "AI4I 2020 Predictive Maintenance
Dataset" (colonnes en anglais avec unités) vers le même format que celui
utilisé jusqu'ici (data/maintenance_data.csv), pour que train_model.py et
app.py n'aient RIEN à changer.

Avant de lancer ce script :
1. Télécharge le dataset sur Kaggle (cherche "AI4I 2020 Predictive
   Maintenance Dataset")
2. Place le fichier téléchargé dans data/ , renommé "ai4i2020.csv"

Usage :
    python adapt_real_dataset.py
"""

import os
import pandas as pd

RAW_PATH = "data/ai4i2020.csv"
OUT_PATH = "data/maintenance_data.csv"

# Mapping colonnes réelles -> colonnes utilisées dans ce projet
COLUMN_MAP = {
    "Type": "type_machine",
    "Air temperature [K]": "temperature_air_K",
    "Process temperature [K]": "temperature_process_K",
    "Rotational speed [rpm]": "vitesse_rotation_rpm",
    "Torque [Nm]": "couple_Nm",
    "Tool wear [min]": "usure_outil_min",
    "Machine failure": "panne",
}


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} introuvable. Télécharge le dataset AI4I 2020 depuis "
            f"Kaggle et place-le dans data/ai4i2020.csv avant de relancer ce script."
        )

    df = pd.read_csv(RAW_PATH)

    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes attendues introuvables dans le fichier téléchargé : {missing}. "
            "Vérifie que tu as bien téléchargé le dataset AI4I 2020 (et pas un autre)."
        )

    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} lignes converties -> {OUT_PATH}")
    print(f"Taux de panne dans le jeu de données réel : {df['panne'].mean():.1%}")
    print("\nTu peux maintenant relancer : python train_model.py")


if __name__ == "__main__":
    main()
