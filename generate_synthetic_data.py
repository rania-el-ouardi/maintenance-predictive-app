"""
generate_synthetic_data.py
---------------------------
Génère un jeu de données SYNTHÉTIQUE de capteurs industriels (température,
vibrations, vitesse de rotation, usure outil) avec une étiquette de panne.

⚠️ Ce jeu de données est fictif — il sert uniquement à faire tourner le
pipeline de bout en bout pendant le développement (pas d'accès internet
dans cet environnement pour télécharger un vrai dataset).

Avant la version finale du projet, remplace ce fichier par un vrai jeu de
données public, par exemple :
  - "AI4I 2020 Predictive Maintenance Dataset" (Kaggle / UCI Machine
    Learning Repository) — cherche ces mots-clés sur kaggle.com
  - ou directement tes propres relevés (si SMI/Managem t'autorise à
    réutiliser une version anonymisée de tes 12 512 relevés)

Usage :
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 8000


def generate():
    machine_type = RNG.choice(["L", "M", "H"], size=N_SAMPLES, p=[0.5, 0.3, 0.2])

    air_temp = RNG.normal(300, 2, N_SAMPLES)  # Kelvin
    process_temp = air_temp + RNG.normal(10, 1, N_SAMPLES)
    rotational_speed = RNG.normal(1500, 180, N_SAMPLES).clip(1000, 2800)
    torque = RNG.normal(40, 10, N_SAMPLES).clip(3, 80)
    tool_wear = RNG.integers(0, 260, N_SAMPLES)

    # Probabilité de panne : construite à partir d'une combinaison de
    # facteurs de stress mécanique/thermique, avec un signal net (peu de
    # bruit) pour que le modèle de démo apprenne une frontière claire.
    stress_score = (
        0.35 * (process_temp - 308).clip(min=0)
        + 0.035 * (tool_wear - 150).clip(min=0)
        + 0.12 * (torque - 50).clip(min=0)
        + 0.02 * (rotational_speed - 2000).clip(min=0)
    )
    failure_prob = 1 / (1 + np.exp(-(stress_score - 3.4)))  # sigmoid, pente marquée
    noise = RNG.normal(0, 0.05, N_SAMPLES)  # léger bruit résiduel
    failure = (RNG.random(N_SAMPLES) < (failure_prob + noise).clip(0, 1)).astype(int)

    df = pd.DataFrame({
        "type_machine": machine_type,
        "temperature_air_K": air_temp.round(1),
        "temperature_process_K": process_temp.round(1),
        "vitesse_rotation_rpm": rotational_speed.round(0),
        "couple_Nm": torque.round(1),
        "usure_outil_min": tool_wear,
        "panne": failure,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/maintenance_data.csv"
    df.to_csv(out_path, index=False)
    print(f"{len(df)} lignes générées -> {out_path}")
    print(f"Taux de panne dans le jeu de données : {df['panne'].mean():.1%}")
