# Maintenance Prédictive — Outil d'aide à la décision

Application web (Streamlit) qui estime la probabilité de panne d'une machine
industrielle à partir de ses relevés capteurs (température, vibrations,
couple, usure), avec diagnostic des causes dominantes et recommandation
d'action.

Ce projet prolonge un **PFA mené chez SMI (Groupe Managem, Tinghir)** sur la
digitalisation du suivi d'un parc de 100 engins (12 512 relevés analysés
sous Power BI, +8 points de disponibilité, ROI < 3 mois). Il combine une
brique prédictive (Machine Learning) avec une lecture inspirée des outils
Lean Six Sigma utilisés sur le terrain — AMDEC, Pareto, Ishikawa.

**Démo en ligne :** [rania-maintenance-predictive.streamlit.app](https://rania-maintenance-predictive.streamlit.app)

## Identité visuelle

Palette et typographie inspirées des salles de supervision industrielle
(SCADA) : fond sombre technique, cadrans de risque, relevés en police
monospace — pas un thème de tableau de bord générique. Chaque machine
analysée s'affiche comme une carte-instrument avec une jauge de risque à
trois zones (faible / modéré / critique) plutôt qu'un simple tableau.

## Aperçu

![capture d'écran à ajouter](docs/screenshot.png)

## Fonctionnement

1. L'utilisateur dépose un fichier CSV de relevés capteurs récents
2. Un modèle Random Forest (scikit-learn) calcule une probabilité de panne
   par machine
3. L'app affiche une jauge de risque, une recommandation d'action, et deux
   analyses complémentaires :
   - **Diagnostic type Ishikawa** — quels facteurs pèsent le plus dans les
     décisions du modèle, regroupés par famille de cause
   - **Pareto des causes** — sur les machines à risque, quelles causes
     dominantes reviennent le plus souvent (règle 80/20)

## Stack technique

- **Python** (pandas, numpy, scikit-learn, joblib, matplotlib)
- **Streamlit** pour l'interface web, avec composants HTML/CSS personnalisés
- **Random Forest Classifier** (`class_weight="balanced"`) pour gérer le
  déséquilibre naturel entre pannes et fonctionnement normal

## Installation & lancement en local

```bash
git clone <url-du-repo>
cd maintenance_predictive_app
pip install -r requirements.txt

# 1. Générer un jeu de données (ou déposer le tien dans data/maintenance_data.csv)
python generate_synthetic_data.py

# 2. Entraîner le modèle
python train_model.py

# 3. Lancer l'app
streamlit run app.py
```

## Passer sur données réelles

Le modèle par défaut est entraîné sur un jeu de données **synthétique**
(`generate_synthetic_data.py`), généré pour prototyper le pipeline de bout
en bout. Pour passer sur données réelles :

1. Télécharge le **AI4I 2020 Predictive Maintenance Dataset** sur Kaggle
2. Place-le dans `data/ai4i2020.csv`
3. Lance `python adapt_real_dataset.py` (convertit automatiquement les
   colonnes vers le format utilisé par ce projet)
4. Relance `python train_model.py`

Le reste du pipeline (`app.py`) n'a besoin d'aucune modification.

## Déploiement (gratuit)

1. Pousser ce repo sur GitHub
2. Créer un compte sur [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connecter le repo → déploiement automatique en quelques minutes
4. Ajouter le lien de l'app dans ce README et sur le CV / LinkedIn

## Prochaines améliorations possibles

- Comparer plusieurs modèles (XGBoost, Gradient Boosting) et documenter les
  performances
- Ajouter un historique des relevés (suivi de tendance par machine)
- Exporter un rapport PDF des machines à risque

## Sources & licences

- **Dataset** : AI4I 2020 Predictive Maintenance Dataset — S. Matzka,
  *"Explainable Artificial Intelligence for Predictive Maintenance
  Applications,"* 2020 Third International Conference on Artificial
  Intelligence for Industries (AI4I). Disponible sur
  [UCI Machine Learning Repository](https://doi.org/10.24432/C5HS5C),
  sous licence **CC BY 4.0** (partage et adaptation autorisés avec
  attribution). Le fichier n'est pas redistribué dans ce dépôt — voir
  section ci-dessus pour le télécharger soi-même.
- **Code** : bibliothèques open-source (scikit-learn, pandas, numpy,
  Streamlit, matplotlib), licences BSD/Apache 2.0.
- **Polices** : Space Grotesk, IBM Plex Sans/Mono (Google Fonts), licence
  SIL Open Font License.

---
Projet réalisé par **Rania El OUARDI** — élève-ingénieure, Génie Industriel,
Excellence Opérationnelle & Smart Manufacturing, ESITH Casablanca.
