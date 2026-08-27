"""
app.py
------
Web-app Streamlit : Outil d'aide à la décision pour la maintenance prédictive.

Identité visuelle : "panneau de contrôle industriel" — cadrans, relevés
techniques, palette sombre inspirée des salles de supervision (SCADA).
Choix délibéré, ancré dans le métier (AMDEC / Pareto / Ishikawa), plutôt
qu'un thème générique de tableau de bord.

Lancer en local :
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Design tokens — source unique pour la palette (réutilisée dans le CSS
# injecté ci-dessous ET dans les graphiques matplotlib, pour la cohérence).
# ---------------------------------------------------------------------------
ARDOISE = "#FFFFFF"   # fond principal
ACIER   = "#F3F6F8"   # fond des panneaux / cartes
TRAIT   = "#D8E0E6"   # bordures, séparateurs
CRAIE   = "#1B2733"   # texte principal
BRUME   = "#5B6B7A"   # texte secondaire
CYAN    = "#1D8C82"   # accent — risque faible / signal actif
AMBRE   = "#C4790E"   # risque modéré
CORAIL  = "#C6402F"   # risque élevé

MODEL_PATH = "models/model.pkl"


def _build_example_datasets():
    """Trois jeux de données d'exemple, dans des domaines volontairement
    différents, pour démontrer concrètement que le mode générique ne se
    limite pas à l'industrie lourde."""
    rng = np.random.default_rng(7)

    n = 180
    auto = pd.DataFrame({
        "type_vehicule": rng.choice(["Citadine", "Berline", "SUV"], n, p=[0.4, 0.35, 0.25]),
        "kilometrage_km": rng.normal(80000, 40000, n).clip(0).round(0),
        "age_vehicule_ans": rng.normal(6, 3, n).clip(0).round(1),
        "temperature_moteur_C": rng.normal(90, 8, n).round(1),
    })
    score = 0.00002 * auto["kilometrage_km"] + 0.15 * auto["age_vehicule_ans"] + 0.08 * (auto["temperature_moteur_C"] - 95).clip(lower=0)
    p = 1 / (1 + np.exp(-(score - 2.3)))
    auto["panne"] = np.where(rng.random(n) < p, "Oui", "Non")

    rng2 = np.random.default_rng(11)
    n = 180
    agro = pd.DataFrame({
        "ligne_production": rng2.choice(["A", "B", "C"], n),
        "temperature_stockage_C": rng2.normal(4, 2.5, n).round(1),
        "humidite_pct": rng2.normal(60, 12, n).round(1),
        "duree_transport_h": rng2.normal(12, 6, n).clip(0).round(1),
        "ph": rng2.normal(6.5, 0.6, n).round(2),
    })
    temp_bad = agro["temperature_stockage_C"] > 7
    transport_bad = agro["duree_transport_h"] > 18
    ph_bad = (agro["ph"] - 6.5).abs() > 0.7
    score = temp_bad.astype(float) * 2.5 + transport_bad.astype(float) * 2.2 + ph_bad.astype(float) * 2.0 + rng2.normal(0, 0.25, n)
    p = 1 / (1 + np.exp(-(score - 1.5)))
    agro["defaut_qualite"] = np.where(rng2.random(n) < p, "Defaut", "OK")

    rng3 = np.random.default_rng(3)
    n = 160
    cosm = pd.DataFrame({
        "type_produit": rng3.choice(["Creme", "Serum", "Lotion"], n),
        "ph": rng3.normal(5.5, 0.9, n).round(2),
        "viscosite_cp": rng3.normal(3000, 1000, n).clip(0).round(0),
        "duree_stockage_mois": rng3.normal(8, 5, n).clip(0).round(1),
    })
    ph_bad2 = (cosm["ph"] < 5.0) | (cosm["ph"] > 6.0)
    store_bad = cosm["duree_stockage_mois"] > 11
    score = ph_bad2.astype(float) * 2.0 + store_bad.astype(float) * 1.8 + rng3.normal(0, 0.4, n)
    p = 1 / (1 + np.exp(-(score - 1.2)))
    cosm["conformite"] = np.where(rng3.random(n) < p, "NonConforme", "Conforme")

    return auto, agro, cosm


EXAMPLE_AUTOMOBILE, EXAMPLE_AGROALIMENTAIRE, EXAMPLE_COSMETIQUE = _build_example_datasets()

ISHIKAWA_MAP = {
    "temperature_air_K": "Milieu (environnement thermique)",
    "temperature_process_K": "Milieu (environnement thermique)",
    "vitesse_rotation_rpm": "Méthode (paramètres process)",
    "couple_Nm": "Méthode (paramètres process)",
    "usure_outil_min": "Matériel (usure machine)",
    "type_machine_L": "Matériel (type de machine)",
    "type_machine_M": "Matériel (type de machine)",
    "type_machine_H": "Matériel (type de machine)",
}

st.set_page_config(
    page_title="Maintenance Prédictive — Aide à la décision",
    page_icon="◈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — polices + composants personnalisés (auto-contenus, ne dépendent pas
# de la structure interne de Streamlit, donc stables entre les versions).
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
}}
h1, h2, h3, .mp-title {{
    font-family: 'Space Grotesk', sans-serif !important;
}}
.mp-mono {{
    font-family: 'IBM Plex Mono', monospace;
}}

/* ---- Hero ---- */
.mp-hero {{
    border-bottom: 1.5px solid {CRAIE};
    padding-bottom: 18px;
    margin-bottom: 26px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 12px;
}}
.mp-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2.5px;
    color: {CYAN};
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.mp-title {{
    font-size: 30px;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.5px;
    color: {CRAIE};
}}
.mp-subtitle {{
    color: {BRUME};
    font-size: 14px;
    margin-top: 6px;
    max-width: 560px;
    line-height: 1.5;
}}
.mp-status {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: {BRUME};
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}}
.mp-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {CYAN};
}}

/* ---- Panel header (au-dessus des graphiques) ---- */
.mp-panel-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {BRUME};
    border-bottom: 1px solid {TRAIT};
    padding-bottom: 8px;
    margin: 28px 0 14px 0;
}}

/* ---- Gauge card ---- */
.mp-card {{
    background: {ACIER};
    border: 1px solid {TRAIT};
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 10px;
}}
.mp-card-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
}}
.mp-machine-id {{
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}}
.mp-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 3px;
    border: 1px solid {TRAIT};
    color: {BRUME};
    background: {ARDOISE};
}}
.mp-readings {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: {BRUME};
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}
.mp-readings b {{ color: {CRAIE}; font-weight: 500; }}
.mp-status-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 3px;
    letter-spacing: 0.5px;
    white-space: nowrap;
}}
.mp-status-critique {{ background: rgba(198,64,47,0.12); color: {CORAIL}; }}
.mp-status-modere   {{ background: rgba(196,121,14,0.12); color: {AMBRE}; }}
.mp-status-faible   {{ background: rgba(29,140,130,0.12); color: {CYAN}; }}

.mp-gauge-row {{ display: flex; align-items: center; gap: 14px; }}
.mp-gauge-track {{
    position: relative; flex: 1; height: 6px;
    background: {TRAIT}; border-radius: 3px;
}}
.mp-gauge-zone {{ position: absolute; top: 0; bottom: 0; border-radius: 3px; }}
.mp-gauge-marker {{
    position: absolute; top: -5px; width: 2px; height: 16px; background: {CRAIE};
}}
.mp-gauge-marker::after {{
    content: ''; position: absolute; top: -4px; left: -3px;
    width: 8px; height: 8px; border-radius: 50%; background: {CRAIE};
}}
.mp-gauge-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 600; width: 50px; text-align: right;
    color: {CRAIE};
}}
.mp-recommendation {{
    margin-top: 12px; font-size: 13px; color: {BRUME}; line-height: 1.5;
    border-left: 2px solid {TRAIT}; padding-left: 12px;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features_num"], bundle["features_cat"]


def risk_zone(p):
    if p < 0.30:
        return "faible", "RISQUE FAIBLE"
    elif p < 0.70:
        return "modere", "RISQUE MODÉRÉ"
    return "critique", "RISQUE ÉLEVÉ"


def identify_causes(row):
    causes = []
    if row.get("temperature_process_K", 0) > 311:
        causes.append("température process élevée")
    if row.get("usure_outil_min", 0) > 200:
        causes.append("usure outil avancée")
    if row.get("couple_Nm", 0) > 55:
        causes.append("couple mécanique élevé")
    if row.get("vitesse_rotation_rpm", 0) > 2200:
        causes.append("vitesse de rotation excessive")
    return causes


def recommandation(row, p):
    if p < 0.30:
        return None
    causes = identify_causes(row)
    if not causes:
        causes = ["combinaison de facteurs de stress mécanique/thermique"]
    if p < 0.70:
        return f"Planifier une inspection sous 7 jours — {', '.join(causes)}."
    return f"Intervention recommandée sous 48h — {', '.join(causes)}."


def render_gauge_card(row, p):
    zone, label = risk_zone(p)
    pct = p * 100
    reco = recommandation(row, p)
    reco_html = f'<div class="mp-recommendation">{reco}</div>' if reco else ""
    return f"""
    <div class="mp-card">
        <div class="mp-card-top">
            <div class="mp-machine-id">
                <span class="mp-badge">TYPE {row.get('type_machine', '?')}</span>
                <span class="mp-readings">
                    <span>Temp. process <b>{row.get('temperature_process_K', 0):.1f}K</b></span>
                    <span>Usure <b>{row.get('usure_outil_min', 0):.0f}min</b></span>
                    <span>Couple <b>{row.get('couple_Nm', 0):.1f}Nm</b></span>
                    <span>Vitesse <b>{row.get('vitesse_rotation_rpm', 0):.0f}rpm</b></span>
                </span>
            </div>
            <span class="mp-status-badge mp-status-{zone}">{label}</span>
        </div>
        <div class="mp-gauge-row">
            <div class="mp-gauge-track">
                <div class="mp-gauge-zone" style="left:0%; width:30%; background:rgba(29,140,130,0.30);"></div>
                <div class="mp-gauge-zone" style="left:30%; width:40%; background:rgba(196,121,14,0.30);"></div>
                <div class="mp-gauge-zone" style="left:70%; width:30%; background:rgba(198,64,47,0.30);"></div>
                <div class="mp-gauge-marker" style="left:{pct:.1f}%;"></div>
            </div>
            <div class="mp-gauge-value">{pct:.1f}%</div>
        </div>
        {reco_html}
    </div>
    """


def style_dark_axes(ax, fig):
    fig.patch.set_facecolor(ACIER)
    ax.set_facecolor(ACIER)
    ax.tick_params(colors=BRUME, labelsize=9)
    ax.xaxis.label.set_color(BRUME)
    ax.yaxis.label.set_color(BRUME)
    ax.title.set_color(CRAIE)
    for spine in ax.spines.values():
        spine.set_color(TRAIT)


def plot_feature_importance(model):
    clf = model.named_steps["clf"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({
        "variable": feature_names,
        "importance": importances,
        "famille": [ISHIKAWA_MAP.get(f, "Autre") for f in feature_names],
    })
    imp_df = imp_df.groupby("famille", as_index=False)["importance"].sum()
    imp_df = imp_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(imp_df["famille"], imp_df["importance"], color=CYAN, height=0.55)
    ax.set_xlabel("Importance relative dans le modèle")
    ax.set_title("Facteurs déterminants — lecture inspirée de l'Ishikawa", fontsize=11, loc="left")
    style_dark_axes(ax, fig)
    fig.tight_layout()
    return fig


def plot_pareto_causes(df, proba):
    at_risk = df[proba >= 0.30]
    if at_risk.empty:
        return None
    all_causes = []
    for _, row in at_risk.iterrows():
        causes = identify_causes(row.to_dict())
        if causes:
            all_causes.append(causes[0])
    if not all_causes:
        return None

    counts = pd.Series(all_causes).value_counts()
    cum_pct = counts.cumsum() / counts.sum() * 100

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.bar(counts.index, counts.values, color=CYAN, width=0.5)
    ax1.set_ylabel("Machines concernées")
    ax1.tick_params(axis="x", rotation=15, colors=BRUME, labelsize=9)
    style_dark_axes(ax1, fig)

    ax2 = ax1.twinx()
    ax2.plot(counts.index, cum_pct.values, color=AMBRE, marker="o", linewidth=1.8)
    ax2.set_ylabel("% cumulé", color=BRUME)
    ax2.set_ylim(0, 110)
    ax2.tick_params(colors=BRUME, labelsize=9)
    ax2.spines["top"].set_color(TRAIT)
    ax2.spines["right"].set_color(TRAIT)

    ax1.set_title("Pareto des causes dominantes sur le parc", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="mp-eyebrow">À propos</div>', unsafe_allow_html=True)
    st.markdown(
        "Développé par **Rania El Ouardi**, élève-ingénieure en Génie "
        "Industriel (ESITH Casablanca), en prolongement d'un PFA mené chez "
        "**SMI (Groupe Managem)** sur la digitalisation d'un parc de 100 "
        "engins (12 512 relevés, +8 points de disponibilité, ROI < 3 mois)."
    )
    st.markdown(
        "L'outil combine une brique prédictive (Random Forest) avec une "
        "lecture inspirée des outils Lean Six Sigma utilisés sur le terrain "
        "— AMDEC, Pareto, Ishikawa."
    )
    st.markdown("[Code source sur GitHub](#)")
    st.caption(
        "Démo entraînée sur données synthétiques par défaut. "
        "Voir adapt_real_dataset.py pour basculer sur le dataset réel AI4I 2020."
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
model, features_num, features_cat = load_model()

st.markdown(f"""
<div class="mp-hero">
    <div>
        <div class="mp-eyebrow">Outil d'aide à la décision</div>
        <h1 class="mp-title">Maintenance Prédictive</h1>
        <div class="mp-subtitle">Estime la probabilité de panne d'une machine à partir de ses relevés capteurs, avec diagnostic des causes dominantes.</div>
    </div>
    <div class="mp-status"><span class="mp-dot"></span>Modèle chargé — prêt à analyser</div>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "Mode",
    ["🏭 Modèle Industrie — pré-entraîné (SMI / AI4I 2020)", "🗂️ Mode générique — entraîner sur mes propres données"],
    label_visibility="collapsed",
    horizontal=True,
)

st.divider()

if mode.startswith("🏭"):

    with st.expander("Format de fichier attendu"):
        st.write("Colonnes requises (CSV ou Excel — .csv, .xlsx, .xls) :")
        st.code(", ".join(features_cat + features_num))
        example = pd.DataFrame({
            "type_machine": ["M", "L", "H"],
            "temperature_air_K": [300.5, 298.9, 302.1],
            "temperature_process_K": [310.2, 308.5, 314.8],
            "vitesse_rotation_rpm": [1520, 1410, 2350],
            "couple_Nm": [42.1, 38.7, 61.2],
            "usure_outil_min": [95, 40, 215],
        })
        st.dataframe(example, width="stretch")

        import io
        excel_buffer = io.BytesIO()
        example.to_excel(excel_buffer, index=False, engine="openpyxl")

        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            st.download_button(
                "Télécharger un exemple CSV",
                example.to_csv(index=False).encode("utf-8"),
                file_name="exemple_relevés.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_xlsx:
            st.download_button(
                "Télécharger un exemple Excel",
                excel_buffer.getvalue(),
                file_name="exemple_relevés.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    # Alias de colonnes reconnus automatiquement — permet d'accepter directement
    # le fichier brut Kaggle (AI4I 2020) sans passer par adapt_real_dataset.py,
    # en plus du format "natif" du projet.
    COLUMN_ALIASES = {
        "type": "type_machine",
        "Type": "type_machine",
        "air temperature [k]": "temperature_air_K",
        "Air temperature [K]": "temperature_air_K",
        "process temperature [k]": "temperature_process_K",
        "Process temperature [K]": "temperature_process_K",
        "rotational speed [rpm]": "vitesse_rotation_rpm",
        "Rotational speed [rpm]": "vitesse_rotation_rpm",
        "torque [nm]": "couple_Nm",
        "Torque [Nm]": "couple_Nm",
        "tool wear [min]": "usure_outil_min",
        "Tool wear [min]": "usure_outil_min",
    }


    def normalize_columns(df):
        """Reconnaît automatiquement plusieurs formats de colonnes courants
        (natif du projet, ou fichier brut Kaggle AI4I 2020) et les convertit
        vers le format interne attendu par le modèle."""
        rename_map = {}
        for col in df.columns:
            key = col.strip()
            if key in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[key]
            elif key.lower() in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[key.lower()]
        return df.rename(columns=rename_map)


    uploaded = st.file_uploader("Fichier de relevés (CSV ou Excel)", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
        df = normalize_columns(df)
        missing = set(features_num + features_cat) - set(df.columns)
        if missing:
            st.error(
                f"Colonnes manquantes ou non reconnues : {', '.join(sorted(missing))}. "
                "Utilisez l'exemple ci-dessus, ou le dataset AI4I 2020 (brut ou converti)."
            )
        else:
            proba = model.predict_proba(df[features_num + features_cat])[:, 1]

            n_high = int((proba >= 0.70).sum())
            n_mid = int(((proba >= 0.30) & (proba < 0.70)).sum())
            n_low = int((proba < 0.30).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Risque élevé", n_high)
            c2.metric("Risque modéré", n_mid)
            c3.metric("Risque faible", n_low)

            order = proba.argsort()[::-1]

            MAX_CARDS = 25
            n_shown = min(MAX_CARDS, len(order))
            header_label = "Machines — triées par risque décroissant"
            if len(order) > MAX_CARDS:
                header_label += f" (top {n_shown} sur {len(order)} — voir le tableau brut pour le parc complet)"
            st.markdown(f'<div class="mp-panel-header">{header_label}</div>', unsafe_allow_html=True)

            for i in order[:MAX_CARDS]:
                row = df.iloc[i].to_dict()
                st.markdown(render_gauge_card(row, proba[i]), unsafe_allow_html=True)

            with st.expander(f"Voir le tableau brut ({len(order)} machines)"):
                df_out = df.copy()
                df_out["probabilité_panne"] = (proba * 100).round(1)
                df_out["niveau_risque"] = [risk_zone(p)[1] for p in proba]
                st.dataframe(df_out.iloc[order], width="stretch")
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    df_out.to_csv(index=False).encode("utf-8"),
                    file_name="resultats_maintenance_predictive.csv",
                    mime="text/csv",
                )

            st.markdown('<div class="mp-panel-header">Analyse</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Quels facteurs pèsent le plus dans les décisions du modèle.")
                st.pyplot(plot_feature_importance(model))
            with col_b:
                st.caption("Sur les machines à risque, quelles causes dominent (règle 80/20).")
                pareto_fig = plot_pareto_causes(df, proba)
                if pareto_fig is not None:
                    st.pyplot(pareto_fig)
                else:
                    st.info("Aucune machine à risque dans ce fichier.")
    else:
        st.info("En attente d'un fichier CSV — utilisez l'exemple ci-dessus pour tester l'outil.")

else:
    # -----------------------------------------------------------------------
    # Mode générique — entraîne un nouveau modèle à la volée sur le fichier
    # déposé, quel que soit le domaine (industrie, agroalimentaire,
    # automobile, cosmétique...), tant qu'il y a une colonne "résultat"
    # binaire et des colonnes de mesures.
    # -----------------------------------------------------------------------
    st.markdown(
        "Dépose un fichier avec une colonne indiquant un problème "
        "(panne, défaut, échec... deux valeurs possibles) et des colonnes de "
        "mesures. L'app entraîne un nouveau modèle sur **ce fichier précis** "
        "— aucune connaissance préalable du domaine n'est nécessaire, ça "
        "fonctionne pour l'industrie comme pour l'agroalimentaire, "
        "l'automobile, la cosmétique, etc."
    )

    with st.expander("📋 Comment doit être structuré mon fichier ? (à lire avant d'uploader)"):
        st.markdown("""
**Trois règles simples à respecter :**

1. **Une ligne = un cas observé** (une machine, un lot de production, un
   véhicule, un produit... peu importe le domaine)
2. **Une colonne "résultat"** avec **exactement 2 valeurs possibles**
   (ex. `Panne` / `OK`, `Défaut` / `Conforme`, `0` / `1`, `Oui` / `Non`).
   Le nom de la colonne n'a pas d'importance — tu la sélectionnes toi-même
   dans un menu après l'upload.
3. **Plusieurs colonnes de mesures** qui pourraient expliquer ce résultat
   (température, durée, taux, catégorie...). Nombres ou texte acceptés.

**Pour un résultat fiable :**
- Au moins **50 à 100 lignes** (en dessous, l'app te préviendra que ce
  n'est pas assez pour être fiable)
- Le moins possible de cases vides
- Des mesures qui ont un lien plausible avec le résultat — l'app te donne
  un score de fiabilité (AUC) après entraînement pour te dire si c'est le
  cas ou non

**Ce qui bloque l'upload :**
- Une colonne résultat avec plus ou moins de 2 valeurs (ex. "Faible /
  Moyen / Élevé" ne marche pas tel quel — il faudrait la simplifier en 2
  catégories avant l'upload)
- Moins de 30 lignes exploitables

**Tu as déjà ton propre fichier ? Tu n'as pas besoin des exemples ci-dessous** —
dépose-le directement dans la zone d'upload plus bas. Si l'app te bloque,
voici comment corriger les cas les plus fréquents, directement dans Excel :

- **"Ma colonne résultat a plus de 2 valeurs"** (ex. Faible/Moyen/Élevé, ou
  une note de 0 à 10) → crée une **nouvelle colonne** à côté avec une
  formule qui simplifie en 2 cas. Exemple dans Excel :
  `=SI(A2>7;"Problème";"OK")` (remplace `A2` et le seuil `7` par tes
  valeurs), puis recopie la formule sur toutes les lignes.
- **"Il y a des cases vides"** → soit tu les remplis avec une valeur
  raisonnable (moyenne de la colonne, par exemple), soit tu supprimes les
  lignes concernées si elles ne sont pas trop nombreuses.
- **"Mes données sont sur plusieurs feuilles Excel"** → l'app ne lit que la
  **première feuille** du fichier. Copie les données utiles sur la
  première feuille avant l'upload, ou exporte cette feuille seule en CSV.
- **"Je ne sais pas quelles colonnes garder"** → garde tout, l'app te laisse
  choisir après l'upload, et tu peux comparer les résultats en
  décochant/recochant des colonnes et en relançant l'entraînement.

**Trois exemples ci-dessous, dans des domaines totalement différents, pour tester tout de suite :**
        """)
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            st.caption("🚗 Automobile — prédiction de panne")
            st.download_button(
                "Télécharger l'exemple",
                EXAMPLE_AUTOMOBILE.to_csv(index=False).encode("utf-8"),
                file_name="exemple_automobile.csv", mime="text/csv", width="stretch", key="dl_auto",
            )
        with ex_col2:
            st.caption("🥫 Agroalimentaire — défaut qualité")
            st.download_button(
                "Télécharger l'exemple",
                EXAMPLE_AGROALIMENTAIRE.to_csv(index=False).encode("utf-8"),
                file_name="exemple_agroalimentaire.csv", mime="text/csv", width="stretch", key="dl_agro",
            )
        with ex_col3:
            st.caption("🧴 Cosmétique — non-conformité")
            st.download_button(
                "Télécharger l'exemple",
                EXAMPLE_COSMETIQUE.to_csv(index=False).encode("utf-8"),
                file_name="exemple_cosmetique.csv", mime="text/csv", width="stretch", key="dl_cosm",
            )

    generic_file = st.file_uploader(
        "Fichier de données (CSV ou Excel)", type=["csv", "xlsx", "xls"], key="generic_upload"
    )

    if generic_file is not None:
        if generic_file.name.lower().endswith((".xlsx", ".xls")):
            gdf = pd.read_excel(generic_file)
        else:
            gdf = pd.read_csv(generic_file)

        st.caption(f"{len(gdf)} lignes, {len(gdf.columns)} colonnes détectées.")
        with st.expander("Aperçu des données"):
            st.dataframe(gdf.head(20), width="stretch")

        target_col = st.selectbox(
            "Quelle colonne indique le problème à prédire ? (doit avoir exactement 2 valeurs possibles)",
            options=gdf.columns,
        )

        valid_target = gdf[target_col].nunique(dropna=True) == 2
        if not valid_target:
            st.error(
                f"La colonne « {target_col} » a {gdf[target_col].nunique(dropna=True)} valeurs "
                "différentes — il en faut exactement 2 (ex. 0/1, Oui/Non, Panne/OK)."
            )
        else:
            values = gdf[target_col].dropna().value_counts()
            default_risk_value = values.idxmin()  # la valeur la plus rare = souvent le "problème"
            risk_value = st.selectbox(
                "Laquelle de ces deux valeurs représente le problème / la panne ?",
                options=values.index.tolist(),
                index=values.index.tolist().index(default_risk_value),
            )

            feature_options = [c for c in gdf.columns if c != target_col]
            feature_cols_generic = st.multiselect(
                "Colonnes à utiliser comme mesures", options=feature_options, default=feature_options
            )

            train_clicked = st.button("Entraîner un modèle sur ce fichier", type="primary")

            if train_clicked and feature_cols_generic:
                clean = gdf.dropna(subset=[target_col])
                y_generic = (clean[target_col] == risk_value).astype(int)
                X_generic = clean[feature_cols_generic]

                num_cols_g = X_generic.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols_g = [c for c in feature_cols_generic if c not in num_cols_g]

                if y_generic.nunique() < 2 or len(clean) < 30:
                    st.error("Pas assez de données ou de diversité pour entraîner un modèle fiable (minimum ~30 lignes avec les deux cas).")
                else:
                    preprocess_g = ColumnTransformer(
                        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_g)],
                        remainder="passthrough", verbose_feature_names_out=False,
                    )
                    model_g = Pipeline([
                        ("preprocess", preprocess_g),
                        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)),
                    ])

                    stratify_g = y_generic if y_generic.value_counts().min() >= 2 else None
                    Xtr, Xte, ytr, yte = train_test_split(
                        X_generic, y_generic, test_size=0.2, random_state=42, stratify=stratify_g
                    )
                    model_g.fit(Xtr, ytr)
                    proba_test = model_g.predict_proba(Xte)[:, 1]
                    auc = roc_auc_score(yte, proba_test) if yte.nunique() == 2 else float("nan")

                    st.session_state["generic_model"] = model_g
                    st.session_state["generic_features"] = feature_cols_generic
                    st.session_state["generic_num_cols"] = num_cols_g
                    st.session_state["generic_cat_cols"] = cat_cols_g
                    st.session_state["generic_auc"] = auc
                    st.session_state["generic_data"] = clean
                    st.session_state["generic_target"] = target_col
                    st.session_state["generic_risk_value"] = risk_value

            if "generic_model" in st.session_state and st.session_state.get("generic_target") == target_col:
                model_g = st.session_state["generic_model"]
                clean = st.session_state["generic_data"]
                feats = st.session_state["generic_features"]
                auc = st.session_state["generic_auc"]

                st.success(f"Modèle entraîné — AUC-ROC sur données de test : {auc:.3f}" if auc == auc else "Modèle entraîné.")
                if auc == auc and auc < 0.65:
                    st.warning(
                        "AUC assez faible (< 0.65) : les colonnes sélectionnées expliquent peu le "
                        "problème dans ce fichier. Les résultats ci-dessous restent indicatifs."
                    )

                proba_all = model_g.predict_proba(clean[feats])[:, 1]

                n_high = int((proba_all >= 0.70).sum())
                n_mid = int(((proba_all >= 0.30) & (proba_all < 0.70)).sum())
                n_low = int((proba_all < 0.30).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Risque élevé", n_high)
                c2.metric("Risque modéré", n_mid)
                c3.metric("Risque faible", n_low)

                order_g = proba_all.argsort()[::-1]
                MAX_CARDS_G = 25
                label_g = f"Lignes — triées par risque décroissant"
                if len(order_g) > MAX_CARDS_G:
                    label_g += f" (top {MAX_CARDS_G} sur {len(order_g)})"
                st.markdown(f'<div class="mp-panel-header">{label_g}</div>', unsafe_allow_html=True)

                display_cols = feats[:4]
                for i in order_g[:MAX_CARDS_G]:
                    row = clean.iloc[i]
                    p = proba_all[i]
                    zone, zlabel = risk_zone(p)
                    readings_html = "".join(
                        f'<span>{c} <b>{row[c]}</b></span>' for c in display_cols
                    )
                    st.markdown(f"""
                    <div class="mp-card">
                        <div class="mp-card-top">
                            <div class="mp-machine-id">
                                <span class="mp-readings">{readings_html}</span>
                            </div>
                            <span class="mp-status-badge mp-status-{zone}">{zlabel}</span>
                        </div>
                        <div class="mp-gauge-row">
                            <div class="mp-gauge-track">
                                <div class="mp-gauge-zone" style="left:0%; width:30%; background:rgba(29,140,130,0.30);"></div>
                                <div class="mp-gauge-zone" style="left:30%; width:40%; background:rgba(196,121,14,0.30);"></div>
                                <div class="mp-gauge-zone" style="left:70%; width:30%; background:rgba(198,64,47,0.30);"></div>
                                <div class="mp-gauge-marker" style="left:{p*100:.1f}%;"></div>
                            </div>
                            <div class="mp-gauge-value">{p*100:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="mp-panel-header">Facteurs déterminants</div>', unsafe_allow_html=True)
                st.caption("Quelles colonnes pèsent le plus dans les décisions de ce modèle (entraîné uniquement sur ce fichier).")
                clf_g = model_g.named_steps["clf"]
                names_g = model_g.named_steps["preprocess"].get_feature_names_out()
                imp_g = pd.DataFrame({"variable": names_g, "importance": clf_g.feature_importances_})

                def origin_col(name):
                    for c in st.session_state["generic_cat_cols"]:
                        if name.startswith(c + "_"):
                            return c
                    return name

                imp_g["colonne"] = imp_g["variable"].apply(origin_col)
                imp_g = imp_g.groupby("colonne", as_index=False)["importance"].sum().sort_values("importance", ascending=True)

                fig_g, ax_g = plt.subplots(figsize=(7, 3.2))
                ax_g.barh(imp_g["colonne"], imp_g["importance"], color=CYAN, height=0.55)
                ax_g.set_xlabel("Importance relative dans le modèle")
                style_dark_axes(ax_g, fig_g)
                fig_g.tight_layout()
                st.pyplot(fig_g)

                with st.expander(f"Voir le tableau complet ({len(clean)} lignes)"):
                    out_g = clean.copy()
                    out_g["probabilité"] = (proba_all * 100).round(1)
                    out_g["niveau_risque"] = [risk_zone(p)[1] for p in proba_all]
                    st.dataframe(out_g.iloc[order_g], width="stretch")
                    st.download_button(
                        "Télécharger les résultats (CSV)",
                        out_g.to_csv(index=False).encode("utf-8"),
                        file_name="resultats_mode_generique.csv",
                        mime="text/csv",
                    )
    else:
        st.info("En attente d'un fichier — l'app entraînera un modèle spécifiquement sur tes données, quel que soit le domaine.")
