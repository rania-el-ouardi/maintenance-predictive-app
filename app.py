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
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

with st.expander("Format de fichier attendu"):
    st.write("Colonnes requises :")
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
    st.download_button(
        "Télécharger un exemple CSV",
        example.to_csv(index=False).encode("utf-8"),
        file_name="exemple_relevés.csv",
        mime="text/csv",
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


uploaded = st.file_uploader("Fichier CSV de relevés", type=["csv"])

if uploaded is not None:
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
