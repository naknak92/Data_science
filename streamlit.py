import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

st.title("🔍 Comparateur de Modèles de Régression")

# --- Chargement des datasets ---
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/learn_model1.csv")
    df2 = pd.read_csv("data/learn_model2.csv")
    df4 = pd.read_csv("data/learn_model4.csv")
    return df1, df2, df4

# --- Chargement des modèles et des colonnes associées ---
@st.cache_resource
def load_models():
    model1 = joblib.load("model1.pkl")          # GridSearchCV (pas pipeline)
    model2_rf = joblib.load("model2.pkl")
    model2_reg = joblib.load("modele2_reg.pkl") # Pipeline complet
    model4 = joblib.load("model4.pkl")
    
    # Chargement des colonnes d'entraînement pour les modèles sans pipeline
    features1 = joblib.load("model1_features.pkl")
    return model1, model2_rf, model2_reg, model4, features1

# Données et modèles
learn_model1, learn_model2, learn_model4 = load_data()
model1, model2_rf, model2_reg, model4, features1 = load_models()

models_data = {
    "Modèle 1 - Random Forest": (learn_model1, model1, features1),
    "Modèle 2 - Random Forest": (learn_model2, model2_rf, None),
    "Modèle 2 - Régression Linéaire": (learn_model2, model2_reg, None),
    "Modèle 4 - Random Forest": (learn_model4, model4, None)
}

# Sélecteur
selected = st.selectbox("Sélectionnez un modèle :", list(models_data.keys()))
data, model, custom_features = models_data[selected]

st.subheader(f"Aperçu des données ({selected})")
st.write(data.head())

if 'target' not in data.columns:
    st.error("❌ Colonne 'target' absente du dataset.")
else:
    try:
        y = data["target"]

        # Cas des modèles avec pipeline complet
        if hasattr(model, "predict") and not hasattr(model, "best_estimator_"):
            X = data.drop(columns=["target", "UNIQUE_ID", "insee_code"], errors="ignore")
        
        # Cas des modèles GridSearchCV (sans pipeline)
        elif custom_features is not None:
            X = data[custom_features]
        else:
            X = data.drop(columns=["target", "UNIQUE_ID", "insee_code"], errors="ignore")

        # Prédictions
        y_pred = model.predict(X)

        # Scores
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.metric("📈 R² (sur tout le dataset)", round(r2, 3))
        st.metric("📉 RMSE", round(rmse, 2))

        # R² validation croisée (si disponible)
        if hasattr(model, "best_score_"):
            st.metric("✅ R² (validation croisée)", round(model.best_score_, 3))
        elif hasattr(model, "cross_val_r2"):
            st.metric("✅ R² (validation croisée)", round(model.cross_val_r2, 3))
        else:
            st.warning("Aucune validation croisée disponible pour ce modèle.")

        # Graphe interactif
        n = st.slider("Nombre de points à afficher :", min_value=50, max_value=min(len(y), 1000), value=100, step=50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y.values[:n], mode='lines+markers', name='Réel'))
        fig.add_trace(go.Scatter(y=y_pred[:n], mode='lines+markers', name='Prédit'))
        fig.update_layout(
            title="Comparaison Réel vs Prédit",
            xaxis_title="Index",
            yaxis_title="Valeur",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
