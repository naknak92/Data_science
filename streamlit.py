import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

st.title("🔍 Comparateur de Modèles Random Forest (avec validation croisée)")

# Chargement des datasets et modèles
@st.cache_data
def load_data():
    df1 = pd.read_csv("learn_model1.csv")
    df2 = pd.read_csv("learn_model2.csv")
    df4 = pd.read_csv("learn_model4.csv")
    return df1, df2, df4

@st.cache_resource
def load_models():
    model1 = joblib.load("model1.pkl")  # GridSearchCV
    model2 = joblib.load("model2.pkl")
    model4 = joblib.load("model4.pkl")
    return model1, model2, model4

# Données et modèles
learn_model1, learn_model2, learn_model4 = load_data()
model1, model2, model4 = load_models()

models_data = {
    "learn_model1": (learn_model1, model1),
    "learn_model2": (learn_model2, model2),
    "learn_model4": (learn_model4, model4)
}

# Sélection du modèle
selected = st.selectbox("Sélectionnez le modèle :", list(models_data.keys()))
data, model = models_data[selected]

st.subheader(f"Aperçu des données - {selected}")
st.write(data.head())

if 'target' not in data.columns:
    st.error("Colonne 'target' absente du dataset.")
else:
    try:
        # Colonnes utilisées pendant l'entraînement
        model_features = model.best_estimator_.feature_names_in_
        X = data[model_features]
        y = data['target']

        # Prédictions
        y_pred = model.best_estimator_.predict(X)
        r2_full = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2_cv = model.best_score_

        # Affichage des scores
        st.metric("MSE (sur tout le dataset)", round(mse, 2))
        st.metric("R² (sur tout le dataset)", round(r2_full, 3))
        st.metric("R² (validation croisée)", round(r2_cv, 3))

        # Sélecteur dynamique pour le graphe
        n = st.slider("Nombre de points à afficher dans le graphe :", min_value=50, max_value=min(len(y), 1000), value=100, step=50)

        # Graphe interactif Plotly
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
