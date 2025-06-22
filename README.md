# Data Science – Comparateur de Modèles

Ce projet propose un pipeline complet d’analyse et de comparaison de modèles de régression à partir de données socio-démographiques. Il inclut un notebook d’analyse, la préparation des données, la création de modèles, et une application interactive Streamlit pour visualiser et comparer les performances.

## 📁 Structure du projet

- `main.ipynb` : Notebook principal d’analyse, préparation et feature engineering sur les données.
- `streamlit.py` : Application Streamlit interactive pour comparer divers modèles de régression.
- `data/` : Dossier contenant les jeux de données nécessaires à l’étude (`learn_model1.csv`, `learn_model2.csv`, `learn_model4.csv`, etc.).
- Fichiers modèles (`model1.pkl`, etc.) à déposer à la racine ou dans un dossier prévu.

## 🚀 Démarrage rapide

### Prérequis

- Python 3.8+
- pip install -r requirements.txt *(à générer selon vos besoins, voir exemple ci-dessous)*

Exemple :
```bash
pip install pandas numpy scikit-learn joblib streamlit plotly
```

### 1. Préparation des données

Placez les fichiers CSV nécessaires dans le dossier `/data` :
- `learn_model1.csv`
- `learn_model2.csv`
- `learn_model4.csv`
- Autres jeux de données référencés dans le notebook.

### 2. Explorations et feature engineering

Le notebook `main.ipynb` détaille :
- Les fusions de données (géographiques, population, etc.)
- Le regroupement intelligent des variables (âge, diplôme, catégorie socio-pro)
- Des analyses statistiques (corrélations, tests de Kolmogorov-Smirnov…)
- La préparation des datasets pour les modèles

### 3. Modélisation & comparaison interactive

Lancez l’application Streamlit :
```bash
streamlit run streamlit.py
```
- Comparez les performances de plusieurs modèles : Random Forest, Régression Linéaire, etc.
- Visualisez R², RMSE et des graphiques interactifs.
- Les modèles doivent être pré-entrainés et leurs fichiers `.pkl` présents.

## 📝 Exemple d’utilisation

1. Exécutez les cellules du notebook pour générer vos datasets et entraîner vos modèles.
2. Placez les modèles sauvegardés (`.pkl`) et les features associées au bon endroit.
3. Démarrez Streamlit, choisissez un modèle, explorez résultats et graphiques.

## 📊 Technologies principales

- Python, pandas, numpy
- scikit-learn, joblib (sauvegarde modèles)
- Streamlit (visualisation interactive)
- plotly (graphiques dynamiques)
- Jupyter Notebook
