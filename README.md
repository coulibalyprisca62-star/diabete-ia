# 🩺 Déploiement d’un modèle IA avec Streamlit

## 📌 Description du projet

Ce projet consiste à développer une application web intelligente permettant de prédire le risque de diabète chez un patient à partir de données médicales.

Le modèle d’intelligence artificielle a été entraîné avec Python et Scikit-learn, puis déployé dans une application interactive avec Streamlit.

---

# 🎯 Objectifs

- Comprendre le cycle complet d’un projet IA
- Entraîner un modèle de Machine Learning
- Sauvegarder le modèle avec Joblib
- Construire une interface web avec Streamlit
- Déployer localement une application IA
- Réfléchir au déploiement Edge AI en contexte de faible connectivité

---

# 🧠 Technologies utilisées

- Python
- Scikit-learn
- Streamlit
- Joblib
- Numpy
- Pandas
- Matplotlib

---

# 📊 Dataset utilisé

Dataset Diabetes.

Variables utilisées :
- Glucose
- Pression artérielle
- IMC
- Insuline
- Âge
- etc.

Le modèle prédit :
- Risque élevé de diabète
- Risque faible

---

# ⚙️ Fonctionnement du projet

## 1. Entraînement du modèle

Le modèle est entraîné avec Scikit-learn.

## 2. Sauvegarde

Le modèle est sauvegardé avec :

```python
joblib.dump(model, "model.pkl")
