import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Connexion")

    user = st.text_input("Utilisateur")
    pwd = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if user == "admin" and pwd == "1234":
            st.session_state["auth"] = True
        else:
            st.error("Identifiants incorrects")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IA Diabète", layout="centered")

# STYLE
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
h1 {color: #2c3e50; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- TITLE ----------------
st.title("🩺 Diagnostic du diabète intelligent")

st.markdown("Remplissez les informations du patient")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Grossesses", 0)
    glucose = st.number_input("Glucose", 0)
    bloodpressure = st.number_input("Pression", 0)
    age = st.number_input("Âge", 1)

with col2:
    skinthickness = st.number_input("Peau", 0)
    insulin = st.number_input("Insuline", 0)
    bmi = st.number_input("IMC", 0.0)
    dpf = st.number_input("DPF", 0.5)

# ---------------- BUTTON ----------------
if st.button("🔍 Diagnostiquer"):

    data = np.array([
        pregnancies, glucose, bloodpressure,
        skinthickness, insulin, bmi, dpf, age
    ]).reshape(1, -1)

    data = scaler.transform(data)

    proba = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]

    # ---------------- RESULT ----------------
    st.subheader("📊 Résultat")

    st.write(f"### Risque de diabète : {proba*100:.2f}%")

    # BAR CHART
    fig, ax = plt.subplots()
    ax.bar(["Sain", "Risque"], [1-proba, proba], color=["green", "red"])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # GAUGE SIMPLE (progress bar)
    st.progress(int(proba * 100))

    # MESSAGE
    if pred == 1:
        st.error("⚠️ Risque élevé de diabète")
    else:
        st.success("✅ Risque faible")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 IA éducative - ne remplace pas un médecin")