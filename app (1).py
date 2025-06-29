
import streamlit as st
import joblib

# Fungsi memuat model, hanya dieksekusi sekali
@st.cache_resource
def load_model():
    return joblib.load("spam_classifier.pkl")

model = load_model()

st.set_page_config(page_title="Deteksi Pesan Spam", page_icon="📨")

st.title("📨 Deteksi Pesan Spam Sederhana")
st.write(
    "Masukkan teks SMS atau email di bawah ini, lalu klik **Prediksi** untuk melihat apakah pesan tersebut spam."
)

# Input teks dari pengguna
user_text = st.text_area("Teks pesan", height=150)

# Tombol prediksi
if st.button("Prediksi") and user_text.strip():
    pred = model.predict([user_text])[0]
    prob = model.predict_proba([user_text])[0].max()
    if pred.lower() == "spam":
        label = "🚫 **SPAM**"
    else:
        label = "✅ **HAM (bukan spam)**"

    st.markdown(f"### Hasil: {label}")
    st.write(f"Tingkat keyakinan: **{prob:.2%}**")

st.caption("Model menggunakan TF‑IDF + Multinomial Naive Bayes.")
