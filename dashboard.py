import streamlit as st

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Dashboard Klasifikasi & Deteksi",
    layout="wide"
)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <h2 style='text-align:center;'>Dashboard</h2>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Dashboard")
    st.markdown("### 🖼️ Klasifikasi Gambar")
    st.markdown("### ✉️ Object Detection")
    st.markdown("---")

    st.image("logo.png", caption="Universitas Syiah Kuala", use_column_width=True)

# ===================== MAIN CONTENT =====================
st.markdown("""
<h2 style="text-align:center;">
Dashboard Klasifikasi Gambar Pada Buah Buahan dan Deteksi Objek Terhadap Ikan Hias
</h2>
""", unsafe_allow_html=True)

st.write("")
st.markdown("""
<p style="text-align:center;">
Selamat datang di Dashboard Klasifikasi Gambar Pada Buah Buahan dan Deteksi Objek
Terhadap Ikan Hias. Dashboard ini memiliki dua fitur:
</p>
""", unsafe_allow_html=True)

# ===================== 2 COLUMN FEATURE =====================
col1, col2 = st.columns([1,1])

with col1:
    st.write(" ")
    st.write(" ")
    st.markdown("<h3 style='text-align:center;'>🖼️ Klasifikasi Gambar</h3>", unsafe_allow_html=True)
    st.write(" ")
    if st.button("Masuk ke Klasifikasi", use_container_width=True):
        st.switch_page("pages/klasifikasi.py")

with col2:
    st.write(" ")
    st.write(" ")
    st.markdown("<h3 style='text-align:center;'>✉️ Object Detection</h3>", unsafe_allow_html=True)
    st.write(" ")
    if st.button("Masuk ke Deteksi Objek", use_container_width=True):
        st.switch_page("pages/deteksi.py")
