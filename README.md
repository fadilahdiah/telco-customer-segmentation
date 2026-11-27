# Dasbor Segmentasi Pelanggan Telco

Segmentasi pelanggan dataset **IBM Telco Customer Churn** dengan K-Means Clustering, ditampilkan sebagai dasbor Streamlit.

**Alur analisis**
1. Pembersihan data: `TotalCharges` → numerik, buang 11 baris kosong (7.032 pelanggan).
2. Feature engineering: `NumServices` (jumlah layanan aktif), `ContractMonths` (1/12/24).
3. Standardisasi (StandardScaler) fitur: tenure, MonthlyCharges, TotalCharges, NumServices, ContractMonths.
4. Penentuan k dengan Elbow & Silhouette → k = 4.
5. K-Means (random_state=42), visualisasi PCA 2D.
6. Profil segmen + churn rate tiap segmen (Churn tidak ikut clustering) dan rekomendasi strategi.

**Menjalankan lokal**
```bash
pip install -r requirements.txt
streamlit run app.py
```
