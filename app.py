"""
Dasbor Segmentasi Pelanggan - Telco Customer Churn
Metode: K-Means Clustering (fitur di-standardisasi), visualisasi PCA 2D.
Jalankan lokal: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentasi Pelanggan Telco", page_icon="📊", layout="wide")

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_FEATURES = {
    "tenure": "Lama berlangganan (bulan)",
    "MonthlyCharges": "Biaya bulanan ($)",
    "TotalCharges": "Total biaya ($)",
    "NumServices": "Jumlah layanan",
    "ContractMonths": "Durasi kontrak (bulan)",
}
COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------- 1. Data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges berisi spasi kosong untuk pelanggan tenure = 0 -> ubah ke numerik lalu hapus
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Feature engineering
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["ContractMonths"] = df["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24})
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data
def elbow_silhouette(X: np.ndarray, k_max: int = 10) -> pd.DataFrame:
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=42)
        rows.append({"k": k, "Inertia (WCSS)": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data
def run_kmeans(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=42)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return km.labels_, sil, coords, pca.explained_variance_ratio_


def name_segments(profile: pd.DataFrame) -> dict:
    """Beri nama segmen berdasarkan posisi rata-rata tenure & biaya bulanan terhadap median antar-klaster."""
    t_med, m_med = profile["tenure"].median(), profile["MonthlyCharges"].median()
    names, used = {}, {}
    for c, r in profile.iterrows():
        loyal = r["tenure"] >= t_med
        high = r["MonthlyCharges"] >= m_med
        if loyal and high:
            n = "Loyal Premium"
        elif loyal and not high:
            n = "Loyal Hemat"
        elif not loyal and high:
            n = "Baru Bernilai Tinggi"
        else:
            n = "Baru Basic"
        used[n] = used.get(n, 0) + 1
        names[c] = n if used[n] == 1 else f"{n} {used[n]}"
    return names


df_raw = load_data(DATA_PATH)

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("⚙️ Pengaturan")
features = st.sidebar.multiselect(
    "Fitur untuk clustering", list(NUM_FEATURES), default=list(NUM_FEATURES),
    format_func=lambda x: NUM_FEATURES[x],
)
if len(features) < 2:
    st.sidebar.error("Pilih minimal 2 fitur.")
    st.stop()
k = st.sidebar.slider("Jumlah segmen (k)", 2, 8, 4)
st.sidebar.markdown("---")
st.sidebar.caption("Variabel **Churn** tidak dipakai saat clustering; hanya untuk mengevaluasi profil tiap segmen.")

# ---------------------------------------------------------------- Modeling
X = StandardScaler().fit_transform(df_raw[features])
labels, sil, coords, evr = run_kmeans(X, k)
df = df_raw.copy()
df["Cluster"] = labels
profile_num = df.groupby("Cluster")[list(NUM_FEATURES)].mean()
seg_names = name_segments(profile_num)
df["Segmen"] = df["Cluster"].map(lambda c: f"S{c + 1} · {seg_names[c]}")
df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
seg_order = sorted(df["Segmen"].unique())
color_map = {s: COLORS[i % len(COLORS)] for i, s in enumerate(seg_order)}

# ---------------------------------------------------------------- Header
st.title("📊 Dasbor Segmentasi Pelanggan Telco")
st.markdown("Segmentasi **7.032 pelanggan** perusahaan telekomunikasi menggunakan **K-Means Clustering** "
            "berdasarkan perilaku berlangganan dan pengeluaran.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah pelanggan", f"{len(df):,}".replace(",", "."))
c2.metric("Jumlah segmen", k)
c3.metric("Silhouette score", f"{sil:.3f}")
c4.metric("Churn rate keseluruhan", f"{df['ChurnFlag'].mean():.1%}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1️⃣ Eksplorasi Data", "2️⃣ Penentuan k", "3️⃣ Hasil Segmentasi", "4️⃣ Profil Segmen", "5️⃣ Data & Unduh"])

# ---------------------------------------------------------------- Tab 1: EDA
with tab1:
    st.subheader("Gambaran dataset")
    a, b = st.columns([2, 1])
    with a:
        st.dataframe(df_raw.drop(columns=["ChurnFlag"]).head(10), width="stretch")
    with b:
        st.markdown(f"""
- **Baris awal:** 7.043 → **{len(df_raw):,}** setelah membuang 11 baris `TotalCharges` kosong (pelanggan baru, tenure 0)
- **Kolom asli:** 21
- **Fitur turunan:** `NumServices` (jumlah layanan aktif), `ContractMonths` (kontrak 1/12/24 bulan)
""".replace(f"{len(df_raw):,}", f"{len(df_raw):,}".replace(",", ".")))
    st.subheader("Statistik deskriptif fitur numerik")
    st.dataframe(df_raw[list(NUM_FEATURES)].describe().T.round(2), width="stretch")

    st.subheader("Distribusi fitur")
    col = st.selectbox("Pilih fitur", list(NUM_FEATURES), format_func=lambda x: NUM_FEATURES[x])
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df_raw, x=col, color="Churn", nbins=40, barmode="overlay",
                                labels={col: NUM_FEATURES[col]}, title=f"Histogram {NUM_FEATURES[col]} per status churn"),
                   width="stretch")
    b.plotly_chart(px.box(df_raw, x="Churn", y=col, color="Churn", labels={col: NUM_FEATURES[col]},
                          title=f"Boxplot {NUM_FEATURES[col]}"), width="stretch")

    st.subheader("Korelasi antar-fitur numerik")
    corr = df_raw[list(NUM_FEATURES) + ["ChurnFlag"]].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width="stretch")

# ---------------------------------------------------------------- Tab 2: k
with tab2:
    st.subheader("Metode Elbow & Silhouette")
    st.markdown("Fitur dinormalisasi dengan **StandardScaler** (z-score) sebelum K-Means agar skala "
                "biaya ($) tidak mendominasi tenure (bulan).")
    es = elbow_silhouette(X)
    a, b = st.columns(2)
    fig = px.line(es, x="k", y="Inertia (WCSS)", markers=True, title="Elbow Method")
    fig.add_vline(x=k, line_dash="dash", line_color="red")
    a.plotly_chart(fig, width="stretch")
    fig = px.line(es, x="k", y="Silhouette", markers=True, title="Silhouette Score")
    fig.add_vline(x=k, line_dash="dash", line_color="red")
    b.plotly_chart(fig, width="stretch")
    st.dataframe(es.round(3), width="stretch", hide_index=True)
    st.info("Kurva elbow mulai melandai di sekitar **k = 4**. Silhouette tertinggi ada di k = 2, tetapi dua "
            "segmen terlalu kasar untuk strategi bisnis; k = 4 memberi segmen yang masih terpisah cukup baik "
            "(silhouette ≈ 0,37) dan jauh lebih mudah ditindaklanjuti. Garis merah = k yang dipilih di sidebar.")

# ---------------------------------------------------------------- Tab 3: Segments
with tab3:
    st.subheader("Visualisasi klaster (PCA 2 dimensi)")
    st.caption(f"PC1 menjelaskan {evr[0]:.1%} dan PC2 {evr[1]:.1%} variasi data (total {evr.sum():.1%}).")
    fig = px.scatter(df, x="PC1", y="PC2", color="Segmen", opacity=0.6, category_orders={"Segmen": seg_order},
                     color_discrete_map=color_map, hover_data=["customerID", "tenure", "MonthlyCharges", "Contract"])
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    size = df["Segmen"].value_counts().reindex(seg_order).reset_index()
    size.columns = ["Segmen", "Jumlah"]
    a.plotly_chart(px.pie(size, names="Segmen", values="Jumlah", hole=0.45, color="Segmen",
                          color_discrete_map=color_map, title="Proporsi pelanggan per segmen"),
                   width="stretch")
    fx = st.selectbox("Sumbu X", list(NUM_FEATURES), index=0, format_func=lambda x: NUM_FEATURES[x])
    fy = st.selectbox("Sumbu Y", list(NUM_FEATURES), index=1, format_func=lambda x: NUM_FEATURES[x])
    b.plotly_chart(px.scatter(df, x=fx, y=fy, color="Segmen", opacity=0.5, color_discrete_map=color_map,
                              category_orders={"Segmen": seg_order},
                              labels={fx: NUM_FEATURES[fx], fy: NUM_FEATURES[fy]},
                              title="Sebaran segmen pada fitur asli"), width="stretch")

# ---------------------------------------------------------------- Tab 4: Profile
with tab4:
    st.subheader("Rata-rata karakteristik tiap segmen")
    prof = df.groupby("Segmen").agg(
        Jumlah=("customerID", "count"),
        Tenure=("tenure", "mean"),
        Biaya_Bulanan=("MonthlyCharges", "mean"),
        Total_Biaya=("TotalCharges", "mean"),
        Jumlah_Layanan=("NumServices", "mean"),
        Durasi_Kontrak=("ContractMonths", "mean"),
        Churn_Rate=("ChurnFlag", "mean"),
    ).reindex(seg_order)
    st.dataframe(prof.style.format({"Tenure": "{:.1f}", "Biaya_Bulanan": "${:.2f}", "Total_Biaya": "${:,.0f}",
                                    "Jumlah_Layanan": "{:.1f}", "Durasi_Kontrak": "{:.1f}", "Churn_Rate": "{:.1%}"})
                 .background_gradient(subset=["Churn_Rate"], cmap="Reds"), width="stretch")

    a, b = st.columns(2)
    # Radar chart (nilai dinormalisasi 0-1)
    norm = (profile_num - profile_num.min()) / (profile_num.max() - profile_num.min() + 1e-9)
    radar = go.Figure()
    for c in norm.index:
        s = f"S{c + 1} · {seg_names[c]}"
        vals = norm.loc[c].tolist()
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[NUM_FEATURES[f] for f in norm.columns] +
                                        [NUM_FEATURES[norm.columns[0]]], fill="toself", name=s,
                                        line_color=color_map[s]))
    radar.update_layout(title="Profil relatif segmen (0 = terendah, 1 = tertinggi)",
                        polar=dict(radialaxis=dict(range=[0, 1])))
    a.plotly_chart(radar, width="stretch")

    cr = prof.reset_index()
    fig = px.bar(cr, x="Segmen", y="Churn_Rate", color="Segmen", color_discrete_map=color_map,
                 text=cr["Churn_Rate"].map("{:.1%}".format), title="Churn rate per segmen")
    fig.add_hline(y=df["ChurnFlag"].mean(), line_dash="dash", annotation_text="rata-rata")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    b.plotly_chart(fig, width="stretch")

    st.subheader("Komposisi variabel kategorik per segmen")
    cat = st.selectbox("Pilih variabel", ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
                                          "TechSupport", "OnlineSecurity", "SeniorCitizen", "Partner",
                                          "Dependents", "gender"])
    ct = pd.crosstab(df["Segmen"], df[cat], normalize="index").reindex(seg_order).reset_index() \
        .melt(id_vars="Segmen", var_name=cat, value_name="Proporsi")
    fig = px.bar(ct, x="Segmen", y="Proporsi", color=cat, barmode="stack", text=ct["Proporsi"].map("{:.0%}".format))
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    st.subheader("💡 Interpretasi & rekomendasi strategi")
    tips = {
        "Loyal Premium": "Pelanggan lama dengan banyak layanan dan pengeluaran tertinggi. **Strategi:** program loyalitas/VIP, "
                         "prioritas layanan, tawarkan upgrade eksklusif; jaga agar tidak pindah ke kompetitor.",
        "Loyal Hemat": "Pelanggan lama, kontrak panjang, tetapi hanya memakai sedikit layanan (banyak yang tanpa internet). "
                       "Churn sangat rendah. **Strategi:** cross-selling paket internet/streaming dengan harga bundling.",
        "Baru Bernilai Tinggi": "Pelanggan relatif baru, biaya bulanan tinggi, mayoritas kontrak bulanan (Fiber optic). "
                                "Churn paling tinggi. **Strategi:** prioritas retensi—diskon migrasi ke kontrak 1–2 tahun, "
                                "gratis TechSupport/OnlineSecurity, perbaiki pengalaman onboarding.",
        "Baru Basic": "Pelanggan baru dengan paket dasar dan biaya rendah, kontrak bulanan. **Strategi:** edukasi manfaat layanan, "
                      "promo upgrade bertahap, dorong pembayaran otomatis untuk menekan churn.",
    }
    for c in sorted(seg_names):
        s = f"S{c + 1} · {seg_names[c]}"
        base = seg_names[c].rstrip(" 0123456789")
        r = prof.loc[s]
        n_fmt = f"{int(r.Jumlah):,}".replace(",", ".")
        st.markdown(f"**{s}** — {n_fmt} pelanggan, tenure {r.Tenure:.0f} bln, "
                    f"${r.Biaya_Bulanan:.0f}/bln, churn {r.Churn_Rate:.1%}  \n{tips.get(base, '')}")

# ---------------------------------------------------------------- Tab 5: Data
with tab5:
    st.subheader("Data pelanggan beserta label segmen")
    sel = st.multiselect("Filter segmen", seg_order, default=seg_order)
    churn_f = st.radio("Status churn", ["Semua", "Yes", "No"], horizontal=True)
    view = df[df["Segmen"].isin(sel)]
    if churn_f != "Semua":
        view = view[view["Churn"] == churn_f]
    cols = ["customerID", "Segmen", "tenure", "MonthlyCharges", "TotalCharges", "NumServices",
            "Contract", "InternetService", "PaymentMethod", "Churn"]
    st.caption(f"{len(view):,} baris".replace(",", "."))
    st.dataframe(view[cols], width="stretch", hide_index=True)
    st.download_button("⬇️ Unduh CSV hasil segmentasi", view[cols].to_csv(index=False).encode("utf-8"),
                       file_name="telco_segmentasi.csv", mime="text/csv")

st.markdown("---")
st.caption("Dataset: IBM Telco Customer Churn (Kaggle) · Metode: StandardScaler + K-Means + PCA · Dibuat dengan Streamlit")
