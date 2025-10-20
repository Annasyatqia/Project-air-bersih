# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ================= CSS ================= #
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================= HEADER ================= #
st.title("💧 Analisis & Prediksi Ketersediaan Air Bersih Jawa Barat")
st.markdown("### Machine Learning Dashboard by Annasya Atqia")
st.markdown("---")

# ================= SIDEBAR ================= #
st.sidebar.title("🔧 Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])

@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        try:
            return pd.read_csv("data/air_bersih_jabar.csv")
        except FileNotFoundError:
            st.error("Dataset default tidak ditemukan. Silakan upload file CSV.")
            return pd.DataFrame()

df = load_data(uploaded_file)
if df.empty:
    st.stop()

st.sidebar.markdown("---")
scaler_choice = st.sidebar.selectbox("Pilih Scaler:", ["MinMaxScaler", "StandardScaler"])
handle_missing = st.sidebar.selectbox("Penanganan Missing Values:", ["Median", "Mean", "Drop"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:** Gunakan dataset dengan kolom numerik untuk fitur dan kolom target 'ketersediaan_air_minum_sumber_kemasan'.")

# ================= TABS ================= #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "⚙️ Preprocessing", "🧠 Modeling", "🔮 Prediksi", "🗺️ Peta Jawa Barat"
])

# ================= TAB 1: EDA ================= #
with tab1:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Preview Dataset")
        st.dataframe(df.head(10))
    with col2:
        st.write("### Statistik Deskriptif")
        st.dataframe(df.describe())
    
    if "tahun" in df.columns:
        st.markdown("---")
        st.write("### Distribusi Tahun")
        fig = px.histogram(df, x="tahun", color_discrete_sequence=["#0d6efd"], title="Distribusi Tahun")
        st.plotly_chart(fig, use_container_width=True)
    
    target_col = "ketersediaan_air_minum_sumber_kemasan"
    if target_col in df.columns:
        st.markdown("---")
        st.write("### Distribusi Target")
        fig2 = px.histogram(df, x=target_col, color=target_col,
                            color_discrete_sequence=["#198754", "#dc3545"], title="Distribusi Target")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.write("### Korelasi Heatmap")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
            st.pyplot(fig)
        
        st.markdown("---")
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            selected_col = st.selectbox("Pilih kolom untuk boxplot:", num_cols)
            fig = px.box(df, y=selected_col, title=f"Boxplot {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tambahan: Scatter plot untuk korelasi tinggi
        st.markdown("---")
        st.write("### Scatter Plot Korelasi Tinggi")
        high_corr_pairs = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool)).stack().nlargest(3).index.tolist()
        if high_corr_pairs:
            for pair in high_corr_pairs:
                fig = px.scatter(df, x=pair[0], y=pair[1], title=f"Scatter Plot {pair[0]} vs {pair[1]}")
                st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: PREPROCESSING ================= #
with tab2:
    st.subheader("⚙️ Data Preprocessing")
    
    # Handle missing values
    if handle_missing == "Median":
        df_processed = df.fillna(df.median(numeric_only=True))
    elif handle_missing == "Mean":
        df_processed = df.fillna(df.mean(numeric_only=True))
    else:
        df_processed = df.dropna()
    
    # Encode target
    target_col = "ketersediaan_air_minum_sumber_kemasan"
    if target_col in df_processed.columns:
        df_processed[target_col] = df_processed[target_col].astype(str).replace({'ADA': 1, 'TIDAK ADA': 0})
    
    # Scaling
    num_cols = df_processed.select_dtypes(include='number').columns
    scaler = MinMaxScaler() if scaler_choice == "MinMaxScaler" else StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    
    # Split data
    exclude_cols = ['id', 'kode_wilayah', 'nama_wilayah']
    available_exclude = [c for c in exclude_cols if c in df_processed.columns]
    X = df_processed.select_dtypes(include='number').drop(columns=[target_col] + available_exclude, errors='ignore')
    y = df_processed[target_col] if target_col in df_processed.columns else None
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, None, None
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Data Setelah Preprocessing")
        st.dataframe(df_processed.head())
    with col2:
        st.write("### Info Dataset")
        st.write(f"Jumlah fitur: {X.shape[1]}")
        st.write(f"Jumlah sampel: {X.shape[0]}")
        if y is not None:
            st.write(f"Distribusi target: {y.value_counts().to_dict()}")

# ================= TAB 3: MODELING ================= #
with tab3:
    st.subheader("🧠 Pilih Model Machine Learning")
    model_choice = st.radio("Pilih model:", ["Random Forest", "Gradient Boosting", "K-Means Clustering"])
    
    if model_choice == "Random Forest":
        st.markdown("### 🌳 Random Forest Classifier")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators:", 50, 300, 150)
            max_depth = st.slider("max_depth:", 5, 20, 10)
        with col2:
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        with st.spinner("Melatih model..."):
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        roc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("F1-score", f"{f1:.3f}")
        with col3:
            st.metric("ROC-AUC", f"{roc:.3f}")
        st.write(f"**Cross-validation Accuracy (mean):** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        st.markdown("---")
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tidak Ada', 'Ada'], yticklabels=['Tidak Ada', 'Ada'])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("### Feature Importance")
        rf_model = pipeline.named_steps['rf']
        importance = rf_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance', ascending=False)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Tidak Ada', 'Ada'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    elif model_choice == "Gradient Boosting":
        st.markdown("### ☀️ Gradient Boosting Classifier")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators:", 50, 300, 200)
            learning_rate = st.slider("learning_rate:", 0.01, 0.3, 0.1)
        with col2:
            max_depth = st.slider("max_depth:", 3, 10, 3)
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        with st.spinner("Melatih model..."):
            gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            y_proba = gb.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(gb, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("F1-score", f"{f1:.3f}")
        with col3:
            st.metric("ROC-AUC", f"{roc:.3f}")
        st.write(f"**Cross-validation Accuracy (mean):** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        st.markdown("---")
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("### Feature Importance")
        importance = gb.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance', ascending=False)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Tidak Ada', 'Ada'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    else:
        st.markdown("### 🔷 K-Means Clustering")
        k = st.slider("Jumlah cluster (k):", 2, 10, 3)
        
        with st.spinner("Melakukan clustering..."):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            sil = silhouette_score(X, clusters)
        
        st.metric("Silhouette Score", f"{sil:.3f}")
        
        df_processed['Cluster'] = clusters
        pca = PCA(2)
        pca_data = pca.fit_transform(X)
        df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters
        
        st.markdown("---")
        st.write("### Visualisasi PCA Clustering")
        fig = px.scatter(df_pca, x='PC1', y='PC2', color=df_pca['Cluster'].astype(str),
                         title=f"Visualisasi PCA Clustering (k={k})",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("### Rata-rata per Cluster")
        cluster_summary = df_processed.groupby('Cluster').mean()
        st.dataframe(cluster_summary)

# ================= TAB 4: PREDIKSI ================= #
with tab4:
    st.subheader("🔮 Prediksi Baru")
    st.write("Masukkan nilai fitur untuk prediksi (gunakan nilai yang sudah di-scale jika perlu):")
    
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            input_data[col] = st.number_input(f"{col}:", value=0.0, step=0.01)
    input_df = pd.DataFrame([input_data])
    
    if st.button("Prediksi"):
        with st.spinner("Memproses prediksi..."):
            if model_choice in ["Random Forest", "Gradient Boosting"]:
                if model_choice == "Random Forest":
                    pred = pipeline.predict(input_df)[0]
                    proba = pipeline.predict_proba(input_df)[0][1]
                else:
                    pred = gb.predict(input_df)[0]
                    proba = gb.predict_proba(input_df)[0][1]
                st.success(f"Prediksi: {'ADA' if pred == 1 else 'TIDAK ADA'}")
                st.info(f"Probabilitas ADA: {proba:.3f}")
            else:
                cluster = kmeans.predict(input_df)[0]
                st.success(f"Cluster: {cluster}")

# ================= TAB 5: PETA JAWA BARAT ================= #
with tab5:
    st.subheader("🗺️ Visualisasi Peta Jawa Barat")
    st.write("Peta menampilkan distribusi ketersediaan air bersih berdasarkan hasil prediksi model.")

    try:
        with open("data/indonesia-edit.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Filter GeoJSON agar hanya menampilkan provinsi Jawa Barat
        geojson_jabar = {
            "type": "FeatureCollection",
            "features": [
                feat for feat in geojson_data["features"]
                if "Jawa Barat" in feat["properties"].get("Propinsi", "")
            ]
        }

        # Pastikan ada kolom wilayah di data hasil prediksi
        if "nama_wilayah" in df_processed.columns:
            df_map = df_processed.copy()
            df_map["nama_wilayah"] = df_map["nama_wilayah"].str.title()
        else:
            st.warning("Kolom 'nama_wilayah' tidak ditemukan, gunakan kolom sesuai data Anda.")
            df_map = df_processed

        # Buat peta interaktif dengan Plotly
        fig = px.choropleth_mapbox(
            df_map,
            geojson=geojson_jabar,
            locations="nama_wilayah",
            featureidkey="properties.Kabupaten",
            color="ketersediaan_air_minum_sumber_kemasan",
            color_continuous_scale="RdPu",
            mapbox_style="carto-positron",
            zoom=7,
            center={"lat": -6.9, "lon": 107.6},
            opacity=0.6,
            title="Distribusi Ketersediaan Air Bersih di Provinsi Jawa Barat"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Gagal memuat peta Jawa Barat: {e}")
