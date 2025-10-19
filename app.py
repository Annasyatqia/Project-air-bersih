# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
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

# CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============ HEADER ============ #
st.title("💧 Analisis & Prediksi Ketersediaan Air Bersih Jawa Barat")
st.markdown("### Machine Learning Dashboard by Annasya Atqia")

# ============ SIDEBAR ============ #
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

# ============ TABS ============ #
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "⚙️ Preprocessing", "🧠 Modeling", "🔮 Prediksi"])

# ============ TAB 1: EDA ============ #
with tab1:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    st.write("### Preview Dataset")
    st.dataframe(df.head(10))
    
    st.write("### Statistik Deskriptif")
    st.dataframe(df.describe())
    
    st.write("### Distribusi Tahun")
    if "tahun" in df.columns:
        fig = px.histogram(df, x="tahun", color_discrete_sequence=["#0d6efd"], title="Distribusi Tahun")
        st.plotly_chart(fig)
    
    target_col = "ketersediaan_air_minum_sumber_kemasan"
    if target_col in df.columns:
        st.write("### Distribusi Target")
        fig2 = px.histogram(df, x=target_col, color=target_col, color_discrete_sequence=["#198754", "#dc3545"], title="Distribusi Target")
        st.plotly_chart(fig2)
        
        st.write("### Korelasi Heatmap")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        st.write("### Boxplot untuk Fitur Numerik")
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            selected_col = st.selectbox("Pilih kolom untuk boxplot:", num_cols)
            fig = px.box(df, y=selected_col, title=f"Boxplot {selected_col}")
            st.plotly_chart(fig)

# ============ TAB 2: PREPROCESSING ============ #
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
    if target_col in df_processed.columns:
        df_processed[target_col] = df_processed[target_col].astype(str).replace({'ADA': 1, 'TIDAK ADA': 0})
    
    # Scaling
    num_cols = df_processed.select_dtypes(include='number').columns
    if scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
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
    
    st.write("Data setelah preprocessing:")
    st.dataframe(df_processed.head())
    st.write(f"Jumlah fitur: {X.shape[1]}, Jumlah sampel: {X.shape[0]}")

# ============ TAB 3: MODELING ============ #
with tab3:
    st.subheader("🧠 Pilih Model Machine Learning")
    
    model_choice = st.radio("Pilih model:", ["Random Forest", "Gradient Boosting", "K-Means Clustering"])
    
    if model_choice == "Random Forest":
        st.markdown("### 🌳 Random Forest Classifier")
        n_estimators = st.slider("n_estimators:", 50, 300, 150)
        max_depth = st.slider("max_depth:", 5, 20, 10)
        
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
        
        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**F1-score:** {f1:.3f}")
        st.write(f"**ROC-AUC:** {roc:.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Ada', 'Ada'], yticklabels=['Tidak Ada', 'Ada'])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Feature Importance
        rf_model = pipeline.named_steps['rf']
        importance = rf_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance', ascending=False)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    
    elif model_choice == "Gradient Boosting":
        st.markdown("### ☀️ Gradient Boosting Classifier")
        n_estimators = st.slider("n_estimators:", 50, 300, 200)
        learning_rate = st.slider("learning_rate:", 0.01, 0.3, 0.1)
        max_depth = st.slider("max_depth:", 3, 10, 3)
        
        gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        y_proba = gb.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        st.write(f"**Accuracy:** {acc:.3f}")
        st.write(f"**F1-score:** {f1:.3f}")
        st.write(f"**ROC-AUC:** {roc:.3f}")
        
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Feature Importance
        importance = gb.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance', ascending=False)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    
    else:
        st.markdown("### 🔷 K-Means Clustering")
        k = st.slider("Jumlah cluster (k):", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        sil = silhouette_score(X, clusters)
        st.write(f"**Silhouette Score:** {sil:.3f}")
        
        df_processed['Cluster'] = clusters
        pca = PCA(2)
        pca_data = pca.fit_transform(X)
        df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters
        
        fig = px.scatter(df_pca, x='PC1', y='PC2', color=df_pca['Cluster'].astype(str),
                         title=f"Visualisasi PCA Clustering (k={k})",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig)
        
        # Cluster Summary
        cluster_summary = df_processed.groupby('Cluster').mean()
        st.write("Rata-rata per cluster:")
        st.dataframe(cluster_summary)

# ============ TAB 4: PREDIKSI ============ #
with tab4:
    st.subheader("🔮 Prediksi Baru")
    st.write("Masukkan nilai fitur untuk prediksi (gunakan nilai yang sudah di-scale jika perlu):")
    
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}:", value=0.0)
    
    input_df = pd.DataFrame([input_data])
    
    if st.button("Prediksi"):
        if model_choice in ["Random Forest", "Gradient Boosting"]:
            if model_choice == "Random Forest":
                pred = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0][1]
            else:
                pred = gb.predict(input_df)[0]
                proba = gb.predict_proba(input_df)[0][1]
            st.write(f"Prediksi: {'ADA' if pred == 1 else 'TIDAK ADA'}")
            st.write(f"Probabilitas ADA: {proba:.3f}")
        else:
            cluster = kmeans.predict(input_df)[0]
            st.write(f"Cluster: {cluster}")

st.success("✅ Analisis selesai! Silakan eksplorasi tab lainnya untuk lebih banyak fitur.")
