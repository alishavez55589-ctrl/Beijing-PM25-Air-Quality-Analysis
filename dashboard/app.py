# ─────────────────────────────────────────────────────────────────────────────
# Beijing Air Quality — Interactive Dashboard
# Phase 9: Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# ── Sidebar Branding ──────────────────────────────────────────────────────────
st.sidebar.title("🌫️ Beijing Air Quality")
st.sidebar.markdown("**Dataset:** Beijing PM2.5 Hourly Data")
st.sidebar.markdown("**Course:** Data Science & Analytics MCT-341L")
st.sidebar.markdown("**UET Lahore — Final Project**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("Use the tabs above to explore")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df          = pd.read_csv('data/cleaned_data.csv')
    cluster_df  = pd.read_csv('data/cluster_data.csv')
    return df, cluster_df

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg_model    = joblib.load('models/regression_model.pkl')
    cls_model    = joblib.load('models/classification_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    feature_names= joblib.load('models/feature_names.pkl')
    kmeans       = joblib.load('models/kmeans_model.pkl')
    pca          = joblib.load('models/pca_model.pkl')
    return reg_model, cls_model, scaler, feature_names, kmeans, pca

# Load everything
df, cluster_df             = load_data()
reg_model, cls_model, scaler, feature_names, kmeans, pca = load_models()

# ── Main Title ────────────────────────────────────────────────────────────────
st.title("🌫️ Beijing Air Quality — Interactive Dashboard")
st.markdown("Explore the dataset, run predictions, and discover pollution patterns.")
st.markdown("---")

# ── Create 4 Tabs ─────────────────────────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📊 Tab A — Dataset Overview",
    "🔍 Tab B — Exploratory Analysis",
    "🤖 Tab C — Model Prediction",
    "🔵 Tab D — Cluster Explorer"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB A — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.header("📊 Dataset Overview")
    st.markdown("Explore the cleaned Beijing PM2.5 dataset with 43,000+ hourly records.")

    # ── Metrics Row ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",     f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{df.shape[1]}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    col4.metric("Years Covered",  f"{df['year'].min()} – {df['year'].max()}")

    st.markdown("---")

    # ── Interactive Dataframe ─────────────────────────────────────────────────
    st.subheader("📋 Full Dataset (sortable & filterable)")
    st.dataframe(df, use_container_width=True, height=300)

    st.markdown("---")

    # ── Column Histogram ──────────────────────────────────────────────────────
    st.subheader("📈 Column Distribution")

    # Only show numeric columns in dropdown
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox(
        "Select a column to view its distribution:",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[selected_col].dropna(), bins=40, color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.set_xlabel(selected_col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of {selected_col}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"Histogram showing the frequency distribution of **{selected_col}** across all records.")

    st.markdown("---")

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.subheader("🔥 Correlation Heatmap")

    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax2, linewidths=0.5,
                annot_kws={"size": 8})
    ax2.set_title("Pearson Correlation Matrix — All Numeric Features", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("Lower triangle shows Pearson correlation between all numeric features. "
               "Red = strong positive, Blue = strong negative correlation.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB B — EXPLORATORY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_b:
    st.header("🔍 Exploratory Analysis")
    st.markdown("Create custom scatter plots to explore relationships between features.")

    # ── Feature Selection ─────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include='object').columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature:", numeric_cols, index=0)
    with col2:
        y_feature = st.selectbox("Select Y-axis feature:", numeric_cols, index=1)

    col3, col4 = st.columns(2)
    with col3:
        show_trend = st.checkbox("Show trend line", value=True)
    with col4:
        color_by = st.selectbox("Colour points by:", ["None"] + cat_cols)

    # ── Range Slider ──────────────────────────────────────────────────────────
    st.markdown("**Filter data range:**")
    col_min = float(df[x_feature].min())
    col_max = float(df[x_feature].max())
    x_range = st.slider(
        f"Filter {x_feature} range:",
        min_value=col_min,
        max_value=col_max,
        value=(col_min, col_max)
    )

    # Filter dataframe based on slider
    plot_df = df[(df[x_feature] >= x_range[0]) & (df[x_feature] <= x_range[1])]

    # ── Guard: same feature selected for X and Y ──────────────────────────────
    if x_feature == y_feature:
        st.warning("⚠️ Please select two different features for X and Y axes.")
    else:
        # Sample for speed if dataset is large
        if len(plot_df) > 5000:
            plot_df = plot_df.sample(5000, random_state=42)

        fig3, ax3 = plt.subplots(figsize=(9, 5))

        if color_by == "None":
            ax3.scatter(plot_df[x_feature], plot_df[y_feature],
                       alpha=0.3, s=8, color='steelblue')
        else:
            # Plot each category in a different colour
            for cat in plot_df[color_by].unique():
                mask = plot_df[color_by] == cat
                ax3.scatter(plot_df[mask][x_feature],
                           plot_df[mask][y_feature],
                           alpha=0.3, s=8, label=str(cat))
            ax3.legend(title=color_by, fontsize=8,
                      bbox_to_anchor=(1.01, 1), loc='upper left')

        # Trend line
        if show_trend:
            x_vals = plot_df[x_feature].values
            y_vals = plot_df[y_feature].values
            mask_finite = np.isfinite(x_vals) & np.isfinite(y_vals)
            z = np.polyfit(x_vals[mask_finite], y_vals[mask_finite], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals[mask_finite].min(),
                                 x_vals[mask_finite].max(), 100)
            ax3.plot(x_line, p(x_line), color='crimson',
                    linewidth=2, linestyle='--', label='Trend line')
            ax3.legend(fontsize=9)

        ax3.set_xlabel(x_feature, fontsize=12)
        ax3.set_ylabel(y_feature, fontsize=12)
        ax3.set_title(f"{x_feature} vs {y_feature}", fontsize=13)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        st.caption(f"Scatter plot of **{x_feature}** vs **{y_feature}** "
                  f"showing {len(plot_df):,} data points.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB C — MODEL PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_c:
    st.header("🤖 Model Prediction")
    st.markdown("Enter meteorological values below and click **Predict** to get "
                "pm2.5 concentration and air quality category.")

    st.markdown("---")

    # ── Input Sliders ─────────────────────────────────────────────────────────
    st.subheader("🎛️ Input Meteorological Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        year         = st.number_input("Year",        min_value=2010, max_value=2015, value=2013)
        month        = st.number_input("Month",       min_value=1,    max_value=12,   value=6)
        day          = st.number_input("Day",         min_value=1,    max_value=31,   value=15)
        hour         = st.number_input("Hour",        min_value=0,    max_value=23,   value=12)
        DEWP         = st.slider("Dew Point (°C)",    min_value=-40,  max_value=30,   value=0)
        TEMP         = st.slider("Temperature (°C)",  min_value=-20,  max_value=42,   value=15)

    with col2:
        PRES              = st.slider("Pressure (hPa)",    min_value=990,  max_value=1045, value=1013)
        Iws               = st.slider("Wind Speed (m/s)",  min_value=0,    max_value=100,  value=10)
        Is                = st.slider("Snow (hours)",       min_value=0,    max_value=27,   value=0)
        Ir                = st.slider("Rain (hours)",       min_value=0,    max_value=36,   value=0)
        is_heating_season = st.selectbox("Heating Season?", [0, 1],
                                          format_func=lambda x: "Yes" if x==1 else "No")

    with col3:
        temp_dewp_gap  = TEMP - DEWP
        wind_pressure  = Iws * PRES
        precip_total   = Is + Ir
        wind_NE        = st.selectbox("Wind Direction NE?", [0, 1],
                                       format_func=lambda x: "Yes" if x==1 else "No")
        wind_NW        = st.selectbox("Wind Direction NW?", [0, 1],
                                       format_func=lambda x: "Yes" if x==1 else "No")
        wind_SE        = st.selectbox("Wind Direction SE?", [0, 1],
                                       format_func=lambda x: "Yes" if x==1 else "No")
        wind_cv        = st.selectbox("Wind Direction CV?", [0, 1],
                                       format_func=lambda x: "Yes" if x==1 else "No")

        st.info(f"**Auto-calculated:**\n\n"
                f"Temp-Dewp Gap: {temp_dewp_gap}°C\n\n"
                f"Wind×Pressure: {wind_pressure:.1f}\n\n"
                f"Total Precip: {precip_total} hrs")

    st.markdown("---")

    # ── Predict Button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Air Quality", use_container_width=True):

        # Build input dataframe matching exact feature order
        input_dict = {
            'year'             : year,
            'month'            : month,
            'day'              : day,
            'hour'             : hour,
            'DEWP'             : DEWP,
            'TEMP'             : TEMP,
            'PRES'             : PRES,
            'Iws'              : Iws,
            'Is'               : Is,
            'Ir'               : Ir,
            'season'           : month % 12 // 3 + 1,  # approximate season
            'time_of_day'      : hour,
            'wind_NE'          : wind_NE,
            'wind_NW'          : wind_NW,
            'wind_SE'          : wind_SE,
            'wind_cv'          : wind_cv,
            'temp_dewp_gap'    : temp_dewp_gap,
            'wind_pressure'    : wind_pressure,
            'precip_total'     : precip_total,
            'is_heating_season': is_heating_season
        }

        # Create dataframe with exact feature names in correct order
        input_df = pd.DataFrame([input_dict])[feature_names]

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # ── Regression Prediction ─────────────────────────────────────────────
        pm25_pred = reg_model.predict(input_scaled)[0]
        pm25_pred = max(0, pm25_pred)   # pm2.5 cannot be negative

        # ── Classification Prediction ─────────────────────────────────────────
        class_pred  = cls_model.predict(input_scaled)[0]
        class_proba = cls_model.predict_proba(input_scaled)[0]
        class_names = cls_model.classes_

        # ── Show Results ──────────────────────────────────────────────────────
        st.markdown("### 📊 Prediction Results")

        col_r, col_c = st.columns(2)

        with col_r:
            st.markdown("**🔢 Regression — PM2.5 Concentration**")
            if pm25_pred <= 35:
                st.success(f"🟢 Predicted PM2.5: **{pm25_pred:.1f} µg/m³**")
            elif pm25_pred <= 75:
                st.warning(f"🟡 Predicted PM2.5: **{pm25_pred:.1f} µg/m³**")
            elif pm25_pred <= 115:
                st.warning(f"🟠 Predicted PM2.5: **{pm25_pred:.1f} µg/m³**")
            else:
                st.error(f"🔴 Predicted PM2.5: **{pm25_pred:.1f} µg/m³**")

        with col_c:
            st.markdown("**🏷️ Classification — Air Quality Category**")
            if class_pred == 'Good':
                st.success(f"🟢 Predicted Class: **{class_pred}**")
            elif class_pred == 'Moderate':
                st.warning(f"🟡 Predicted Class: **{class_pred}**")
            elif class_pred in ['Unhealthy_Sensitive', 'Unhealthy']:
                st.warning(f"🟠 Predicted Class: **{class_pred}**")
            else:
                st.error(f"🔴 Predicted Class: **{class_pred}**")

        # ── Probability Bar Chart ─────────────────────────────────────────────
        st.markdown("**📊 Class Probabilities:**")
        proba_df = pd.DataFrame({
            'Air Quality Class': class_names,
            'Probability'      : np.round(class_proba, 3)
        }).sort_values('Probability', ascending=False)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        colors_bar = ['green' if c == class_pred else 'steelblue'
                      for c in proba_df['Air Quality Class']]
        ax4.barh(proba_df['Air Quality Class'],
                 proba_df['Probability'], color=colors_bar)
        ax4.set_xlabel('Probability', fontsize=12)
        ax4.set_title('Predicted Class Probabilities\n'
                      '(Green = predicted class)', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        st.caption("Bar chart shows the model's confidence for each air quality category.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB D — CLUSTER EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab_d:
    st.header("🔵 Cluster Explorer")
    st.markdown("Explore the 6 pollution clusters discovered by K-Means clustering.")

    # ── Cluster labels ────────────────────────────────────────────────────────
    cluster_labels_d = cluster_df['cluster'].values
    unique_clusters  = sorted(cluster_df['cluster'].unique())

    # ── PCA for visualisation ─────────────────────────────────────────────────
    feature_cols = [c for c in cluster_df.columns
                    if c not in ['cluster', 'air_quality', 'pm2.5']]
    X_cluster_sc = scaler.transform(
        cluster_df[feature_names]
    )
    X_pca_d = pca.transform(X_cluster_sc)

    # ── Cluster selector ──────────────────────────────────────────────────────
    selected_cluster = st.selectbox(
        "Select a cluster to highlight:",
        unique_clusters,
        format_func=lambda x: f"Cluster {x}"
    )

    # ── PCA Scatter Plot ──────────────────────────────────────────────────────
    colors_d = ['steelblue','crimson','green','orange','purple','brown']

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    for i in unique_clusters:
        mask  = cluster_labels_d == i
        color = colors_d[i]
        alpha = 0.6 if i == selected_cluster else 0.1   # highlight selected
        size  = 15  if i == selected_cluster else 5

        ax5.scatter(X_pca_d[mask, 0], X_pca_d[mask, 1],
                   c=color, alpha=alpha, s=size,
                   label=f'Cluster {i}' + (' ← selected' if i == selected_cluster else ''))

    # Centroids
    centroids_pca_d = pca.transform(kmeans.cluster_centers_)
    for i in unique_clusters:
        ax5.scatter(centroids_pca_d[i, 0], centroids_pca_d[i, 1],
                   c=colors_d[i], marker='*', s=400,
                   edgecolors='black', linewidths=0.8, zorder=5)

    ax5.set_xlabel('PCA Component 1', fontsize=12)
    ax5.set_ylabel('PCA Component 2', fontsize=12)
    ax5.set_title(f'K-Means Clusters — Cluster {selected_cluster} Highlighted\n'
                  'Stars = centroids | Bright = selected cluster', fontsize=13)
    ax5.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    st.caption(f"Cluster {selected_cluster} shown in full colour. "
               "All other clusters are greyed out for comparison.")

    st.markdown("---")

    # ── Cluster vs Overall Comparison ─────────────────────────────────────────
    st.subheader(f"📊 Cluster {selected_cluster} vs Overall Dataset")

    numeric_cluster_cols = ['pm2.5', 'TEMP', 'DEWP', 'PRES',
                            'Iws', 'is_heating_season', 'precip_total']

    # Mean of selected cluster
    cluster_mean  = cluster_df[cluster_df['cluster'] == selected_cluster][numeric_cluster_cols].mean()
    # Mean of full dataset
    overall_mean  = cluster_df[numeric_cluster_cols].mean()

    # Build comparison table
    compare_table = pd.DataFrame({
        f'Cluster {selected_cluster} Mean': cluster_mean.round(2),
        'Overall Mean'                     : overall_mean.round(2),
        'Delta (Cluster − Overall)'        : (cluster_mean - overall_mean).round(2)
    })

    st.dataframe(compare_table, use_container_width=True)
    st.caption("Table shows how this cluster's average feature values compare to the overall dataset.")

    # ── Delta Bar Chart ───────────────────────────────────────────────────────
    delta = cluster_mean - overall_mean

    fig6, ax6 = plt.subplots(figsize=(9, 5))
    bar_colors = ['steelblue' if v >= 0 else 'crimson' for v in delta.values]
    ax6.barh(delta.index, delta.values, color=bar_colors)
    ax6.axvline(0, color='black', linewidth=0.8)
    ax6.set_xlabel('Delta (Cluster Mean − Overall Mean)', fontsize=12)
    ax6.set_title(f'What makes Cluster {selected_cluster} distinctive?\n'
                  'Blue = above average | Red = below average', fontsize=13)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig6)
    st.caption(f"Features where Cluster {selected_cluster} differs most from "
               "the overall dataset average.")