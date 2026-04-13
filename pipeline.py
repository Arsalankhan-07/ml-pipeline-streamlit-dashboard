"""
╔══════════════════════════════════════════════════════════════════╗
║   Interactive Machine Learning Pipeline Dashboard               ║
║   CS-303B | Machine Learning & ANN | CA-2 Project               ║
║   Production-Ready Streamlit Application                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import io, pickle, base64, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, mutual_info_regression,
)
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Set2

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def init_state():
    defaults = dict(
        dark_mode=True, df_raw=None, df_clean=None, target_col=None,
        problem_type="Classification", selected_features=None,
        X_train=None, X_test=None, y_train=None, y_test=None,
        trained_model=None, scaler=None, le_dict={}, active_step=1,
        model_name=None, model_params={}, cv_scores=None,
        tuned_model=None, metrics_before=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CSS INJECTION                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def inject_css(dark: bool):
    if dark:
        bg="rgba(13,17,23,0.0)"; surface="#161B22"; card="#21262D"
        border="#30363D"; text="#E6EDF3"; muted="#8B949E"
        accent="#58A6FF"; accent2="#3FB950"; accent3="#FF7B72"
        accent4="#D2A8FF"; grad1="#1F6FEB"; grad2="#238636"
    else:
        bg="rgba(246,248,250,0.0)"; surface="#FFFFFF"; card="#F0F2F5"
        border="#D0D7DE"; text="#1F2328"; muted="#656D76"
        accent="#0969DA"; accent2="#1A7F37"; accent3="#CF222E"
        accent4="#8250DF"; grad1="#0969DA"; grad2="#1A7F37"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
:root{{--bg:{bg};--surface:{surface};--card:{card};--border:{border};
      --text:{text};--muted:{muted};--accent:{accent};--accent2:{accent2};
      --accent3:{accent3};--accent4:{accent4};--grad1:{grad1};--grad2:{grad2};}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;color:var(--text);}}
.main .block-container{{padding:1.5rem 2rem 3rem;max-width:1400px;}}
section[data-testid="stSidebar"]{{background:var(--surface);border-right:1px solid var(--border);}}
section[data-testid="stSidebar"] .block-container{{padding:1.5rem 1rem;}}

/* Hero */
.hero-banner{{background:linear-gradient(135deg,var(--grad1) 0%,var(--grad2) 100%);
  border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.2);}}
.hero-banner::before{{content:'';position:absolute;top:-40%;right:-10%;
  width:350px;height:350px;background:rgba(255,255,255,0.06);border-radius:50%;}}
.hero-title{{font-family:'Sora',sans-serif;font-size:1.9rem;font-weight:800;
  color:#fff;margin:0 0 0.3rem;letter-spacing:-0.5px;}}
.hero-sub{{font-size:0.9rem;color:rgba(255,255,255,0.82);margin:0;}}
.hero-badge{{display:inline-block;background:rgba(255,255,255,0.18);
  border:1px solid rgba(255,255,255,0.3);border-radius:20px;
  padding:0.2rem 0.75rem;font-size:0.72rem;color:#fff;
  font-family:'JetBrains Mono',monospace;margin-bottom:0.6rem;}}

/* Stepper */
.stepper{{display:flex;gap:0.35rem;flex-wrap:wrap;margin-bottom:1.5rem;}}
.step-pill{{display:flex;align-items:center;gap:0.35rem;padding:0.3rem 0.75rem;
  border-radius:50px;font-size:0.75rem;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);
  color:var(--muted);white-space:nowrap;}}
.step-pill.active{{background:var(--accent);border-color:var(--accent);
  color:#fff;box-shadow:0 2px 12px rgba(88,166,255,0.35);}}
.step-pill.done{{background:var(--accent2);border-color:var(--accent2);color:#fff;}}
.step-num{{width:17px;height:17px;border-radius:50%;background:rgba(255,255,255,0.22);
  display:inline-flex;align-items:center;justify-content:center;font-size:0.62rem;font-weight:700;}}

/* Cards / Metric tiles */
.metric-tile{{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:1rem;text-align:center;}}
.metric-val{{font-family:'JetBrains Mono',monospace;font-size:1.55rem;
  font-weight:700;color:var(--accent);margin-bottom:0.2rem;}}
.metric-label{{font-size:0.7rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.6px;font-weight:600;}}
.info-box{{background:rgba(88,166,255,0.08);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:0.75rem 1rem;font-size:0.84rem;margin-bottom:0.75rem;}}
.success-box{{background:rgba(63,185,80,0.1);border-left:3px solid var(--accent2);
  border-radius:0 8px 8px 0;padding:0.75rem 1rem;font-size:0.84rem;margin-bottom:0.75rem;}}
.warn-box{{background:rgba(255,123,114,0.1);border-left:3px solid var(--accent3);
  border-radius:0 8px 8px 0;padding:0.75rem 1rem;font-size:0.84rem;margin-bottom:0.75rem;}}

/* Streamlit widget tweaks */
.stTabs [data-baseweb="tab-list"]{{gap:0.4rem;border-bottom:2px solid var(--border);}}
.stTabs [data-baseweb="tab"]{{border-radius:8px 8px 0 0;padding:0.45rem 1.1rem;
  font-weight:600;font-size:0.82rem;color:var(--muted);border:none;background:transparent;}}
.stTabs [aria-selected="true"]{{color:var(--accent)!important;
  border-bottom:2px solid var(--accent)!important;background:var(--card)!important;}}
.stButton>button{{font-family:'Inter',sans-serif;font-weight:600;font-size:0.84rem;
  border-radius:8px;border:1.5px solid var(--border);background:var(--surface);
  color:var(--text);transition:all 0.2s;}}
.stButton>button:hover{{border-color:var(--accent);color:var(--accent);}}
.stDownloadButton>button{{background:linear-gradient(135deg,var(--grad1),var(--grad2));
  color:white!important;border:none!important;border-radius:8px;font-weight:600;}}
</style>""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
def info(msg):    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)
def success(msg): st.markdown(f'<div class="success-box">✅ {msg}</div>', unsafe_allow_html=True)
def warn(msg):    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)

def metric_tiles(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, val) in zip(cols, metrics.items()):
        with col:
            st.markdown(f'<div class="metric-tile"><div class="metric-val">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def get_download_link(obj, filename, label):
    buf = io.BytesIO(); pickle.dump(obj, buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<a href="data:file/pkl;base64,{b64}" download="{filename}">'
            f'<button style="background:linear-gradient(135deg,#1F6FEB,#238636);'
            f'color:#fff;border:none;padding:0.45rem 1.2rem;border-radius:8px;'
            f'font-weight:600;cursor:pointer;">⬇ {label}</button></a>')

def plotly_theme(dark): return "plotly_dark" if dark else "plotly_white"

def build_model(name, params, pt):
    name = name or ""
    if name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=1000)
    elif name == "Linear Regression":
        return LinearRegression()
    elif name == "SVM":
        return SVC(**params) if pt == "Classification" else SVR(**params)
    elif name == "Random Forest":
        return (RandomForestClassifier(**params, random_state=42)
                if pt == "Classification"
                else RandomForestRegressor(**params, random_state=42))
    elif name == "KMeans (Clustering)":
        return KMeans(**params, random_state=42, n_init="auto")
    return None

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:800;
    background:linear-gradient(135deg,#58A6FF,#3FB950);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🧠 ML Pipeline
  </div>
  <div style="font-size:0.7rem;color:#8B949E;margin-top:0.2rem;">CS-303B · CA-2 Project</div>
</div>""", unsafe_allow_html=True)
        st.markdown("---")

        dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown("---")
        st.markdown("**🎯 Problem Type**")
        prob = st.selectbox("Select task", ["Classification", "Regression"],
                            index=0 if st.session_state.problem_type=="Classification" else 1,
                            label_visibility="collapsed")
        st.session_state.problem_type = prob
        col = "#58A6FF" if prob=="Classification" else "#3FB950"
        st.markdown(f'<div style="background:{col}22;border:1px solid {col}55;border-radius:8px;'
                    f'padding:0.45rem 0.75rem;font-size:0.78rem;color:{col};font-weight:600;margin-top:0.3rem;">'
                    f'{"🔵 Classification Task" if prob=="Classification" else "📈 Regression Task"}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**📋 Pipeline Steps**")
        steps = [("1","Data Input"),("2","EDA"),("3","Preprocessing"),
                 ("4","Outlier Detection"),("5","Feature Selection"),("6","Data Split"),
                 ("7","Model Selection"),("8","Training & CV"),("9","Metrics"),("10","HP Tuning")]
        active = st.session_state.active_step
        for num, name in steps:
            n = int(num)
            icon = "✅" if n < active else ("▶" if n == active else "○")
            color = "#3FB950" if n < active else ("#58A6FF" if n == active else "#8B949E")
            weight = "700" if n == active else "400"
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.45rem;'
                        f'padding:0.22rem 0;font-size:0.79rem;color:{color};font-weight:{weight};">'
                        f'<span>{icon}</span><span>{num}. {name}</span></div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-size:0.7rem;color:#8B949E;text-align:center;">'
                    'Streamlit · sklearn · plotly · pandas</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  HERO + STEPPER                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def render_hero():
    pt = st.session_state.problem_type
    st.markdown(f"""
<div class="hero-banner">
  <div class="hero-badge">CS-303B · Machine Learning & ANN · CA-2 Project Exhibition</div>
  <div class="hero-title">🧠 Interactive ML Pipeline Dashboard</div>
  <div class="hero-sub">AutoML · EDA · Preprocessing · Outlier Detection · Feature Engineering ·
  Model Training · Hyperparameter Tuning &nbsp;|&nbsp; Mode: <strong>{pt}</strong></div>
</div>""", unsafe_allow_html=True)

def render_stepper():
    labels = ["Data Input","EDA","Preprocessing","Outliers",
              "Features","Split","Model","Training","Metrics","Tuning"]
    active = st.session_state.active_step
    pills = ""
    for i, name in enumerate(labels, 1):
        cls = "done" if i < active else ("active" if i == active else "")
        icon = "✓" if i < active else str(i)
        pills += f'<div class="step-pill {cls}"><span class="step-num">{icon}</span>{name}</div>'
    st.markdown(f'<div class="stepper">{pills}</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — DATA INPUT                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_data_input():
    st.markdown("### 📂 Step 1 · Data Input")
    dark, tmpl = st.session_state.dark_mode, plotly_theme(st.session_state.dark_mode)
    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded is None:
        info("Upload a CSV file to begin the pipeline.")
        with st.expander("💡 No dataset? Load a sample"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🌸 Iris  (Classification)", use_container_width=True):
                    from sklearn.datasets import load_iris
                    d = load_iris(as_frame=True)
                    st.session_state.df_raw = d.frame
                    st.session_state.target_col = "target"
                    st.session_state.active_step = 2
                    st.rerun()
            with c2:
                if st.button("🏠 California Housing  (Regression)", use_container_width=True):
                    from sklearn.datasets import fetch_california_housing
                    d = fetch_california_housing(as_frame=True)
                    st.session_state.df_raw = d.frame
                    st.session_state.target_col = "MedHouseVal"
                    st.session_state.active_step = 2
                    st.rerun()
        return

    df = pd.read_csv(uploaded)
    st.session_state.df_raw = df.copy()

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl in [(c1,f"{df.shape[0]:,}","Rows"),
                           (c2,str(df.shape[1]),"Columns"),
                           (c3,str(df.isnull().sum().sum()),"Missing"),
                           (c4,str(df.select_dtypes(include="number").shape[1]),"Numeric Cols")]:
        col.markdown(f'<div class="metric-tile"><div class="metric-val">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📋 Preview", "📊 Statistics", "🔵 PCA Visualisation"])

    with t1:
        n = st.slider("Rows to display", 5, min(100, len(df)), 10)
        st.dataframe(df.head(n), use_container_width=True)

    with t2:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with t3:
        num_df = df.select_dtypes(include="number").dropna()
        if num_df.shape[1] < 2:
            warn("Need ≥ 2 numeric columns for PCA.")
        else:
            pca_cols = st.multiselect("Features for PCA", num_df.columns.tolist(),
                                       default=num_df.columns.tolist()[:min(10, len(num_df.columns))])
            pca_dim  = st.radio("Dimensions", [2, 3], horizontal=True)
            if pca_cols and len(pca_cols) >= pca_dim:
                X_s = StandardScaler().fit_transform(num_df[pca_cols])
                pca = PCA(n_components=pca_dim)
                C   = pca.fit_transform(X_s)
                ev  = pca.explained_variance_ratio_
                color_col = st.selectbox("Colour by", ["None"] + df.columns.tolist())
                color_vals = df[color_col].astype(str) if color_col != "None" else None

                if pca_dim == 2:
                    fig = px.scatter(x=C[:,0], y=C[:,1], color=color_vals,
                                     labels={"x":f"PC1 ({ev[0]*100:.1f}%)",
                                             "y":f"PC2 ({ev[1]*100:.1f}%)"},
                                     title="PCA 2D", template=tmpl,
                                     color_discrete_sequence=PALETTE)
                else:
                    fig = px.scatter_3d(x=C[:,0], y=C[:,1], z=C[:,2], color=color_vals,
                                        labels={"x":f"PC1 ({ev[0]*100:.1f}%)",
                                                "y":f"PC2 ({ev[1]*100:.1f}%)",
                                                "z":f"PC3 ({ev[2]*100:.1f}%)"},
                                        title="PCA 3D", template=tmpl,
                                        color_discrete_sequence=PALETTE)
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

                ev_df = pd.DataFrame({"Component":[f"PC{i+1}" for i in range(len(ev))],
                                      "Explained Variance %": ev*100})
                st.plotly_chart(px.bar(ev_df, x="Component", y="Explained Variance %",
                                       title="Scree Plot", template=tmpl,
                                       color_discrete_sequence=[PALETTE[0]]),
                                use_container_width=True)

    st.markdown("---")
    st.markdown("**🎯 Select Target Column**")
    ca, cb = st.columns([3,1])
    with ca:
        target = st.selectbox("Target", df.columns.tolist(),
                              index=len(df.columns)-1, label_visibility="collapsed")
    with cb:
        if st.button("✅ Confirm & Continue →", use_container_width=True):
            st.session_state.target_col = target
            st.session_state.active_step = 2
            success(f"Target: **{target}**")
            st.rerun()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — EDA                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_eda():
    if st.session_state.df_raw is None: warn("Upload data first."); return
    df   = st.session_state.df_raw.copy()
    tmpl = plotly_theme(st.session_state.dark_mode)
    target = st.session_state.target_col
    st.markdown("### 🔍 Step 2 · Exploratory Data Analysis")

    t1,t2,t3,t4 = st.tabs(["📉 Missing Values","📊 Distributions","🔥 Correlation","🎯 Target"])

    with t1:
        miss = (df.isnull().sum().rename("Missing").reset_index()
                .rename(columns={"index":"Column"}))
        miss["Pct"] = (miss["Missing"]/len(df)*100).round(2)
        miss = miss[miss["Missing"]>0].sort_values("Missing", ascending=False)
        if miss.empty:
            success("No missing values found!")
        else:
            st.plotly_chart(px.bar(miss, x="Column", y="Pct", text="Missing",
                                   color="Pct", color_continuous_scale="Reds",
                                   title="Missing Values (%)", template=tmpl),
                            use_container_width=True)
            st.dataframe(miss, use_container_width=True)

    with t2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if num_cols:
            st.markdown("**Numeric Features**")
            sel = st.multiselect("Select columns", num_cols,
                                  default=num_cols[:min(4, len(num_cols))])
            for col in sel:
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=[f"{col} — Histogram", f"{col} — Box"])
                fig.add_trace(go.Histogram(x=df[col], marker_color=PALETTE[0], opacity=0.8), 1, 1)
                fig.add_trace(go.Box(y=df[col], marker_color=PALETTE[1]), 1, 2)
                fig.update_layout(template=tmpl, showlegend=False, height=300, margin=dict(t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
        if cat_cols:
            st.markdown("**Categorical Features**")
            sel_c = st.multiselect("Categorical cols", cat_cols,
                                    default=cat_cols[:min(3, len(cat_cols))])
            for col in sel_c:
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "Count"]
                st.plotly_chart(px.bar(vc, x=col, y="Count", template=tmpl,
                                       color_discrete_sequence=[PALETTE[2]],
                                       title=f"{col} — Value Counts"),
                                use_container_width=True)

    with t3:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2: warn("Need ≥ 2 numeric columns."); return
        corr = num_df.corr()
        fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Pearson Correlation Heatmap",
                         template=tmpl, aspect="auto")
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if st.checkbox("Show scatter matrix (top 5 numeric cols)"):
            st.plotly_chart(px.scatter_matrix(df[num_df.columns[:5].tolist()],
                                               template=tmpl, title="Scatter Matrix"),
                            use_container_width=True)

    with t4:
        if target and target in df.columns:
            if df[target].dtype == "object" or df[target].nunique() <= 20:
                vc = df[target].value_counts().reset_index()
                vc.columns = [target,"Count"]
                ca, cb = st.columns(2)
                with ca:
                    st.plotly_chart(px.bar(vc, x=target, y="Count", template=tmpl,
                                           color_discrete_sequence=[PALETTE[0]],
                                           title="Target Distribution"), use_container_width=True)
                with cb:
                    st.plotly_chart(px.pie(vc, names=target, values="Count",
                                           template=tmpl, title="Target Share",
                                           color_discrete_sequence=PALETTE), use_container_width=True)
            else:
                st.plotly_chart(px.histogram(df, x=target, template=tmpl, marginal="violin",
                                             color_discrete_sequence=[PALETTE[0]],
                                             title="Target Distribution"), use_container_width=True)
        else:
            info("Select a target in Step 1 first.")

    st.markdown("---")
    if st.button("✅ Continue to Preprocessing →"): st.session_state.active_step = 3

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — PREPROCESSING                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_preprocessing():
    if st.session_state.df_raw is None: warn("Upload data first."); return
    df     = st.session_state.df_raw.copy()
    target = st.session_state.target_col
    st.markdown("### ⚙️ Step 3 · Data Preprocessing")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Missing Value Imputation**")
        num_strat = st.selectbox("Numeric strategy", ["mean","median","most_frequent"])
        cat_strat = st.selectbox("Categorical strategy", ["most_frequent","constant"])
        st.markdown("**Feature Scaling**")
        scale_m = st.selectbox("Scaling", ["StandardScaler","MinMaxScaler","None"])
    with cb:
        st.markdown("**Categorical Encoding**")
        enc_m = st.selectbox("Encoding", ["Label Encoding","One-Hot Encoding"])
        st.markdown("**Missing summary**")
        mc = df.isnull().sum(); mc = mc[mc>0]
        if mc.empty: success("No missing values!")
        else: st.dataframe(mc.rename("Missing"), use_container_width=True)

    if st.button("🚀 Apply Preprocessing", use_container_width=True):
        with st.spinner("Processing..."):
            df_p = df.copy()
            y = df_p[target].copy() if target and target in df_p.columns else None
            X = df_p.drop(columns=[target]) if y is not None else df_p.copy()

            num_cols = X.select_dtypes(include="number").columns.tolist()
            cat_cols = X.select_dtypes(exclude="number").columns.tolist()

            if num_cols:
                X[num_cols] = SimpleImputer(strategy=num_strat).fit_transform(X[num_cols])
            le_dict = {}
            if cat_cols:
                X[cat_cols] = SimpleImputer(strategy=cat_strat,
                                             fill_value="missing").fit_transform(X[cat_cols])
                if enc_m == "Label Encoding":
                    for c in cat_cols:
                        le = LabelEncoder()
                        X[c] = le.fit_transform(X[c].astype(str))
                        le_dict[c] = le
                else:
                    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            st.session_state.le_dict = le_dict

            scaler = None
            if scale_m == "StandardScaler": scaler = StandardScaler()
            elif scale_m == "MinMaxScaler": scaler  = MinMaxScaler()
            if scaler:
                X[X.columns] = scaler.fit_transform(X)
            st.session_state.scaler = scaler

            df_clean = X.copy()
            if y is not None: df_clean[target] = y.values
            st.session_state.df_clean = df_clean

        success(f"Done! Shape: **{df_clean.shape}**")
        st.dataframe(df_clean.head(), use_container_width=True)
        st.session_state.active_step = 4

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — OUTLIER DETECTION                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_outlier_detection():
    df = st.session_state.df_clean
    if df is None: warn("Complete preprocessing first."); return
    target = st.session_state.target_col
    tmpl   = plotly_theme(st.session_state.dark_mode)
    st.markdown("### 🔎 Step 4 · Outlier Detection")

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    feat_cols = [c for c in num_cols if c != target]
    method    = st.selectbox("Detection method", ["IQR","Isolation Forest","DBSCAN"])
    mask      = pd.Series([False]*len(df), index=df.index)

    if method == "IQR":
        mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        for c in feat_cols:
            Q1,Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3-Q1
            mask |= (df[c] < Q1-mult*IQR) | (df[c] > Q3+mult*IQR)
    elif method == "Isolation Forest":
        cont = st.slider("Contamination", 0.01, 0.3, 0.05, 0.01)
        pred = IsolationForest(contamination=cont, random_state=42).fit_predict(df[feat_cols].fillna(0))
        mask = pd.Series(pred==-1, index=df.index)
    elif method == "DBSCAN":
        eps   = st.slider("eps", 0.1, 5.0, 0.5, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)
        X_s   = StandardScaler().fit_transform(df[feat_cols].fillna(0))
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X_s)
        mask   = pd.Series(labels==-1, index=df.index)

    n_out = mask.sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Outliers", f"{n_out:,}")
    c3.metric("Outlier %", f"{n_out/len(df)*100:.1f}%")

    if len(feat_cols) >= 2:
        fig = px.scatter(df, x=feat_cols[0], y=feat_cols[1],
                         color=mask.map({True:"Outlier",False:"Normal"}),
                         color_discrete_map={"Outlier":"#FF7B72","Normal":"#3FB950"},
                         title=f"Outliers — {method}", template=tmpl)
        st.plotly_chart(fig, use_container_width=True)

    if n_out > 0:
        if st.checkbox("👁 Preview outlier rows"):
            st.dataframe(df[mask].head(20), use_container_width=True)
        if st.button("🗑️ Remove outliers"):
            st.session_state.df_clean = df[~mask].reset_index(drop=True)
            success(f"Removed {n_out} rows. New shape: **{st.session_state.df_clean.shape}**")
    else:
        success("No outliers with current settings.")

    st.markdown("---")
    if st.button("✅ Continue to Feature Selection →"): st.session_state.active_step = 5

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — FEATURE SELECTION                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_feature_selection():
    df = st.session_state.df_clean
    if df is None: warn("Complete preprocessing first."); return
    target = st.session_state.target_col
    pt     = st.session_state.problem_type
    tmpl   = plotly_theme(st.session_state.dark_mode)
    st.markdown("### 🧬 Step 5 · Feature Selection")

    if target not in df.columns: warn(f"Target '{target}' not in cleaned dataframe."); return
    X = df.drop(columns=[target]); y = df[target]
    num_X = X.select_dtypes(include="number")
    method = st.selectbox("Method", ["Variance Threshold","Correlation","Information Gain"])
    selected = X.columns.tolist()

    if method == "Variance Threshold":
        thresh = st.slider("Variance threshold", 0.0, 1.0, 0.01, 0.01)
        sel = VarianceThreshold(threshold=thresh); sel.fit(num_X.fillna(0))
        kept = num_X.columns[sel.get_support()].tolist(); selected = kept
        var_df = pd.DataFrame({"Feature":num_X.columns,
                               "Variance":num_X.var().values,
                               "Selected":num_X.columns.isin(kept)}).sort_values("Variance",ascending=False)
        st.plotly_chart(px.bar(var_df, x="Feature", y="Variance",
                               color="Selected", color_discrete_map={True:"#3FB950",False:"#FF7B72"},
                               title="Feature Variances", template=tmpl),
                        use_container_width=True)

    elif method == "Correlation":
        ct = st.slider("Min |correlation| with target", 0.0, 1.0, 0.05, 0.01)
        y_num = y if y.dtype != "object" else y.astype("category").cat.codes
        cs = num_X.corrwith(y_num).abs()
        kept = cs[cs >= ct].index.tolist(); selected = kept
        cdf = cs.reset_index(); cdf.columns = ["Feature","Correlation"]
        cdf = cdf.sort_values("Correlation", ascending=False)
        cdf["Selected"] = cdf["Feature"].isin(kept)
        st.plotly_chart(px.bar(cdf, x="Feature", y="Correlation",
                               color="Selected", color_discrete_map={True:"#3FB950",False:"#FF7B72"},
                               title="Feature-Target Correlation", template=tmpl),
                        use_container_width=True)

    elif method == "Information Gain":
        k = st.slider("Top-K features", 1, max(1,len(num_X.columns)), min(10,len(num_X.columns)))
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        scores = (mutual_info_classif(num_X.fillna(0), y_enc, random_state=42)
                  if pt == "Classification"
                  else mutual_info_regression(num_X.fillna(0), y_enc, random_state=42))
        igdf = pd.DataFrame({"Feature":num_X.columns,"IG":scores}).sort_values("IG",ascending=False)
        kept = igdf.head(k)["Feature"].tolist(); selected = kept
        igdf["Selected"] = igdf["Feature"].isin(kept)
        st.plotly_chart(px.bar(igdf, x="Feature", y="IG",
                               color="Selected", color_discrete_map={True:"#3FB950",False:"#FF7B72"},
                               title="Information Gain", template=tmpl),
                        use_container_width=True)

    st.markdown("**✏️ Manually adjust selection**")
    selected = st.multiselect("Final features", X.columns.tolist(),
                               default=[f for f in selected if f in X.columns][:min(len(selected),len(X.columns))])
    c1,c2 = st.columns(2)
    c1.metric("Total Features", len(X.columns))
    c2.metric("Selected", len(selected))

    if st.button("✅ Confirm & Continue →", use_container_width=True):
        if not selected: warn("Select at least 1 feature."); return
        st.session_state.selected_features = selected
        st.session_state.active_step = 6
        success(f"Selected {len(selected)} features.")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 6 — DATA SPLIT                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_data_split():
    df    = st.session_state.df_clean
    feats = st.session_state.selected_features
    target = st.session_state.target_col
    if df is None or not feats: warn("Complete feature selection first."); return
    pt   = st.session_state.problem_type
    tmpl = plotly_theme(st.session_state.dark_mode)
    st.markdown("### ✂️ Step 6 · Train-Test Split")

    ca, cb = st.columns(2)
    with ca: test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    with cb: seed = st.number_input("Random seed", 0, 9999, 42)

    X = df[feats]; y = df[target]
    if pt == "Classification" and y.dtype == "object":
        le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
        st.session_state.le_dict["__target__"] = le

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed))

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Train Samples", f"{len(X_train):,}")
    c2.metric("Test Samples",  f"{len(X_test):,}")
    c3.metric("Train %",       f"{(1-test_size)*100:.0f}%")
    c4.metric("Test %",        f"{test_size*100:.0f}%")

    fig = go.Figure(go.Pie(labels=["Train","Test"], values=[len(X_train),len(X_test)],
                            marker_colors=["#3FB950","#FF7B72"], hole=0.55,
                            textinfo="label+percent"))
    fig.update_layout(template=tmpl, height=260, margin=dict(t=10,b=10,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("✅ Confirm Split & Continue →", use_container_width=True):
        st.session_state.X_train = X_train; st.session_state.X_test  = X_test
        st.session_state.y_train = y_train; st.session_state.y_test  = y_test
        st.session_state.active_step = 7
        success(f"Split done — {len(X_train):,} train / {len(X_test):,} test.")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 7 — MODEL SELECTION                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_model_selection():
    pt = st.session_state.problem_type
    st.markdown("### 🤖 Step 7 · Model Selection")
    opts = (["Logistic Regression","SVM","Random Forest","KMeans (Clustering)"]
            if pt == "Classification"
            else ["Linear Regression","SVM","Random Forest"])
    name   = st.selectbox("Model", opts)
    params = {}

    if name == "SVM":
        kernel = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"])
        C = st.number_input("C", 0.01, 100.0, 1.0, 0.1)
        params = {"kernel": kernel, "C": C}
    elif name == "Random Forest":
        n_est = st.slider("Trees", 10, 500, 100, 10)
        max_d = st.selectbox("Max depth", [None, 5, 10, 20, 50])
        params = {"n_estimators": n_est, "max_depth": max_d}
    elif name == "KMeans (Clustering)":
        k = st.slider("Clusters K", 2, 15, 3)
        params = {"n_clusters": k}
    elif name == "Logistic Regression":
        C = st.number_input("C", 0.01, 100.0, 1.0, 0.1)
        params = {"C": C}

    info(f"Selected: **{name}**")
    if st.button("✅ Confirm Model →", use_container_width=True):
        st.session_state.model_name   = name
        st.session_state.model_params = params
        st.session_state.active_step  = 8
        success(f"Model **{name}** confirmed.")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 8 — TRAINING & K-FOLD CV                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_training():
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    if X_train is None: warn("Complete data split first."); return
    name   = st.session_state.model_name
    params = st.session_state.model_params or {}
    pt     = st.session_state.problem_type
    tmpl   = plotly_theme(st.session_state.dark_mode)
    if not name: warn("Select a model first."); return

    st.markdown("### 🏋️ Step 8 · Training & K-Fold Cross Validation")
    k = st.slider("K (folds)", 2, 15, 5)

    if st.button("🚀 Train Model", use_container_width=True):
        model = build_model(name, params, pt)
        if model is None: warn("Could not build model."); return

        with st.spinner(f"Training {name}..."):
            is_cluster = name == "KMeans (Clustering)"
            if is_cluster:
                model.fit(X_train)
                st.session_state.trained_model = model
                st.session_state.active_step = 9
                success("KMeans trained! (Unsupervised — no CV)")
                return

            metric = "accuracy" if pt=="Classification" else "r2"
            scores = cross_val_score(model, X_train, y_train,
                                     cv=KFold(n_splits=k, shuffle=True, random_state=42),
                                     scoring=metric)
            st.session_state.cv_scores = scores
            model.fit(X_train, y_train)
            st.session_state.trained_model = model

        c1,c2,c3 = st.columns(3)
        c1.metric(f"Mean CV {metric.upper()}", f"{scores.mean():.4f}")
        c2.metric("Std Dev", f"±{scores.std():.4f}")
        c3.metric("Best Fold", f"{scores.max():.4f}")

        fold_df = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(k)], "Score":scores})
        fig = px.bar(fold_df, x="Fold", y="Score",
                     title=f"K-Fold CV ({metric})", template=tmpl,
                     color="Score", color_continuous_scale="Teal")
        fig.add_hline(y=scores.mean(), line_dash="dash", line_color="#FF7B72",
                      annotation_text=f"Mean={scores.mean():.4f}")
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        success(f"Training complete! Mean {metric}: **{scores.mean():.4f}**")
        st.session_state.active_step = 9

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 9 — METRICS                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_metrics():
    model   = st.session_state.trained_model
    X_test  = st.session_state.X_test
    y_test  = st.session_state.y_test
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    if model is None: warn("Train a model first."); return

    pt   = st.session_state.problem_type
    tmpl = plotly_theme(st.session_state.dark_mode)
    name = st.session_state.model_name or ""
    st.markdown("### 📊 Step 9 · Evaluation Metrics")

    if name == "KMeans (Clustering)":
        labels = model.predict(X_test)
        feats  = st.session_state.selected_features
        if len(feats) >= 2:
            fig = px.scatter(x=X_test.iloc[:,0], y=X_test.iloc[:,1],
                             color=labels.astype(str), template=tmpl,
                             title="KMeans Cluster Assignments",
                             labels={"x":feats[0],"y":feats[1]},
                             color_discrete_sequence=PALETTE)
            st.plotly_chart(fig, use_container_width=True)
        info("Unsupervised — cluster assignments shown above.")
        return

    y_pred  = model.predict(X_test)
    y_tr_pr = model.predict(X_train)

    if pt == "Classification":
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc_tr = accuracy_score(y_train, y_tr_pr)
        st.session_state.metrics_before = {"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1}

        metric_tiles({"Accuracy":f"{acc:.4f}","Precision":f"{prec:.4f}",
                      "Recall":f"{rec:.4f}","F1-Score":f"{f1:.4f}"})
        st.markdown("<br>", unsafe_allow_html=True)

        gap = acc_tr - acc
        if gap > 0.15:    warn(f"Possible **overfitting** — Train: {acc_tr:.4f} | Test: {acc:.4f} | Gap: {gap:.4f}")
        elif acc < 0.6:   warn(f"Possible **underfitting** — Test accuracy: {acc:.4f}")
        else:             success(f"Good generalisation — Train: {acc_tr:.4f} | Test: {acc:.4f}")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acc*100,
            number={"suffix":"%","font":{"size":36}},
            title={"text":"Test Accuracy"},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":"#3FB950"},
                   "steps":[{"range":[0,50],"color":"#FF7B72"},
                             {"range":[50,75],"color":"#FFA657"},
                             {"range":[75,100],"color":"#3FB950"}],
                   "threshold":{"line":{"color":"#58A6FF","width":4},
                                "thickness":0.75,"value":acc_tr*100}}))
        gauge.update_layout(template=tmpl, height=300, margin=dict(t=30,b=10,l=30,r=30))
        st.plotly_chart(gauge, use_container_width=True)

        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, template=tmpl,
                                   color_continuous_scale="Blues",
                                   title="Confusion Matrix",
                                   labels={"x":"Predicted","y":"Actual"}),
                        use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train", x=["Accuracy"], y=[acc_tr], marker_color="#58A6FF"))
        fig.add_trace(go.Bar(name="Test",  x=["Accuracy"], y=[acc],    marker_color="#3FB950"))
        fig.update_layout(barmode="group", template=tmpl, height=300, title="Train vs Test Accuracy")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        r2_tr = r2_score(y_train, y_tr_pr)
        st.session_state.metrics_before = {"MAE":mae,"MSE":mse,"RMSE":rmse,"R2":r2}

        metric_tiles({"MAE":f"{mae:.4f}","MSE":f"{mse:.4f}","RMSE":f"{rmse:.4f}","R²":f"{r2:.4f}"})
        st.markdown("<br>", unsafe_allow_html=True)

        gap = r2_tr - r2
        if gap > 0.2:   warn(f"Possible **overfitting** — Train R²: {r2_tr:.4f} | Test R²: {r2:.4f}")
        elif r2 < 0.4:  warn(f"Possible **underfitting** — R²: {r2:.4f}")
        else:           success(f"Good fit — Train R²: {r2_tr:.4f} | Test R²: {r2:.4f}")

        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        fig = px.scatter(x=y_test, y=y_pred, template=tmpl,
                         labels={"x":"Actual","y":"Predicted"},
                         title="Actual vs Predicted", opacity=0.65,
                         color_discrete_sequence=[PALETTE[0]])
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color="#FF7B72", dash="dash", width=2))
        st.plotly_chart(fig, use_container_width=True)

        residuals = np.array(y_test) - y_pred
        st.plotly_chart(px.histogram(residuals, template=tmpl, nbins=40,
                                     title="Residuals Distribution",
                                     color_discrete_sequence=[PALETTE[1]]),
                        use_container_width=True)

    st.markdown("---")
    st.markdown("**⬇ Download Trained Model**")
    st.markdown(get_download_link(model, f"model_{name.replace(' ','_')}.pkl",
                                  "Download Model (.pkl)"), unsafe_allow_html=True)
    st.markdown("---")
    if st.button("✅ Continue to Hyperparameter Tuning →"): st.session_state.active_step = 10

# ╔══════════════════════════════════════════════════════════════════╗
# ║  STEP 10 — HYPERPARAMETER TUNING                                ║
# ╚══════════════════════════════════════════════════════════════════╝
def step_hp_tuning():
    name    = st.session_state.model_name
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test  = st.session_state.X_test
    y_test  = st.session_state.y_test
    pt      = st.session_state.problem_type
    tmpl    = plotly_theme(st.session_state.dark_mode)

    if X_train is None or not name: warn("Complete training first."); return
    if name == "KMeans (Clustering)": info("HP tuning not applicable for KMeans."); return

    st.markdown("### ⚡ Step 10 · Hyperparameter Tuning")
    tune = st.radio("Search strategy", ["GridSearchCV","RandomizedSearchCV"], horizontal=True)

    grids = {
        "Logistic Regression": {"C":[0.01,0.1,1.0,10.0,100.0],"solver":["lbfgs","liblinear"]},
        "Linear Regression":   {},
        "SVM":                 {"C":[0.1,1.0,10.0],"kernel":["rbf","linear","poly"]},
        "Random Forest":       {"n_estimators":[50,100,200],"max_depth":[None,5,10,20],
                                "min_samples_split":[2,5,10]},
    }
    grid = grids.get(name, {})
    if not grid: info(f"No hyperparameters defined for {name}."); return

    st.markdown(f"**Search space for `{name}`:**")
    for k,v in grid.items(): st.markdown(f"- `{k}`: {v}")

    cv_k   = st.slider("CV folds", 2, 10, 3)
    metric = "accuracy" if pt=="Classification" else "r2"

    if st.button("🔬 Run Search", use_container_width=True):
        base = build_model(name, {}, pt)
        if base is None: warn("Cannot build model."); return

        with st.spinner("Searching..."):
            if tune == "GridSearchCV":
                searcher = GridSearchCV(base, grid, cv=cv_k, scoring=metric, n_jobs=-1)
            else:
                searcher = RandomizedSearchCV(base, grid, n_iter=10, cv=cv_k,
                                              scoring=metric, n_jobs=-1, random_state=42)
            searcher.fit(X_train, y_train)

        best   = searcher.best_estimator_
        bparams = searcher.best_params_
        st.session_state.tuned_model = best

        st.markdown("**🏆 Best Parameters:**")
        for k,v in bparams.items(): st.markdown(f"- `{k}`: `{v}`")

        before = st.session_state.metrics_before or {}
        if pt == "Classification":
            after_score  = accuracy_score(y_test, best.predict(X_test))
            before_score = before.get("Accuracy", 0.0)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Before", x=["Accuracy"], y=[before_score], marker_color="#FF7B72"))
            fig.add_trace(go.Bar(name="After",  x=["Accuracy"], y=[after_score],  marker_color="#3FB950"))
            fig.update_layout(barmode="group", template=tmpl, height=350,
                               title="Performance: Before vs After Tuning")
            st.plotly_chart(fig, use_container_width=True)
            delta = after_score - before_score
            (success if delta >= 0 else warn)(f"Accuracy {'improved' if delta>=0 else 'changed'} by **{delta*100:.2f}%** → {after_score:.4f}")
        else:
            y_pred_t = best.predict(X_test)
            after_r2 = r2_score(y_test, y_pred_t)
            after_mae = mean_absolute_error(y_test, y_pred_t)
            br2  = before.get("R2",  0.0)
            bmae = before.get("MAE", 0.0)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Before", x=["R²","MAE"], y=[br2,bmae],       marker_color="#FF7B72"))
            fig.add_trace(go.Bar(name="After",  x=["R²","MAE"], y=[after_r2,after_mae], marker_color="#3FB950"))
            fig.update_layout(barmode="group", template=tmpl, height=350,
                               title="Performance: Before vs After Tuning")
            st.plotly_chart(fig, use_container_width=True)
            success(f"R² after tuning: **{after_r2:.4f}** | MAE: **{after_mae:.4f}**")

        st.markdown("---")
        st.markdown("**⬇ Download Tuned Model**")
        st.markdown(get_download_link(best, f"tuned_{name.replace(' ','_')}.pkl",
                                      "Download Tuned Model (.pkl)"), unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  EXTRA: PREDICT ON NEW DATA                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
def section_predict_new():
    model = st.session_state.tuned_model or st.session_state.trained_model
    feats = st.session_state.selected_features
    if model is None: return

    st.markdown("---")
    st.markdown("### 🔮 Predict on New Data")
    up = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="pred_upload")
    if up:
        df_new = pd.read_csv(up)
        missing = [c for c in feats if c not in df_new.columns]
        if missing: warn(f"Missing columns: {missing}"); return
        X_new = df_new[feats].copy()
        sc = st.session_state.scaler
        if sc: X_new = pd.DataFrame(sc.transform(X_new), columns=feats)
        df_new["Prediction"] = model.predict(X_new)
        st.dataframe(df_new.head(20), use_container_width=True)
        csv = df_new.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions CSV", csv, "predictions.csv", "text/csv")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
def main():
    inject_css(st.session_state.dark_mode)
    render_sidebar()
    render_hero()
    render_stepper()

    tabs = st.tabs(["📂 Data Input","🔍 EDA","⚙️ Preprocessing","🔎 Outliers",
                    "🧬 Features","✂️ Split","🤖 Model","🏋️ Training","📊 Metrics","⚡ Tuning"])

    with tabs[0]:  step_data_input()
    with tabs[1]:  step_eda()
    with tabs[2]:  step_preprocessing()
    with tabs[3]:  step_outlier_detection()
    with tabs[4]:  step_feature_selection()
    with tabs[5]:  step_data_split()
    with tabs[6]:  step_model_selection()
    with tabs[7]:  step_training()
    with tabs[8]:
        step_metrics()
        section_predict_new()
    with tabs[9]:  step_hp_tuning()

    st.markdown("---")
    st.markdown("""
<div style="text-align:center;font-size:0.73rem;color:#8B949E;padding:0.75rem 0;">
  🧠 <strong>ML Pipeline Dashboard</strong> &nbsp;·&nbsp;
  CS-303B Machine Learning & ANN &nbsp;·&nbsp; CA-2 Project Exhibition &nbsp;·&nbsp;
  Streamlit · scikit-learn · Plotly · pandas · numpy
</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
