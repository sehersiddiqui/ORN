from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

st.set_page_config(page_title="ORN Research Dashboard", page_icon="🦷", layout="wide", initial_sidebar_state="expanded")

# Custom visual styling
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#102b4e 0%,#1e6480 65%,#4a9b9a 100%); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.05rem; line-height: 1.18; margin: 0 0 .4rem 0; color: white;}
.hero p {font-size: 1rem; margin: .25rem 0; color: #eef7fb;}
.note {background:#f0f6fa; border-left: 5px solid #2c7a91; padding: .8rem 1rem; border-radius: 8px; margin:.5rem 0 1rem 0; color:#17324d;}
.warning {background:#fff7e8; border-left: 5px solid #d99a28; padding: .8rem 1rem; border-radius: 8px; margin:.5rem 0 1rem 0; color:#5a4218;}
.small-muted {color:#64748b;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "TTE_ORN_data_Figshare.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    if "ORN status" not in data:
        raise ValueError("Expected outcome column 'ORN status' was not found.")
    data["ORN status"] = pd.to_numeric(data["ORN status"], errors="raise").astype(int)
    return data

df = load_data()
n = len(df)
events = int(df["ORN status"].sum())
prev = events / n
non_events = n - events

st.markdown("""
<div class="hero">
  <h1>ORN Research Progress Dashboard</h1>
  <p><b>Longitudinal Multimodal Progression Modelling of Osteoradionecrosis of the Jaw</b></p>
  <p>Research progress review · Seher Siddiqui & Arnav Jain · NMIMS MPSTME</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Explore the research")
    section = st.radio("Choose a section", [
        "Executive overview", "Who is in the cohort?", "Dose–volume analysis",
        "Data quality & relationships", "Baseline model laboratory",
        "Project stage & next steps", "Data dictionary"
    ])
    st.divider()
    st.caption("Dataset: TTE_ORN_data_Figshare.csv")
    st.caption(f"{n:,} records · {df.shape[1]} columns")
    st.caption("All charts are interactive: hover, zoom, select, and toggle legends.")
    st.divider()
    st.caption("Research prototype only. Not for diagnosis or treatment decisions.")

def explainer(title, what, how_to_read, why_matters, caution=None):
    st.subheader(title)
    st.markdown(f"**What this shows:** {what}")
    st.markdown(f"**How to read it:** {how_to_read}")
    st.markdown(f"**Why it matters:** {why_matters}")
    if caution:
        st.markdown(f"<div class='warning'><b>Important caution:</b> {caution}</div>", unsafe_allow_html=True)

def kpi(label, value, help_text=None):
    st.metric(label, value, help=help_text)

if section == "Executive overview":
    st.header("The project, in one minute")
    st.markdown("""
    This dashboard separates **what the current data already support** from **what the proposed multimodal system still needs**.
    The available cohort supports retrospective clinical/dosimetric analysis and baseline modelling. It does not contain the linked serial CT radiomics and biomarker values needed to claim a trained T0–T4 LSTM, CT-crop ResNet-18, or full Temporal Fusion Transformer.
    """)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Patients / records", f"{n:,}", "Rows in the supplied dataset.")
    with c2: kpi("Recorded ORN events", f"{events:,}", "Rows with ORN status = 1.")
    with c3: kpi("Recorded ORN prevalence", f"{prev:.1%}", "Observed outcome proportion in this dataset.")
    with c4: kpi("Available columns", f"{df.shape[1]}", "Clinical, outcome, and dose-volume variables.")
    st.markdown("<div class='note'><b>Headline:</b> The real-data foundation and initial tabular modelling workflow are in place. The next major dependency is approved, patient-linked longitudinal imaging and outcomes.</div>", unsafe_allow_html=True)
    col1,col2=st.columns([1,1])
    with col1:
        fig=go.Figure(data=[go.Pie(labels=["Observed ORN","No recorded ORN"],values=[events,non_events],hole=.58,
            marker=dict(line=dict(color="white",width=2)),textinfo="label+percent",
            hovertemplate="%{label}<br>Records: %{value:,}<br>Share: %{percent}<extra></extra>")])
        fig.update_layout(title="Cohort composition",height=390,margin=dict(l=10,r=10,t=60,b=10),legend=dict(orientation="h",y=-.08))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Interpretation: ORN is the minority class, so accuracy alone can be misleading. Recall, precision–recall curves, and calibration should also be considered.")
    with col2:
        if "ORN Grade (Tsai)" in df:
            counts=df["ORN Grade (Tsai)"].value_counts(dropna=False).sort_index().reset_index()
            counts.columns=["Recorded value","Records"]
            fig=px.bar(counts,x="Recorded value",y="Records",text="Records",title="Values recorded in the Tsai grade field",
                       color="Records",color_continuous_scale="Blues")
            fig.update_layout(height=390,margin=dict(l=10,r=10,t=60,b=10),coloraxis_showscale=False)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Interpretation: the largest bar corresponds to non-ORN records in this file. Verify the coding/data dictionary before interpreting this field as a clinical grade distribution.")
    st.subheader("What is done versus what remains")
    status=pd.DataFrame([
        ["Real cohort audit and descriptive analysis","Available"],
        ["Clinical/DVH baseline classifiers","Prototype available"],
        ["ROC/PR, calibration and confusion-matrix evaluation","Available in baseline workflow"],
        ["Patient-linked serial T0–T4 radiomics","Not supplied"],
        ["ResNet-18 on real ORN-labelled CT crops","Not trained"],
        ["Full multimodal TFT + biomarkers","Not trained / data dependent"],
        ["Clinical treatment recommendation","Not validated; not for use"]
    ],columns=["Workstream","Current status"])
    st.dataframe(status,use_container_width=True,hide_index=True)
    st.markdown("<div class='warning'><b>Panel wording:</b> describe this as an exploratory real-data research prototype. Do not present unavailable imaging models as completed experiments or the dashboard as a clinical decision tool.</div>",unsafe_allow_html=True)

elif section == "Who is in the cohort?":
    st.header("Who is in the cohort?")
    explainer("Patient profile", "Age distribution, available categorical patient/tumour descriptors, and recorded ORN status.",
              "Hover on bars to see counts. Select a category in the legend to hide/show a group.",
              "This shows what kinds of patients are represented and helps spot imbalance before modelling.",
              "These are retrospective cohort descriptions, not evidence that a factor causes ORN.")
    if "Age" in df:
        temp=df.copy()
        temp["Outcome"]=temp["ORN status"].map({0:"No recorded ORN",1:"Observed ORN"})
        fig=px.histogram(temp,x="Age",color="Outcome",barmode="overlay",opacity=.68,nbins=25,
                         title="Age distribution by observed ORN status")
        fig.update_layout(height=430,xaxis_title="Age",yaxis_title="Number of records")
        st.plotly_chart(fig,use_container_width=True)
        st.info("Interpretation: compare where the two distributions overlap and where they differ. Overlap is expected; age alone should not be treated as a reliable predictor.")
    candidates=["Gender","Smoking status","Chemotherapy","Pre-RT dental extractions","Postop RT vs. Definitive RT","HPV/P16 +Ve","Tumor site group","T Stage","N Stage"]
    available=[c for c in candidates if c in df.columns]
    chosen=st.selectbox("Choose a characteristic to inspect",available)
    counts=df.groupby([chosen,"ORN status"],dropna=False).size().reset_index(name="Records")
    counts["Outcome"]=counts["ORN status"].map({0:"No recorded ORN",1:"Observed ORN"})
    fig=px.bar(counts,x=chosen,y="Records",color="Outcome",barmode="group",title=f"{chosen}: record counts by outcome",
               hover_data=["ORN status"])
    fig.update_layout(height=460,xaxis_title=chosen,yaxis_title="Records")
    st.plotly_chart(fig,use_container_width=True)
    rate=df.groupby(chosen,dropna=False)["ORN status"].agg(["count","mean"]).reset_index()
    rate.columns=[chosen,"Records","Observed ORN fraction"]
    rate["Observed ORN fraction"]=rate["Observed ORN fraction"].map(lambda x:f"{x:.1%}")
    st.markdown("**Group summary**")
    st.dataframe(rate,use_container_width=True,hide_index=True)
    st.caption("Interpretation: groups with small counts can have unstable percentages. Always read the denominator alongside the rate.")

elif section == "Dose–volume analysis":
    st.header("Radiation dose–volume analysis")
    st.markdown("DVH variables describe how much of the mandible receives a specified radiation dose, or the dose received by a specified fraction of the mandible.")
    dose_cols=[c for c in df.columns if (c.startswith("D") and "%" in c) or (c.startswith("V") and c.endswith("Gy"))]
    if not dose_cols:
        st.warning("No dose-volume columns were found.")
    else:
        metric=st.selectbox("Choose a dose–volume metric",dose_cols,index=dose_cols.index("D30%") if "D30%" in dose_cols else 0)
        outcome_names={0:"No recorded ORN",1:"Observed ORN"}
        temp=df[["ORN status",metric]].copy()
        temp[metric]=pd.to_numeric(temp[metric],errors="coerce")
        temp["Outcome"]=temp["ORN status"].map(outcome_names)
        fig=px.box(temp,x="Outcome",y=metric,color="Outcome",points="outliers",
                   title=f"{metric}: distribution by observed ORN status",
                   color_discrete_map={"No recorded ORN":"#377eb8","Observed ORN":"#e07a3f"})
        fig.update_layout(height=450,xaxis_title="",yaxis_title=f"{metric} (check units in source)")
        st.plotly_chart(fig,use_container_width=True)
        stats=temp.groupby("Outcome")[metric].agg(["count","median","mean","std"]).round(2)
        st.dataframe(stats,use_container_width=True)
        a=stats.loc["No recorded ORN","median"] if "No recorded ORN" in stats.index else np.nan
        b=stats.loc["Observed ORN","median"] if "Observed ORN" in stats.index else np.nan
        if pd.notna(a) and pd.notna(b):
            direction="higher" if b>a else "lower" if b<a else "similar"
            st.markdown(f"**Plain-language interpretation:** the observed-ORN group has a {direction} median {metric} ({b:.2f}) than the no-recorded-ORN group ({a:.2f}). This is a descriptive difference, not proof of causality.")
        st.markdown("### Compare several metrics")
        selected=st.multiselect("Select metrics",dose_cols,default=[c for c in ["D30%","D50%","V50Gy","V60Gy"] if c in dose_cols])
        if selected:
            long=df[["ORN status"]+selected].melt(id_vars="ORN status",var_name="Metric",value_name="Value")
            long["Outcome"]=long["ORN status"].map(outcome_names)
            fig=px.box(long,x="Metric",y="Value",color="Outcome",facet_col="Metric",facet_col_wrap=2,points=False,
                       title="Selected dose-volume distributions")
            fig.update_layout(height=650)
            st.plotly_chart(fig,use_container_width=True)
        st.markdown("<div class='note'><b>Why it matters:</b> DVH features are clinically interpretable baseline predictors and a sensible starting point for comparison with published NTCP approaches such as PREDMORN.</div>",unsafe_allow_html=True)
        st.markdown("<div class='warning'><b>Benchmark caution:</b> these plots do not reproduce PREDMORN and are not a head-to-head benchmark. The cohort, endpoint, and validation scheme must be harmonized for a fair comparison.</div>",unsafe_allow_html=True)

elif section == "Data quality & relationships":
    st.header("Data quality and variable relationships")
    explainer("Missingness", "The percentage of missing values in each column.",
              "Longer bars mean more values are missing. Use the table to see exact percentages.",
              "Missingness can bias a model if handled poorly; preprocessing must be learned on training data only.",
              "The dataset copy currently supplied may have complete columns, but this should be rechecked after any data update.")
    miss=(df.isna().mean()*100).sort_values(ascending=False).head(25).reset_index()
    miss.columns=["Variable","Missing (%)"]
    fig=px.bar(miss,x="Missing (%)",y="Variable",orientation="h",title="Variables with the most missing values")
    fig.update_layout(height=550,yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(miss.round(2),use_container_width=True,hide_index=True)
    numeric=df.select_dtypes(include=np.number).drop(columns=["Study ID"],errors="ignore")
    selected=[c for c in ["Age","Smoking Pack-Years","Mandible volume (cc)","D30%","D50%","D70%","V30Gy","V50Gy","V60Gy","ORN status"] if c in numeric]
    if len(selected)>1:
        fig=px.imshow(df[selected].corr(),text_auto=".2f",aspect="auto",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",
                      title="Correlation heatmap: selected numeric variables")
        fig.update_layout(height=620)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Correlation ranges from -1 to +1. Positive means variables tend to increase together; negative means one tends to decrease as the other increases. Correlation does not establish causation.")
    st.subheader("Dataset audit")
    audit=pd.DataFrame([
        ["Rows",len(df)],["Columns",df.shape[1]],["Duplicate Study IDs",int(df["Study ID"].duplicated().sum()) if "Study ID" in df else "ID unavailable"],
        ["Rows with missing cells",int(df.isna().any(axis=1).sum())],["Total missing cells",int(df.isna().sum().sum())]
    ],columns=["Audit check","Result"])
    st.dataframe(audit,use_container_width=True,hide_index=True)

elif section == "Baseline model laboratory":
    st.header("Baseline model laboratory")
    st.markdown("Train a small set of tabular models on a stratified holdout split. This is a reproducible exploratory demonstration; results may vary with the selected split and predictors.")
    with st.expander("How to interpret the metrics",expanded=True):
        st.markdown("""
        - **ROC-AUC:** ability to rank ORN cases above non-cases (0.5 is chance; 1.0 is perfect ranking).
        - **Average precision:** summarizes precision–recall performance and is useful when ORN cases are the minority class.
        - **Recall / sensitivity:** fraction of recorded ORN cases detected at the selected threshold.
        - **Precision:** fraction of predicted ORN cases that are recorded ORN.
        - **Brier score:** average squared probability error; lower is better.
        - **Threshold:** default 0.50 for the confusion matrix and F1. This is not automatically the clinically appropriate threshold.
        """)
    target="ORN status"
    test_size=st.slider("Holdout test proportion",0.15,0.35,0.20,0.05)
    threshold=st.slider("Classification threshold",0.10,0.90,0.50,0.05)
    seed=st.number_input("Random seed",min_value=0,max_value=9999,value=42,step=1)
    excluded=[c for c in ["Study ID",target,"ORN Grade (Tsai)","Time to event","Overall survival","Overall survival time"] if c in df.columns]
    X=df.drop(columns=excluded).copy()
    y=df[target].astype(int)
    # Dose features are numeric in this dataset; coerce all object predictors to categorical.
    num_cols=X.select_dtypes(include=np.number).columns.tolist()
    cat_cols=[c for c in X.columns if c not in num_cols]
    pre=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num_cols),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat_cols)
    ])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=float(test_size),random_state=int(seed),stratify=y)
    model_names=st.multiselect("Models to run",["Logistic regression","Random Forest","Extra Trees","XGBoost (if installed)"],default=["Logistic regression","Random Forest"])
    models={
        "Logistic regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
        "Random Forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=int(seed),n_jobs=-1),
        "Extra Trees":ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=int(seed),n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        scale=max((ytr==0).sum()/max((ytr==1).sum(),1),1)
        models["XGBoost (if installed)"]=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,
                reg_lambda=2,eval_metric="logloss",random_state=int(seed),scale_pos_weight=scale,n_jobs=4)
    except Exception:
        pass
    run=st.button("Train / refresh baseline models",type="primary")
    if run:
        fitted={}; probs={}; rows=[]
        with st.spinner("Training models on the selected split..."):
            for name in model_names:
                if name not in models:
                    continue
                pipe=Pipeline([("preprocess",pre),("model",models[name])])
                pipe.fit(Xtr,ytr)
                p=pipe.predict_proba(Xte)[:,1]
                pred=(p>=threshold).astype(int)
                fitted[name]=pipe; probs[name]=p
                rows.append({"Model":name,"Test patients":len(yte),"ROC-AUC":roc_auc_score(yte,p),
                    "Average precision":average_precision_score(yte,p),"F1":f1_score(yte,pred,zero_division=0),
                    "Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),
                    "Brier":brier_score_loss(yte,p)})
        if not rows:
            st.warning("Choose at least one available model.")
        else:
            metrics=pd.DataFrame(rows).sort_values("ROC-AUC",ascending=False)
            st.session_state["baseline_metrics"]=metrics
            st.session_state["baseline_probs"]={k:v.tolist() for k,v in probs.items()}
            st.session_state["baseline_yte"]=yte.tolist()
            st.session_state["baseline_threshold"]=threshold
            st.session_state["baseline_model_names"]=list(probs.keys())
            st.success("Training finished for this holdout split.")
    if "baseline_metrics" in st.session_state:
        metrics=st.session_state["baseline_metrics"]
        st.subheader("Model comparison")
        st.dataframe(metrics.round(3),use_container_width=True,hide_index=True)
        fig=px.bar(metrics.melt(id_vars="Model",value_vars=["ROC-AUC","Average precision"],var_name="Metric",value_name="Score"),
                   x="Model",y="Score",color="Metric",barmode="group",title="Discrimination metrics (higher is better)")
        fig.update_layout(height=430,yaxis_range=[0,1])
        st.plotly_chart(fig,use_container_width=True)
        yte=np.asarray(st.session_state["baseline_yte"])
        probs={k:np.asarray(v) for k,v in st.session_state["baseline_probs"].items()}
        fig=go.Figure()
        for name,p in probs.items():
            fpr,tpr,_=roc_curve(yte,p)
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{name} (AUC {roc_auc_score(yte,p):.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash",color="gray")))
        fig.update_layout(title="ROC curves",xaxis_title="False positive rate",yaxis_title="True positive rate",xaxis_range=[0,1],yaxis_range=[0,1],height=460)
        st.plotly_chart(fig,use_container_width=True)
        fig=go.Figure()
        for name,p in probs.items():
            precision,recall,_=precision_recall_curve(yte,p)
            fig.add_trace(go.Scatter(x=recall,y=precision,mode="lines",name=name))
        fig.add_hline(y=float(yte.mean()),line_dash="dash",annotation_text=f"Test-set prevalence {yte.mean():.2f}")
        fig.update_layout(title="Precision–recall curves",xaxis_title="Recall",yaxis_title="Precision",xaxis_range=[0,1],yaxis_range=[0,1],height=460)
        st.plotly_chart(fig,use_container_width=True)
        selected=st.selectbox("Confusion matrix for",list(probs.keys()))
        pred=(probs[selected]>=st.session_state["baseline_threshold"]).astype(int)
        cm=confusion_matrix(yte,pred,labels=[0,1])
        fig=px.imshow(cm,text_auto=True,x=["Predicted no ORN","Predicted ORN"],y=["Actual no ORN","Actual ORN"],
                      color_continuous_scale="Blues",title=f"Confusion matrix: {selected}")
        fig.update_layout(height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("<div class='warning'><b>Validation warning:</b> this is a single random holdout. It is not external validation, and the displayed threshold is a modelling choice—not a treatment threshold.</div>",unsafe_allow_html=True)
    else:
        st.info("Select model(s), set the split and threshold, then click **Train / refresh baseline models**.")

elif section == "Project stage & next steps":
    st.header("Where the research stands")
    rows=[
        ("Phase 1 — literature, cohort audit and data design","Foundation established","Real outcome cohort loaded; descriptive analysis and dataset audit available."),
        ("Phase 2 — preprocessing and feature extraction","Partially complete","Clinical/DVH features are usable. Patient-linked serial CT radiomics and biomarker tables are not supplied."),
        ("Phase 3 — baseline classifiers","Prototype available","Classical tabular models can be trained and evaluated on a holdout split."),
        ("LSTM on T0–T4 radiomics","Blocked by data","Needs one row per patient × timepoint with measured radiomics features and a consistent patient outcome."),
        ("ResNet-18 on CT crops","Not trained","Needs real CT crops/segmentations and verified labels linked to those same patients."),
        ("Temporal Fusion Transformer","Future stage","Requires aligned longitudinal imaging, dosimetry, clinical and biomarker branches."),
        ("Treatment triage recommendation","Not validated","Needs explicit treatment/response labels, clinical governance and prospective validation.")
    ]
    status=pd.DataFrame(rows,columns=["Workstream","Status","What supports this status"])
    st.dataframe(status,use_container_width=True,hide_index=True)
    st.subheader("What to request from the hospital")
    st.markdown("""
    1. Ethics-approved, de-identified patient key that links imaging, dose, outcomes and follow-up.
    2. CT/OPG scans and available CT grades at defined time windows.
    3. RT plans, RTSTRUCT/RTDOSE, and mandibular DVH.
    4. ORN diagnosis/onset, progression/grade, last follow-up and censoring.
    5. Clinical covariates such as dental extraction, primary site, stage, chemotherapy and tobacco exposure.
    6. Biomarkers only if actually measured and accessible; document assay, units and collection time.
    """)
    st.markdown("<div class='note'><b>Suggested next milestone:</b> agree on the endpoint, data dictionary and patient-linkage feasibility before training more complex models.</div>",unsafe_allow_html=True)
    st.markdown("<div class='warning'><b>Safety and governance:</b> research prototype only. No individual patient predictions or treatment recommendations are provided by this dashboard.</div>",unsafe_allow_html=True)

elif section == "Data dictionary":
    st.header("Explore the supplied data dictionary")
    st.markdown("Search columns and inspect their non-missing counts, data types, and example values.")
    search=st.text_input("Search variable names",placeholder="e.g. D30%, Smoking, ORN status")
    info=[]
    for c in df.columns:
        if search.lower() in c.lower():
            vals=df[c].dropna().astype(str).unique()[:5]
            info.append({"Variable":c,"Type":str(df[c].dtype),"Non-missing":int(df[c].notna().sum()),
                         "Missing":int(df[c].isna().sum()),"Example values":" · ".join(vals)})
    st.dataframe(pd.DataFrame(info),use_container_width=True,hide_index=True)
    cols=st.multiselect("Preview selected columns",list(df.columns),default=["ORN status","Age","D30%"] if all(c in df for c in ["ORN status","Age","D30%"]) else list(df.columns[:3]))
    if cols:
        st.dataframe(df[cols].head(30),use_container_width=True,hide_index=True)
    st.caption("Patient identifiers are not shown as a chart variable by default. Do not export or display identifiable patient-level data without authorization.")

st.divider()
st.caption("ORN Research Progress Dashboard · NMIMS MPSTME · Retrospective research prototype · Not medical advice or clinical decision support.")
