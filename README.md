# ORN Progression Intelligence

### A Multimodal AI Research Framework for Osteoradionecrosis of the Jaw

**Longitudinal risk modelling · Radiation dosimetry · CT radiomics ·
Explainable clinical research**

*Research project in development · NMIMS MPSTME · Tata Memorial Hospital
collaboration context*


![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Interactive%20Visuals-Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Local%20Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-0F766E)
![Clinical](https://img.shields.io/badge/Clinical%20Use-Not%20Validated-B45309)


**From retrospective clinical and dosimetric analysis toward
longitudinal, multimodal ORN research.**

------------------------------------------------------------------------

## ✦ The idea

Osteoradionecrosis of the jaw (ORNJ) is a serious complication
associated with head-and-neck radiotherapy. Its development and
progression are influenced by interacting clinical and treatment-related
factors, making it a challenging problem for static, single-modality
prediction.

**ORN Progression Intelligence** is a research framework designed to
investigate whether longitudinal clinical data, radiation dose--volume
information, CT-derived radiomics, and---where available---biomarkers
can be combined to better characterize ORN risk and progression over
time.

The long-term research direction is a multimodal, time-aware modelling
pipeline with interpretable outputs that can support scientific
investigation and, only after appropriate validation, future
clinician-led evaluation.

> **Current stage:** the real-data cohort audit, exploratory analysis,
> baseline clinical/dosimetric modelling workflow, and local interactive
> dashboard are in place. The full longitudinal imaging and multimodal
> model has **not** yet been trained because verified, patient-linked
> serial CT/radiomics and biomarker data are not currently available in
> the working dataset.

## ✦ At a glance

  -----------------------------------------------------------------------
  Area                                Current status
  ----------------------------------- -----------------------------------
  Retrospective clinical/dosimetric   Available for exploratory analysis
  cohort                              

  Cohort size in supplied dataset     **1,129 records**

  Recorded ORN events                 **198 (17.5%)**

  Available variables                 **61 columns**, including
                                      clinical/outcome and dose--volume
                                      fields

  Exploratory data analysis           Implemented

  Clinical/DVH baseline classifiers   Prototype workflow implemented

  Interactive Plotly dashboard        Local Streamlit app implemented

  Patient-linked T0--T4 radiomics     Not available in the current
                                      working data

  ResNet-18 on ORN-labelled CT crops  Not trained

  LSTM on longitudinal radiomics      Not trained on verified linked
                                      sequences

  Full multimodal Temporal Fusion     Future milestone
  Transformer                         

  Clinical deployment                 Not validated; not intended for
                                      clinical use
  -----------------------------------------------------------------------

## ✦ What exists today

### 1. Real-cohort exploration

The supplied retrospective cohort supports the current exploratory work,
including:

-   Cohort size and recorded ORN prevalence
-   Patient and tumour characteristics
-   Radiation dose--volume distributions
-   Missingness and selected-variable correlation analysis
-   Data-quality checks and outcome-distribution review

The current dataset contains clinical and dosimetric variables. Its
presence does **not** imply that corresponding serial CT images,
radiomics values, or biomarkers are available for the same patients.

### 2. Baseline modelling

The baseline workflow explores conventional tabular classifiers using
available clinical and dose--volume information. Depending on the
environment and installed packages, the workflow supports logistic
regression, Random Forest, Extra Trees, and XGBoost when available.

Evaluation includes metrics and visual diagnostics such as:

-   ROC-AUC
-   Average precision and precision--recall curves
-   Precision, recall, and F1 at a stated threshold
-   Brier score / probability calibration views
-   Confusion matrices

These are **exploratory retrospective baselines**, not clinical
performance claims. Results from different notebooks may use different
data splits and preprocessing choices; they should not be compared as
though they came from one locked experiment. A consistent protocol,
patient-level partitioning, repeated or nested validation, uncertainty
intervals, leakage checks, and external validation remain important next
steps.

### 3. Local interactive Plotly dashboard

The dashboard is built with **Streamlit + Plotly** so the research can
be explored in a browser on a local machine.

The dashboard includes:

-   **Executive overview** --- cohort summary and current project status
-   **Who is in the cohort?** --- age and categorical subgroup views
-   **Dose--volume analysis** --- selectable DVH metrics and group
    comparisons
-   **Data quality & relationships** --- missingness, correlations, and
    dataset audit
-   **Baseline model laboratory** --- configurable holdout split, model
    selection, threshold, and performance plots
-   **Project stage & next steps** --- current status and data
    requirements
-   **Data dictionary** --- searchable variable names, types,
    missingness, and sample values

Each major chart is accompanied by plain-language guidance explaining
what it shows, how to read it, why it matters, and where interpretation
should be cautious.

### 4. Research architecture under consideration

The planned research direction is multimodal and longitudinal:

``` text
Clinical history ───────────┐
Radiotherapy / DVH ─────────┤
Serial CT / radiomics ──────┼──> Time-aware multimodal modelling
Biomarkers, if available ───┘          │
                                       ├──> Risk / progression estimates
                                       ├──> Model interpretation
                                       └──> Research evaluation
```

Potential model components include a classical clinical/dosimetric
baseline, an LSTM for linked longitudinal radiomics, a single-timepoint
ResNet-18 imaging ablation, and a later Temporal Fusion Transformer
(TFT) if the data volume, temporal coverage, and outcome definitions
justify it.

**These are planned or data-dependent components---not all completed
models.**

------------------------------------------------------------------------

## ✦ Data: what is available, and what is missing?

### Currently available

The working analysis uses the supplied `TTE_ORN_data_Figshare.csv`
cohort, containing 1,129 records and 61 columns. The data include the
recorded ORN outcome and clinical/dosimetric variables, including DVH
measurements.

A separate radiomics spreadsheet provides selected **feature
definitions/names**. It is not a patient-level radiomics matrix and does
not contain the measured T0--T4 feature sequences needed to train the
proposed LSTM.

### Still required for longitudinal imaging models

To train the LSTM or ResNet-18 for ORN prediction, the project needs
real imaging features or CT crops linked to verified outcomes for the
**same patients**.

For a five-timepoint LSTM, the intended input schema is conceptually:

  --------------------------------------------------------------------------------
  patient_id      timepoint          ORN status   radiomics_feature_1 ...
  --------------- ------------ ---------------- --------------------- ------------
  De-identified   T0             Verified label        Measured value ...
  ID                                                                  

  De-identified   T1               Same patient        Measured value ...
  ID                                    outcome                       
                                     definition                       

  De-identified   T2               Same patient        Measured value ...
  ID                                    outcome                       
                                     definition                       

  De-identified   T3               Same patient        Measured value ...
  ID                                    outcome                       
                                     definition                       

  De-identified   T4               Same patient        Measured value ...
  ID                                    outcome                       
                                     definition                       
  --------------------------------------------------------------------------------

This is an **illustration of the required schema**, not a claim that
such a complete file currently exists.

Do not create missing measurements, fabricate timepoints, or assign ORN
labels from one cohort to unrelated CT images from another cohort.
Public CT datasets can support imaging-pipeline development, but
supervised ORN prediction requires verified outcome linkage and
compatible endpoints.

### High-value data requirements for the next collaboration milestone

Subject to institutional approvals, ethics requirements, data-use
agreements, and de-identification, the most useful data linkage would
include:

1.  A consistent de-identified patient key across modalities
2.  CT scans and available jaw/mandible segmentations at defined
    timepoints
3.  Radiotherapy plans, RT structures, RT dose, and mandibular DVH
4.  ORN diagnosis, onset, grade/progression, follow-up, and censoring
    information
5.  Relevant clinical covariates, with definitions and units
6.  Biomarker measurements and collection times, if available

A clear data dictionary and endpoint definition should precede
multimodal training.

------------------------------------------------------------------------

## ✦ Scientific interpretation and guardrails

-   **Association is not causation.** A dose--volume difference between
    outcome groups does not prove that a dose metric independently
    causes ORN.
-   **A holdout score is not external validation.** Report the split,
    endpoint, prevalence, threshold, and uncertainty alongside model
    metrics.
-   **Calibration matters.** A model can rank cases reasonably yet
    produce probabilities that do not match observed event rates.
-   **Avoid leakage.** Exclude post-outcome variables from predictors;
    fit imputation, scaling, feature selection, and threshold tuning
    only within the training process.
-   **Benchmark fairly.** Published PREDMORN or other NTCP results are
    contextual references unless population, endpoint, predictors,
    follow-up, and evaluation design are made comparable.
-   **Keep clinical governance central.** No output from this research
    prototype should be used to diagnose ORN, guide treatment, or
    replace clinician judgement.

## ✦ Current roadmap

  -----------------------------------------------------------------------
  Stage                   Goal                    Status
  ----------------------- ----------------------- -----------------------
  **Phase 1**             Literature review,      Foundation established
                          cohort audit,           
                          endpoint/data design    

  **Phase 2**             Data preprocessing, DVH Partially complete
                          analysis, feature       
                          preparation             

  **Phase 3**             Clinical/dosimetric     Exploratory prototype
                          baseline models and     
                          diagnostics             

  **Next**                Confirm access to       Data feasibility
                          patient-linked serial   priority
                          CT/radiomics and        
                          outcomes                

  **Then**                Train and validate LSTM Blocked by data
                          / ResNet-18 using       availability
                          appropriate linked data 

  **Later**               Evaluate multimodal TFT Depends on data
                          and explainability      coverage and sample
                                                  size

  **Future research**     External validation and Not yet completed
                          clinician-led           
                          evaluation              
  -----------------------------------------------------------------------

## ✦ Run the local dashboard

### Requirements

-   Python 3.10--3.12 recommended
-   The project folder containing `app.py`, `requirements.txt`, and
    `TTE_ORN_data_Figshare.csv`

### Windows

1.  Download and extract the dashboard package.
2.  Open the extracted folder.
3.  Double-click `run_dashboard.bat`.

Or run in PowerShell from the project folder:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, typically
`http://localhost:8501`.

### Dashboard files

``` text
ORN_Tata_Local_Dashboard/
├── app.py
├── requirements.txt
├── run_dashboard.bat
├── README.md
└── TTE_ORN_data_Figshare.csv
```

The bundled CSV makes the dashboard self-contained for the current
cohort. Keep patient-level or restricted data out of public
repositories; the included dataset should only be shared if its licence
and data-use conditions permit it.

------------------------------------------------------------------------

## ✦ Suggested next milestone

**Move from a tabular prototype to a validated longitudinal dataset.**

Before adding model complexity, align with the clinical collaborators on
the ORN endpoint, timepoint definitions, imaging availability, dose
linkage, follow-up/censoring, and feasible cohort size. Then lock a
reproducible baseline evaluation protocol. This sequence will make any
later LSTM, ResNet-18, or TFT result substantially more defensible.

------------------------------------------------------------------------

### Built for careful clinical AI research

**ORN Progression Intelligence**\
*An evolving research prototype --- transparent about what is measured,
what is modelled, and what remains to be validated.*
