import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
ARTIFACTS_DIR = ROOT / "Artifacts"
DATA_DIR = ROOT / "Processed Datasets"
PLOTS_DIR = ROOT / "EDA" / "Plots"

BASELINE_MODELS_PATH = ARTIFACTS_DIR / "baseline_models.joblib"
PREPROCESSORS_PATH = ARTIFACTS_DIR / "preprocessors.joblib"
BASELINE_RESULTS_PATH = ARTIFACTS_DIR / "base_result.joblib"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.joblib"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
SAMPLE_DATA_PATH = DATA_DIR / "final_train_cleaned.csv"
HF_DATASET_REPO = "Yuvaraj-Dey-2006/house-credit-processed"
HF_DATASET_FILENAME = "final_train_cleaned.csv"

# Model artifacts hosted on the same private HF repo, downloaded at
# runtime instead of being committed to git. Required artifacts raise
# if still missing after the download attempt; optional ones
# (best_params, final_model) are skipped silently if absent.
HF_ARTIFACT_REPO = "Yuvaraj-Dey-2006/house-credit-processed"  # can be same repo or a separate one
HF_REQUIRED_ARTIFACTS = {
    BASELINE_MODELS_PATH: "baseline_models.joblib",
    PREPROCESSORS_PATH: "preprocessors.joblib",
    BASELINE_RESULTS_PATH: "base_result.joblib",
}
HF_OPTIONAL_ARTIFACTS = {
    BEST_PARAMS_PATH: "best_params.joblib",
    FINAL_MODEL_PATH: "final_model.joblib",
}
SUN_ICON_PATH = APP_DIR / "assets" / "sun.svg"
MOON_ICON_PATH = APP_DIR / "assets" / "moon.svg"
PROFILE_IMAGE_PATH = APP_DIR / "assets" / "profile.png"

MODEL_OPTIONS = {
    "CatBoost Classifier": ("catbc_base", "pp_catbc", "catboost"),
    "LightGBM Classifier": ("lgbmc_base", "pp_lgbm", "lightgbm"),
    "XG Boost Classifier": ("xgbc_base", "pp_xg", "xgboost"),
    "Elastic Net log reg": ("sgd_base", "pp_elasticnet", "elasticnet"),
}


st.set_page_config(
    page_title="House Credit Prediction",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


def svg_mask_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'url("data:image/svg+xml;base64,{encoded}")'


def image_data_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def apply_theme(dark_mode):
    sun_icon = svg_mask_url(SUN_ICON_PATH)
    moon_icon = svg_mask_url(MOON_ICON_PATH)

    if dark_mode:
        colors = {
            "app_bg": "#0f172a",
            "panel_bg": "#111827",
            "soft_bg": "#1e293b",
            "sidebar_bg": "#172033",
            "sidebar_text": "#e6edf7",
            "sidebar_muted": "#b8c5d6",
            "tab_text": "#dbe8f6",
            "dataframe_bg": "#1f2937",
            "dataframe_header_bg": "#273548",
            "dataframe_header_text": "#f8fafc",
            "cell_even_bg": "#111827",
            "cell_odd_bg": "#1f2937",
            "dropzone_bg": "#0f172a",
            "dropzone_text": "#e2e8f0",
            "input_bg": "#1e293b",
            "input_text": "#e5edf6",
            "text": "#e5edf6",
            "muted": "#a8b3c2",
            "border": "#475569",
            "border_soft": "rgba(148, 163, 184, 0.7)",
            "accent": "#2dd4bf",
            "accent_soft": "#123a3c",
            "focus_ring": "#7dd3fc",
            "progress_track": "#334155",
            "theme_icon": moon_icon,
            "theme_icon_rotation": "360deg",
            "chart_1": "#60a5fa",
            "chart_2": "#2dd4bf",
            "chart_3": "#f87171",
            "chart_4": "#fbbf24",
        }
    else:
        colors = {
            "app_bg": "#ffffff",
            "panel_bg": "#ffffff",
            "soft_bg": "#f8fafc",
            "sidebar_bg": "#eef6f7",
            "sidebar_text": "#102338",
            "sidebar_muted": "#486176",
            "tab_text": "#0f172a",
            "dataframe_bg": "#f8fafc",
            "dataframe_header_bg": "#dfeaf7",
            "dataframe_header_text": "#162538",
            "cell_even_bg": "#ffffff",
            "cell_odd_bg": "#f8fafc",
            "dropzone_bg": "#ffffff",
            "dropzone_text": "#0f172a",
            "input_bg": "#eef2f7",
            "input_text": "#0f172a",
            "text": "#0f172a",
            "muted": "#475569",
            "border": "#c7d2e0",
            "border_soft": "rgba(148, 163, 184, 0.85)",
            "accent": "#0f766e",
            "accent_soft": "#f8fbfb",
            "focus_ring": "#2563eb",
            "progress_track": "#dfeaf7",
            "theme_icon": sun_icon,
            "theme_icon_rotation": "0deg",
            "chart_1": "#2563eb",
            "chart_2": "#60a5fa",
            "chart_3": "#dc2626",
            "chart_4": "#f59e0b",
        }

    css_vars = "\n".join(
        f"        --{key.replace('_', '-')}: {value};" for key, value in colors.items()
    )
    st.markdown(
        """
    <style>
    :root {
__CSS_VARS__
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--app-bg);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.45rem;
    }
    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--sidebar-text);
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--sidebar-muted);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--sidebar-text);
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border);
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        background: var(--input-bg);
        color: var(--input-text);
        border-color: var(--border);
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] svg,
    [data-testid="stSidebar"] input {
        color: var(--input-text);
        fill: var(--input-text);
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div {
        color: var(--sidebar-muted);
    }
    * {
        box-sizing: border-box;
    }
    button,
    [role="button"],
    input,
    textarea,
    select {
        outline: none !important;
        box-shadow: none !important;
        appearance: none !important;
    }
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    select:focus-visible,
    [role="button"]:focus-visible,
    [data-testid="stFileUploaderDropzone"]:focus-within,
    [data-testid="stDataFrame"]:focus-within,
    [data-testid="stTable"]:focus-within {
        outline: none !important;
        box-shadow: none !important;
        border-color: var(--border-soft) !important;
    }
    .vega-embed .vega-actions,
    .vega-embed .vega-actions *,
    .vega-embed .vega-bindings,
    .vega-embed .vega-bindings * {
        display: none !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[role="combobox"],
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] button,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div[role="button"] {
        background: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px dashed var(--border-soft) !important;
        border-radius: 8px;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] button *,
    div[data-baseweb="select"] * {
        color: var(--input-text) !important;
    }
    [data-testid="stNumberInput"] button {
        display: none !important;
    }
    [data-testid="stNumberInput"] input {
        -moz-appearance: textfield;
    }
    [data-testid="stNumberInput"] input::-webkit-inner-spin-button,
    [data-testid="stNumberInput"] input::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--dropzone-bg) !important;
        border: 1px dashed var(--border-soft) !important;
        border-radius: 10px;
        min-height: 3.1rem;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(180deg, #2563eb, #1d4ed8) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(37, 99, 235, 0.9) !important;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: none !important;
        min-height: 1.9rem;
        padding-top: 0.22rem;
        padding-bottom: 0.22rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        line-height: 1.2;
    }
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(180deg, #3b82f6, #2563eb) !important;
        border-color: #93c5fd !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploaderFile"] {
        background: var(--soft-bg);
        border: 1px dashed var(--border-soft);
        border-radius: 8px;
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderFile"] svg {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploaderDropzone"] > div * {
        background: transparent !important;
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderDropzone"] > div svg:not([data-testid="stFileUploaderFile"] svg) {
        fill: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] label,
    [data-testid="stFileUploaderDropzone"] div {
        background: transparent !important;
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderDropzone"] > div > div {
        background: transparent !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    .stDataFrame {
        background: var(--dataframe-bg);
        border: 1px dashed var(--border-soft);
        border-radius: 10px;
        overflow: hidden;
    }
    .static-table {
        background: var(--dataframe-bg);
        border: 1px dashed var(--border-soft);
        border-radius: 10px;
        overflow: auto;
        margin: 0.4rem 0 1rem;
    }
    .static-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    .static-table th,
    .static-table td {
        padding: 0.45rem 0.65rem;
        white-space: nowrap;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] * {
        --gdg-bg-cell: var(--cell-even-bg) !important;
        --gdg-bg-cell-medium: var(--cell-odd-bg) !important;
        --gdg-bg-header: var(--dataframe-header-bg) !important;
        --gdg-bg-header-has-focus: var(--dataframe-header-bg) !important;
        --gdg-bg-header-hovered: var(--dataframe-header-bg) !important;
        --gdg-bg-group-header: var(--dataframe-header-bg) !important;
        --gdg-bg-group-header-hovered: var(--dataframe-header-bg) !important;
        --gdg-text-header: var(--dataframe-header-text) !important;
        --gdg-text-group-header: var(--dataframe-header-text) !important;
    }
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stTable"] > div {
        background: var(--dataframe-bg);
    }
    div[data-testid="stDataFrame"] table,
    div[data-testid="stTable"] table {
        background: var(--dataframe-bg);
        color: var(--text);
    }
    div[data-testid="stDataFrame"] thead th,
    div[data-testid="stDataFrame"] thead tr,
    div[data-testid="stTable"] thead th,
    div[data-testid="stTable"] thead tr {
        background: var(--dataframe-header-bg) !important;
        color: var(--dataframe-header-text) !important;
        border-bottom: 1px solid var(--border-soft) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(odd) td,
    div[data-testid="stTable"] tbody tr:nth-child(odd) td {
        background: var(--cell-odd-bg) !important;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) td,
    div[data-testid="stTable"] tbody tr:nth-child(even) td {
        background: var(--cell-even-bg) !important;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stTable"] th,
    div[data-testid="stTable"] td {
        background: transparent !important;
        color: var(--text) !important;
        border-color: var(--border-soft) !important;
    }
    [data-testid="stProgress"] [data-baseweb="progress-bar"] {
        background: #ffffff !important;
        border: 1px solid var(--border);
        border-radius: 999px;
        overflow: hidden;
    }
    [data-testid="stProgress"] [data-baseweb="progress-bar"] > div {
        background: linear-gradient(90deg, var(--accent), #34d399) !important;
        border-radius: 999px;
    }
    .profile-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.65rem;
        padding: 2.65rem 0 1.05rem;
        margin-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }
    .profile-photo {
        width: 104px;
        height: 104px;
        border-radius: 999px;
        object-fit: cover;
        object-position: center 34%;
        border: 3px solid var(--accent);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.28);
    }
    .profile-name {
        color: var(--sidebar-text);
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
        text-align: center;
    }
    .main .block-container {
        padding-top: 1.4rem;
        max-width: 1240px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--text);
    }
    p, label, span, div {
        color: inherit;
    }
    div[data-testid="stMetric"] {
        background: var(--soft-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
    div[data-testid="stMetric"] label {
        color: var(--muted);
    }
    div[data-testid="stMetricValue"] {
        color: var(--text);
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.05;
        font-size: clamp(1.35rem, 2.1vw, 2rem);
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: var(--muted);
    }
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button p {
        color: var(--tab-text);
    }
    [data-testid="stTabs"] button[aria-selected="true"],
    [data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--accent);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--dropzone-bg);
        border-color: var(--border);
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section div {
        background: var(--dropzone-bg);
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p {
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploader"] button {
        background: #e5e7eb;
        border-color: #cbd5e1;
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderFile"] {
        color: var(--dropzone-text) !important;
    }
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFile"] p,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] span {
        color: var(--dropzone-text) !important;
    }
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbar"] button {
        background: #ffffff;
    }
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbar"] *,
    div[role="tooltip"],
    div[role="tooltip"] * {
        color: #0f172a !important;
    }
    div[role="tooltip"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1;
    }
    div[data-testid="stNumberInput"] input {
        background: var(--input-bg);
        color: var(--input-text);
    }
    .status-band {
        border: 1px solid var(--border);
        border-left: 6px solid var(--accent);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: var(--accent-soft);
        color: var(--text);
        margin: 0.4rem 0 1.2rem;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border-color: var(--border);
    }
    .theme-toggle-wrap {
        display: flex;
        justify-content: flex-end;
        min-height: 1px;
    }
    .st-key-theme_icon_button {
        align-items: flex-start;
        position: fixed;
        top: 0.72rem;
        left: 0.78rem;
        z-index: 999999;
        margin: 0;
    }
    .st-key-theme_icon_button div[data-testid="stButton"] {
        display: flex;
        justify-content: flex-start;
    }
    .st-key-theme_icon_button button {
        position: relative;
        display: grid;
        place-items: center;
        width: 46px;
        height: 46px;
        min-height: 46px;
        padding: 0;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--soft-bg);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        transition:
            background 180ms ease,
            border-color 180ms ease,
            box-shadow 180ms ease,
            transform 180ms ease;
    }
    .st-key-theme_icon_button button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.24);
    }
    .st-key-theme_icon_button button p {
        width: 0;
        height: 0;
        overflow: hidden;
        opacity: 0;
        position: absolute;
    }
    .st-key-theme_icon_button button::before {
        content: "";
        position: absolute;
        inset: 0;
        width: 28px;
        height: 28px;
        display: block;
        margin: auto;
        background-color: var(--text);
        -webkit-mask: var(--theme-icon) center / contain no-repeat;
        mask: var(--theme-icon) center / contain no-repeat;
        transition:
            transform 260ms ease,
            opacity 180ms ease,
            background-color 180ms ease;
        transform: rotate(var(--theme-icon-rotation)) scale(1);
    }
    .st-key-theme_icon_button button:hover::before {
        transform: rotate(var(--theme-icon-rotation)) scale(1.08);
        background-color: var(--accent);
    }
    </style>
        """.replace("__CSS_VARS__", css_vars),
        unsafe_allow_html=True,
    )


def ensure_model_artifacts():
    """Download missing model artifacts from the private Hugging Face
    dataset repository into the local Artifacts directory."""

    token = st.secrets.get("HF_TOKEN")

    if not token:
        st.error("HF_TOKEN is not configured in Streamlit Secrets.")
        return

    all_artifacts = {
        **HF_REQUIRED_ARTIFACTS,
        **HF_OPTIONAL_ARTIFACTS,
    }

    missing = {
        local_path: remote_filename
        for local_path, remote_filename in all_artifacts.items()
        if not local_path.exists()
    }

    if not missing:
        return

    try:
        from huggingface_hub import hf_hub_download

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        for local_path, remote_filename in missing.items():
            try:
                hf_hub_download(
                    repo_id=HF_ARTIFACT_REPO,
                    filename=remote_filename,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(ARTIFACTS_DIR),
                )

            except Exception as exc:
                if local_path in HF_REQUIRED_ARTIFACTS:
                    st.error(
                        f"Could not download required artifact "
                        f"`{remote_filename}`: {exc}"
                    )
                continue

    except Exception as exc:
        st.error(f"Could not connect to Hugging Face: {exc}")


@st.cache_resource(show_spinner="Loading trained artifacts...")
def load_artifacts():
    ensure_model_artifacts()
    missing = [
        path
        for path in [BASELINE_MODELS_PATH, PREPROCESSORS_PATH, BASELINE_RESULTS_PATH]
        if not path.exists()
    ]
    if missing:
        missing_names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required artifact(s): {missing_names}")

    return {
        "models": joblib.load(BASELINE_MODELS_PATH),
        "preprocessors": joblib.load(PREPROCESSORS_PATH),
        "results": joblib.load(BASELINE_RESULTS_PATH),
        "best_params": (
            joblib.load(BEST_PARAMS_PATH) if BEST_PARAMS_PATH.exists() else {}
        ),
        "final_model": (
            joblib.load(FINAL_MODEL_PATH) if FINAL_MODEL_PATH.exists() else None
        ),
    }


def ensure_sample_data():
    """Download the processed sample CSV from the private Hugging Face
    dataset repository into the local Processed Datasets folder."""

    if SAMPLE_DATA_PATH.exists():
        return

    token = st.secrets.get("HF_TOKEN")

    if not token:
        return

    try:
        from huggingface_hub import hf_hub_download

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_DATASET_FILENAME,
            repo_type="dataset",
            token=token,
            local_dir=str(DATA_DIR),
        )

    except Exception as exc:
        st.warning(f"Could not fetch sample dataset: {exc}")

@st.cache_data(show_spinner="Reading sample applicants...")
def load_sample_data(n_rows=5000):
    ensure_sample_data()
    if not SAMPLE_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_DATA_PATH, nrows=n_rows)


def get_expected_features(preprocessors):
    for preprocessor in preprocessors.values():
        feature_names = getattr(preprocessor, "feature_names_in_", None)
        if feature_names is not None:
            return list(feature_names)
    sample_df = load_sample_data()
    return [col for col in sample_df.columns if col not in {"TARGET", "SK_ID_CURR"}]


def get_active_features(artifacts):
    final_model = artifacts.get("final_model")
    if final_model:
        return final_model["feature_names"]
    return get_expected_features(artifacts["preprocessors"])


def prepare_features(df, expected_features):
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    for column in expected_features:
        if column not in cleaned.columns:
            cleaned[column] = np.nan

    return cleaned[expected_features]


def transform_for_model(features, preprocessor, model_family):
    transformed = preprocessor.transform(features)

    if model_family == "catboost" and isinstance(transformed, pd.DataFrame):
        object_columns = transformed.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        for column in object_columns:
            transformed[column] = transformed[column].astype(str)

    if model_family == "lightgbm" and isinstance(transformed, pd.DataFrame):
        object_columns = transformed.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            transformed[column] = transformed[column].astype("category")

    return transformed


def predict_credit_risk(raw_df, model_name, artifacts):
    final_model = artifacts.get("final_model")
    if final_model:
        expected_features = final_model["feature_names"]
        features = prepare_features(raw_df, expected_features)
        transformed = transform_for_model(
            features,
            final_model["preprocessor"],
            final_model["model_family"],
        )
        probabilities = final_model["model"].predict_proba(transformed)[:, 1]
        return pd.DataFrame(
            {
                "default_probability": probabilities,
                "risk_band": pd.cut(
                    probabilities,
                    bins=[-0.001, 0.2, 0.5, 1.0],
                    labels=["Low", "Moderate", "High"],
                ).astype(str),
            },
            index=raw_df.index,
        )

    model_key, preprocessor_key, model_family = MODEL_OPTIONS[model_name]
    expected_features = get_expected_features(artifacts["preprocessors"])
    features = prepare_features(raw_df, expected_features)
    transformed = transform_for_model(
        features,
        artifacts["preprocessors"][preprocessor_key],
        model_family,
    )
    probabilities = artifacts["models"][model_key].predict_proba(transformed)[:, 1]
    return pd.DataFrame(
        {
            "default_probability": probabilities,
            "risk_band": pd.cut(
                probabilities,
                bins=[-0.001, 0.2, 0.5, 1.0],
                labels=["Low", "Moderate", "High"],
            ).astype(str),
        },
        index=raw_df.index,
    )


def format_percent(value):
    return f"{value:.1%}"


def style_dataframe(dataframe, dark_mode):
    header_bg = "#273548" if dark_mode else "#dfeaf7"
    header_text = "#f8fafc" if dark_mode else "#162538"
    cell_bg = "#111827" if dark_mode else "#ffffff"
    alt_cell_bg = "#1f2937" if dark_mode else "#f8fafc"
    text = "#e5edf6" if dark_mode else "#0f172a"
    border = "rgba(148, 163, 184, 0.75)"
    return dataframe.style.set_properties(
        **{"background-color": cell_bg, "color": text, "border": f"1px solid {border}"}
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", header_bg),
                    ("color", header_text),
                    ("font-weight", "700"),
                    ("border", f"1px solid {border}"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [("background-color", alt_cell_bg)],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [("background-color", cell_bg)],
            },
            {
                "selector": "td",
                "props": [("border", f"1px solid {border}")],
            },
        ]
    ).hide(axis="index")


def render_static_table(styler, max_height=None):
    """Render a pandas Styler as real HTML instead of st.dataframe's canvas
    grid. st.dataframe (Glide Data Grid) ignores Styler table-level styles
    like header background/color, so header theming never applied there.
    This guarantees the header colors actually render."""
    html = styler.to_html()
    if max_height:
        html = (
            f'<div class="static-table" style="max-height:{max_height}px; '
            f'overflow-y:auto;">{html}</div>'
        )
    else:
        html = f'<div class="static-table">{html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_artifact_status(best_params, deployed_model_name, final_model):
    if final_model:
        text = (
            f"Deployed model: {deployed_model_name}. "
            f"This final artifact was refit on all {final_model['trained_rows']:,} cleaned training rows."
        )
        st.markdown(f"<div class='status-band'>{text}</div>", unsafe_allow_html=True)
        return

    if best_params:
        tuned_models = ", ".join(name.upper() for name in best_params)
        text = (
            f"Deployed model: {deployed_model_name}. "
            f"Tuning results found for {tuned_models}; scoring uses the best saved model artifact available."
        )
    else:
        text = f"Deployed model: {deployed_model_name}. Scoring uses the best saved baseline model artifact."
    st.markdown(f"<div class='status-band'>{text}</div>", unsafe_allow_html=True)


try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

sample_df = load_sample_data()
expected_features = get_active_features(artifacts)
results = artifacts["results"].copy()
results = results.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
model_name = (
    artifacts["final_model"]["model_name"]
    if artifacts["final_model"]
    else results.loc[0, "Models"] if not results.empty else "CatBoost Classifier"
)

if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = False

apply_theme(st.session_state.dark_theme)

st.title("House Credit Prediction")
st.caption(
    "Interactive credit-default risk scoring from the selected best Home Credit model."
)

render_artifact_status(artifacts["best_params"], model_name, artifacts["final_model"])

with st.sidebar:
    if st.button("Toggle theme", help="Switch theme", key="theme_icon_button"):
        st.session_state.dark_theme = not st.session_state.dark_theme
        st.rerun()
    st.markdown(
        f"""
        <div class="profile-card">
            <img class="profile-photo" src="{image_data_url(PROFILE_IMAGE_PATH)}" alt="Yuvaraj Dey profile photo">
            <div class="profile-name">Yuvaraj Dey</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Scoring")
    st.write(f"Deployed model: **{model_name}**")
    threshold = st.slider(
        "High-risk threshold",
        min_value=0.05,
        max_value=0.95,
        value=(
            float(artifacts["final_model"].get("threshold", 0.48))
            if artifacts["final_model"]
            else 0.50
        ),
        step=0.05,
    )
    st.divider()
    st.write(f"Expected features: **{len(expected_features)}**")
    st.write(f"Sample rows loaded: **{len(sample_df):,}**")

metric_cols = st.columns(4)
if not results.empty:
    best_row = results.iloc[0]
    metric_cols[0].metric("Best ROC-AUC", format_percent(best_row["ROC-AUC"]))
    metric_cols[1].metric("Recall", format_percent(best_row["RECALL"]))
    metric_cols[2].metric("Precision", format_percent(best_row["PRECISION"]))
    metric_cols[3].metric("Accuracy", format_percent(best_row["ACCURACY"]))

tab_score, tab_batch, tab_metrics, tab_eda = st.tabs(
    ["Applicant Score", "Batch Score", "Model Metrics", "EDA"]
)

with tab_score:
    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        st.subheader("Applicant")
        if sample_df.empty:
            st.warning("Sample data is unavailable. Use the batch tab to upload a CSV.")
            selected = pd.DataFrame()
        else:
            id_values = (
                sample_df["SK_ID_CURR"]
                if "SK_ID_CURR" in sample_df.columns
                else pd.Series(dtype="int64")
            )
            default_applicant_id = int(id_values.iloc[0]) if not id_values.empty else 0
            typed_applicant_id = st.number_input(
                "Applicant ID",
                min_value=0,
                value=default_applicant_id,
                step=1,
                format="%d",
            )
            matched_rows = sample_df.index[id_values == typed_applicant_id].tolist()

            if matched_rows:
                row_number = matched_rows[0]
                selected = sample_df.iloc[[row_number]]
                actual_target = selected.get("TARGET", pd.Series(["Unknown"])).iloc[0]
                st.metric("Sample row", row_number)
                st.metric("Actual target", actual_target)
            else:
                selected = pd.DataFrame()
                st.warning("Applicant ID was not found in the loaded sample rows.")

    with right:
        st.subheader("Prediction")
        if not selected.empty:
            prediction = predict_credit_risk(selected, model_name, artifacts)
            probability = float(prediction["default_probability"].iloc[0])
            decision = "High risk" if probability >= threshold else "Standard"

            score_cols = st.columns([0.35, 0.35, 0.30])
            score_cols[0].metric("Default probability", format_percent(probability))
            score_cols[1].metric("Risk band", prediction["risk_band"].iloc[0])
            score_cols[2].metric("Decision", decision)
            st.progress(min(max(probability, 0.0), 1.0))

            preview_cols = [
                col
                for col in [
                    "AMT_CREDIT",
                    "AMT_INCOME_TOTAL",
                    "AMT_ANNUITY",
                    "DAYS_BIRTH",
                    "DAYS_EMPLOYED",
                    "EXT_SOURCE_1",
                    "EXT_SOURCE_2",
                    "EXT_SOURCE_3",
                ]
                if col in selected.columns
            ]
            if preview_cols:
                render_static_table(
                    style_dataframe(selected[preview_cols], st.session_state.dark_theme)
                )

with tab_batch:
    st.subheader("Upload Applicants")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        missing_columns = sorted(set(expected_features) - set(upload_df.columns))
        predictions = predict_credit_risk(upload_df, model_name, artifacts)
        scored = pd.concat(
            [upload_df.reset_index(drop=True), predictions.reset_index(drop=True)],
            axis=1,
        )
        scored["above_threshold"] = scored["default_probability"] >= threshold

        batch_cols = st.columns(3)
        batch_cols[0].metric("Rows scored", f"{len(scored):,}")
        batch_cols[1].metric(
            "Average probability", format_percent(scored["default_probability"].mean())
        )
        batch_cols[2].metric(
            "Above threshold", f"{int(scored['above_threshold'].sum()):,}"
        )

        if missing_columns:
            st.warning(
                f"{len(missing_columns)} expected feature(s) were missing and filled with blank values."
            )

        render_static_table(
            style_dataframe(
                scored[
                    ["default_probability", "risk_band", "above_threshold"]
                    + list(upload_df.columns[:8])
                ],
                st.session_state.dark_theme,
            ),
            max_height=480,
        )
    else:
        st.info(
            "Upload a CSV with the same feature columns as the cleaned training data."
        )

with tab_metrics:
    st.subheader("Baseline Model Comparison")
    display_results = results.copy()
    for column in ["ACCURACY", "PRECISION", "RECALL", "F1", "ROC-AUC"]:
        display_results[column] = display_results[column].map(format_percent)
    render_static_table(style_dataframe(display_results, st.session_state.dark_theme))

    chart_data = results.set_index("Models")[["ROC-AUC", "RECALL", "PRECISION", "F1"]]
    chart_palette = (
        ["#60a5fa", "#2dd4bf", "#f87171", "#fbbf24"]
        if st.session_state.dark_theme
        else ["#2563eb", "#60a5fa", "#dc2626", "#f59e0b"]
    )
    st.bar_chart(chart_data, color=chart_palette)

    if artifacts["best_params"]:
        st.subheader("Saved Tuning Summary")
        tuning_rows = []
        for model_key, info in artifacts["best_params"].items():
            tuning_rows.append(
                {
                    "Model": model_key.upper(),
                    "Validation ROC-AUC": info.get("val_roc_auc"),
                    "Parameters": info.get("params"),
                }
            )
        render_static_table(
            style_dataframe(pd.DataFrame(tuning_rows), st.session_state.dark_theme)
        )

with tab_eda:
    st.subheader("Project EDA")
    plot_paths = sorted(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
    if plot_paths:
        selected_plot = st.selectbox(
            "Plot",
            options=plot_paths,
            format_func=lambda path: path.stem,
        )
        st.image(str(selected_plot), width="stretch")
    else:
        st.info("No EDA plot images were found.")