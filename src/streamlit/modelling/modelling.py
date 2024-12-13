import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
import sys

from src.features.helpers.FakeDateCreator import FakeDateCreator
# from src.features.helpers.streamlit_helpers import load_markdown, display_notebook
from src.features.helpers.streamlit_helpers import extract_notebook_images, extract_code_cells

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))


def display_page():
    st.title("Modelling")

    # =======================================
    st.markdown("## 2 Approach Overview")
    st.markdown("""
        The modelling phase aims to predict future blood glucose levels using machine learning algorithms.
        This section outlines the approach, including data preprocessing, feature engineering, model selection, and evaluation strategies.
        """)

    # =======================================
    st.markdown("### Modelling Approach 1")
    st.markdown("""
        **Data Preprocessing**:
        - **Time Alignment**: Resample data to 5-minute intervals for consistency.
        - **Missing Values**: Impute missing values and interpolate time series data.
        - **Outlier Detection**: Identify and handle extreme values.
        - **Feature Selection**: Choose relevant features for model training.
        - **Normalization**: Scale features to improve model performance.
        - **Data Split**: Divide data into training and validation sets.
        - **Target Variable**: Predict blood glucose levels one hour ahead.
        - **Feature Engineering**: Create lag features for time-dependent predictions.
        - **Model Selection**: Evaluate regression models for best performance.
        - **Hyperparameter Tuning**: Optimize model parameters for accuracy.
        - **Model Evaluation**: Assess model performance using metrics like RMSE.
        - **Feature Importance**: Identify key features influencing predictions.
        - **Model Interpretation**: Understand model decisions and predictions.
        """)

    # =======================================
    st.markdown("### Modelling Approach 2")
    st.markdown("""
        **Data Augmentation and Preprocessing**:
        - **Time Alignment**: Resample data to 5-minute intervals for consistency.
        - **Missing Values**: Impute missing values and interpolate time series data.
        - **Outlier Detection**: Identify and handle extreme values.
        - **Feature Selection**: Choose relevant features for model training.
        - **Normalization**: Scale features to improve model performance.
        - **Data Split**: Divide data into training and validation sets.
        - **Target Variable**: Predict blood glucose levels one hour ahead.
        - **Feature Engineering**: Create lag features for time-dependent predictions.
        - **Model Selection**: Evaluate regression models for best performance.
        - **Hyperparameter Tuning**: Optimize model parameters for accuracy.
        - **Model Evaluation**: Assess model performance using metrics like RMSE.
        - **Feature Importance**: Identify key features influencing predictions.
        - **Model Interpretation**: Understand model decisions and predictions.
        """)

    # =======================================
