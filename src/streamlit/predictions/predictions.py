import streamlit as st
import pandas as pd
import numpy as np
import os

# from src.features.helpers.streamlit_helpers import load_markdown, display_notebook
from src.features.helpers.streamlit_helpers import  load_model_data, display_rand_row, rand_daily_profile, global_predictions

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

def display_page():
    st.title("Predictions")

    # =======================================
    st.markdown("## Overview")
    st.markdown("""
        Below, we provide an example of what  the predictions of the two models look like. 
        In the first example, we show predictions for single data points or timestamps.
        In the secon example, we show the hourly predcited and real bg+1:00 values.
        """)

    # =======================================
    st.markdown("### Single Data point Prediction")
    options = ["Model 1 - MaverickSensor", "Model 2 - RapidJuxtapose"]
    selected_option = st.selectbox("Choose a model", options)

    # Display the selected option
    st.write(f"You selected: {selected_option}")
    model , features , target, _ = load_model_data( selected_option )

    st.markdown("""Click the button below to predict on a random data point ( time stamp )""")
    if st.button( "Generate Prediction" , key = "generate_prediction" ):
        st.write( display_rand_row( features , target , model ) )


    # =======================================
    st.markdown("#### Random daily profile")
    p_id_options = ['p01', 'p02', 'p04', 'p05', 'p06', 'p10', 'p11', 'p12']
    p_id_selected_option = st.selectbox("Choose a patient ID", p_id_options)
    common_preds_df = global_predictions()

    st.markdown("""Click the button below to predict on a random data point ( time stamp )""")
    if st.button("'Display random daily profile'",key='random_daily_profile'):
        fig , _ = rand_daily_profile( p_id_selected_option , common_preds_df )
        st.pyplot( fig )