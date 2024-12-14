import streamlit as st
import nbformat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.helpers.FakeDateCreator import FakeDateCreator
from src.features.helpers.streamlit_helpers import load_markdown, display_notebook

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "..", "data", "raw", "train.csv")

# setup the menu structure
st.sidebar.title("BrisT1D Blood Glucose Prediction Competition")
pages=["Kaggle Challenge", "Data Exploration", "Data Vizualization", "Modelling", "Predictions", "Conclusion & Perspectives"]
page=st.sidebar.radio("Menu", pages)


# load the dataset
@st.cache_data
def load_data(): 
  return pd.read_csv(file_path, low_memory=False)


st.sidebar.info(load_markdown(os.path.join(current_dir, "markdown", "team.md")))

if page == pages[0] : 
  
  st.markdown(load_markdown(os.path.join(current_dir, "markdown", "01-kaggle-competition.md")))
  

if page == pages[1] : 
  st.write("# Dataset Description and Structure")

  display_notebook(os.path.join(current_dir, "notebooks", "04-data-distribution.ipynb"))

if page == pages[2] : 
  st.write("# Data Vizualization")

  patients = load_data()

  # Transform the data using the Helper class
  patients = FakeDateCreator.create_pseudo_datetime(patients)

  #st.dataframe(patients.head(10), use_container_width=True)

  # Inputs for filtering
  # 1. Dropdown for p_num
  available_patients = list(patients['p_num'].unique()) + ["all"] # Get all unique patient IDs and additional 'all' option
  patient_id = st.selectbox("Select patient ID", options=available_patients, index=0)

  # 2. Calendar widget for 'specific_day'
  min_date = patients['pseudo_datetime'].min().date()  # Earliest date in the dataset
  max_date = patients['pseudo_datetime'].max().date()  # Latest date in the dataset
  specific_day = st.date_input("Select a day", value=min_date, min_value=min_date, max_value=max_date)

  # Filter the data
  filtered_data = FakeDateCreator.filter_patient_data(patients, patient_id, specific_day)

  # Plot the data if any rows are returned
  if not filtered_data.empty:
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(filtered_data['pseudo_datetime'], filtered_data['bg+1:00'], label='Glucose Levels', color='red')

      # Format the x-axis
      ax.set_xlim([pd.to_datetime(specific_day), pd.to_datetime(specific_day) + pd.Timedelta(days=1)])
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
      ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

      # Improve plot appearance
      ax.set_xlabel('Time (HH:MM)')
      ax.set_ylabel('Blood Glucose (mmol/L)')
      ax.set_title(f'Blood Glucose Over Time on {specific_day} for Patient {patient_id}')
      ax.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()

      # Display plot in Streamlit
      st.pyplot(fig)
  else:
      st.write(f"No data available for patient {patient_id} on {specific_day}.")


if page == pages[3] : 
  st.write("# Modelling")


if page == pages[4] : 
  st.write("# Predictions")

if page == pages[5] : 
  st.write("# Conclusions & Perspectives")