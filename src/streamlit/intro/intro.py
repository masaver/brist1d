from src.features.helpers.streamlit_helpers import load_markdown

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
import sys

from src.features.helpers.FakeDateCreator import FakeDateCreator
#from src.features.helpers.streamlit_helpers import load_markdown, display_notebook
from src.features.helpers.streamlit_helpers import extract_notebook_images, extract_code_cells

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
file_path = os.path.join(project_root, "data", "raw", "train.csv")

# load the dataset
@st.cache_data
def load_data(): 
  return pd.read_csv(file_path, low_memory=False, index_col=0)


def display_page():
      # Anchor at the top
      st.markdown("<a name='top'></a>", unsafe_allow_html=True)
      
      #st.title("Introduction") 

      st.markdown("# Introduction")
      st.markdown("""This project deals with the prediction of blood glucose levels one hour ahead of time of patients with type 1 diabetes. 
                    Predicting blood glucose fluctuations is crucial for managing type 1 diabetes. An accurate forecast can help alleviate some of the challenges faced by individuals with the condition.

## Description

### Type 1 Diabetes

Type 1 diabetes (T1D) is a chronic disease in which the body is no longer able to produce the hormone insulin. Insulin is required by the body to regulate the amount of glucose (sugar) in the bloodstream. Without treatment T1D results in high blood glucose levels which can cause symptoms like frequent urination, increased thirst, increased hunger, weight loss, blurry vision, tiredness, and slow wound healing. Ultimately high blood glucose levels will be fatal. In order to survive those suffering from T1D need to inject insulin to manage their blood glucose levels. Since also low blood glucose levels are potentially life-threatening and insulin counteracts the level of blood glucose it is important to establish a careful insulin management. There are many other factors that impact blood glucose levels, including eating, physical activity, stress, illness, sleep and alcohol. Hence calculating how much insulin to apply is complex. But the continuous need to think about how an action may impact blood glucose levels and what to do to counteract them is a significant burden for those with T1D.

### The Goal

Therefore developing algorithms which can reliably predict blood glucose levels in the future can play an important role in T1D management. Algorithms of varying levels of complexity have been developed that perform this prediction but the noisy nature of health data and the numerous unmeasured factors that impact the target mean there is a limit to how effective they can be. This project aims to use a newly collected dataset in order to predict blood glucose levels one hour ahead of time of patients with T1D.

                  
### The Dataset

The data used in this project was part of a kaggle competition (https://www.kaggle.com/competitions/brist1d) in which our team also participated. It is part of a bigger newly collected dataset of real-world data collected from young adults in the UK who suffer from T1D. All participants used continuous glucose monitors, insulin pumps and were given a smartwatch as part of the study to collect activity data. It is structured as follows:
""")


      # =======================================
      #st.markdown("## <a name='dataset-description-and-structure'></a> Dataset Description and Structure", unsafe_allow_html=True)

      st.markdown("""
        We're provided with two datasets: **Train** and **Test**, tailored for blood glucose prediction and for the model evaluation. 

        **Training Set:**
        - Comprises the first three months for 9 participants.
        - Includes blood glucose levels 1hr ahead of time for model training.
        - Samples are chronological with overlap.

        **Testing Set:**
        - Covers randomized samples for 15 participants, some of which are **not included** in the **Train** dataset.
        """)
    #   st.markdown("""
    #     ### Challenges:
    #     - **Unseen Participants:** Test set includes participants absent from training, adding complexity.
    #     - **Missing Data:** Incomplete or noisy medical data.
    #     - **Time Resolution:** Different time resolutions among patients.
    #     """)

      # Load the data
      patients = load_data()
      # Transform the data using the Helper class
      patients_d = FakeDateCreator.create_pseudo_datetime(patients)

      # Display dataset with toggle
      with st.expander("🔍 View Training Dataset (Top 10 Rows)"):
            st.dataframe(patients.head(10), use_container_width=True)

      # Compact statistics for quick insights
      with st.expander("🔢 Quick Insights"):
            
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Total Records:**")
                st.write(f"📊 `{len(patients):,}`")

                st.write("**Number of Columns:**")  
                st.write(f"📋 `{patients.shape[1]}`")

            with col2:
                st.write("**Unique Participants:**")
                st.write(f"👤 `{patients['p_num'].nunique()}`")

                st.write("**Date Range:**")
                st.write(f"📅 `{patients_d['pseudo_datetime'].min().date()} - {patients_d['pseudo_datetime'].max().date()}`")

      # Column Desciption
      # Define column description data as a dictionary
      column_data = {
            "#Column": ["1", "2", "3", "4-75", "76-147", "148-219", "220-291", "292-363", "364-435", "436-507", "508"],
            "Name": ["id", "p_num", "time", "bg-X:XX", "insulin-X:XX", "carbs-X:XX", "hr-X:XX", "steps-X:XX", "cals-X:XX", "activity-X:XX", "bg+1:00"],
            "Description": [
                "row id consisting of participant number and a count for that participant",
                "participant number",
                "time of day in the format HH:MM:SS",
                "blood glucose reading in mmol/L, X:XX(H:MM) time in the past",
                "total insulin dose received in units in the last 5 minutes, X:XX(H:MM) time in the past",
                "total carbohydrate value consumed in grammes in the last 5 minutes, X:XX(H:MM) time in the past",
                "mean heart rate in beats per minute in the last 5 minutes, X:XX(H:MM) time in the past",
                "total steps walked in the last 5 minutes, X:XX(H:MM) time in the past",
                "total calories burnt in the last 5 minutes, X:XX(H:MM) time in the past",
                "self-declared activity performed in the last 5 minutes, X:XX(H:MM) time in the past",
                "blood glucose levels 1 hour in the future, not provided in test.csv",               
            ],
            "Type": ["string", "string", "string", "float", "float", "float", "float", "float", "string", "string", "float"]
        }

      # Convert to DataFrame
      column_description_df = pd.DataFrame(column_data)

      # Display column description in an expander
      with st.expander("📋 Column Description"):
          st.table(column_description_df)

        
      # Simulated DataFrame for Time Resolution
      resolutions = pd.DataFrame({
            "p_num": ["p01", "p02", "p03", "p04", "p05", "p06", "p10", "p11", "p12"],
            "time_resolution_in_minutes": [15, 5, 5, 5, 15, 5, 15, 5, 5]
        })
      # Add a column for color grouping
      resolutions['color_group'] = resolutions['time_resolution_in_minutes'].map({5: "5 min", 15: "15 min"})

      # Display "Time Resolution" Section
      with st.expander("⏱️ Time Resolution"):
            #st.markdown("### Time Resolution per Patient in Training Data")
            
            plt.figure(figsize=(6, 4))
            sns.barplot(x="p_num", y="time_resolution_in_minutes", data=resolutions, hue="color_group", palette={"5 min": "skyblue", "15 min": "orange"})
            plt.title("Time Resolution per Patient in Training Data")
            plt.xlabel("Patient Number")
            plt.ylabel("Time Resolution (Minutes)")
            plt.tight_layout()
            
            # Render the plot in Streamlit
            st.pyplot(plt)
      st.markdown("""### Evaluation

The metric to evaluate the performance of our models will be the Root Mean Square Error (RMSE) of the predicted blood glucose levels an hour into the future and the actual values at that time.

The RMSE is defined as: 

$$ \\text{RMSE} = \sqrt{\\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

where $n$ is the number of samples, $\hat{y}_i$ is the $i$-th predicted value and $y_i$ is the $i$-th actual value.""")  

        # =======================================


#st.markdown(load_markdown(os.path.join(current_dir, "intro.md")))