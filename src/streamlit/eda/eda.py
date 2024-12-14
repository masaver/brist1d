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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
file_path = os.path.join(project_root, "data", "raw", "train.csv")


# load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(file_path, low_memory=False)


def display_page():
    st.title("Exploratory Data Analysis")

    # =======================================
    st.markdown("## Dataset Description and Structure")

    st.markdown("""
        We're provided with two datasets: **Train** and **Test**, tailored for blood glucose prediction and for the model evaluation.

        **Training Set:**
        - Comprises the first three months for 9 participants.
        - Includes future blood glucose values for model training.
        - Samples are chronological with overlap.

        **Testing Set:**
        - Covers the later study period for 15 unseen participants.
        - Randomized, non-overlapping samples.
        """)
    st.markdown("""
        ### Challenges:
        - **Missing Data:** Incomplete or noisy medical data.
        - **Device Variability:** Different CGM, insulin pump, and smartwatch models introduce variability.
        - **Unseen Participants:** Test set includes participants absent from training, adding complexity.
        """)

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
        "Name": ["id", "p_num", "time", "bg-X:XX", "insulin-X:XX", "carbs-X:XX", "hr-X:XX", "steps-X:XX", "cals-X:XX", "activity-X:XX", "bg-X:XX+1"],
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
            "blood glucose reading in mmol/L, X:XX+1(H:MM) time in the future, not provided in test.csv",
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
        # st.markdown("### Time Resolution per Patient in Training Data")

        plt.figure(figsize=(6, 4))
        sns.barplot(x="p_num", y="time_resolution_in_minutes", data=resolutions, hue="color_group", palette={"5 min": "skyblue", "15 min": "orange"})
        plt.title("Time Resolution per Patient in Training Data")
        plt.xlabel("Patient Number")
        plt.ylabel("Time Resolution (Minutes)")
        plt.tight_layout()

        # Render the plot in Streamlit
        st.pyplot(plt)

    # =======================================
    st.markdown("## Quality Control and Assurance")
    st.markdown("### Data Consistency")

    st.markdown("""
            Before starting the analysis, we validated the dataset to ensure consistency and reliability.
            The training dataset consists of daily time series for each patient, and the validation focused on checking the sequential order of rows and consistency of lag features.

            #### Validation Approach:
            - Compared lag features row-by-row with the preceding rows.
            - Flagged gaps or inconsistencies in the time series data.
            """)

    with st.expander("✅ Data Consitency Control"):
        # Simulated Validation Results DataFrame
        validation_results = pd.DataFrame({
            "hr": [0, 0, 0, 0, 0, 0, 0, 68, 0],
            "steps": [0, 0, 0, 0, 0, 0, 0, 26, 0],
            "cals": [0, 0, 0, 0, 0, 0, 0, 45, 0],
            "total": [16865, 26335, 26427, 25047, 16248, 16674, 25874, 25205, 26048]
        }, index=["p01", "p02", "p03", "p04", "p05", "p06", "p10", "p11", "p12"])

        st.table(validation_results)

        # Plotting the inconsistencies for visualization
        st.markdown("#### Inconsistencies Detected")
        plt.figure(figsize=(8, 4))
        melted_results = validation_results.drop(columns=["total"]).reset_index().melt(id_vars="index", var_name="Parameter", value_name="Inconsistencies")
        sns.barplot(data=melted_results, x="index", y="Inconsistencies", hue="Parameter", palette="viridis")
        plt.title("Inconsistencies by Parameter and Patient")
        plt.xlabel("Patient ID")
        plt.ylabel("Number of Inconsistencies")
        plt.legend(title="Parameter")
        plt.tight_layout()
        st.pyplot(plt)

        # Conclusion
        st.markdown("""
            #### Conclusion
            - The dataset is highly reliable, with only a few inconsistencies detected.
            - **Patient p11** shows minor issues in `hr`, `steps`, and `cals` due to overlapping time values.
            - These can be fixed by shifting the datetime index for specific rows.
            - No discrepancies were found in the target column (`bg+1:00`), confirming dataset validity.
            """)

    # Outlier and Anomaly Section
    st.markdown("### Outliers and Anomalies")
    st.markdown("""
            Outliers and anomalies were analyzed for all variable groups to identify extreme values or ouliers.
            Below are the key findings for each variable group.
            #### 📝 Notes:
            - **IQR Method Limitation**: The commonly used IQR (Interquartile Range) method for outlier detection was not applicable
                to this dataset due to its **skewed distribution**.
            - **Alternative Approach**: Instead, statistical and visual methods were used:
                - Extreme values were inspected using `describe()` to identify minimum, maximum, and statistical ranges.
                - Distributions were analyzed using histograms and density plots for more detailed insights.
            """)

    # Simulated summary of outlier analysis (replace with actual data as needed)
    outlier_summary = pd.DataFrame({
        "Variable Group": ["bg", "insulin", "carbs", "hr", "steps", "cals"],
        "Key Findings": [
            "Extreme values (2.2 to 27.8 mmol/L) observed, but still realistic for some patients.",
            "Negative values detected for Patient p12 (e.g., -0.3078). Positive extremes found up to 46.311 units.",
            "Values range from 1.0 to 852.0 grams. Many missing values; potential exclusion from the model.",
            "Heart rate ranges from 37.6 to 185.3 bpm. Unusually low HR for p06 and high HR for p02.",
            "Steps range from 0 to 1359 steps. Extreme value for p02 during intense walking activity.",
            "Values (0.03 to 116.1 kcal) appear realistic, considering potential physical activities."
        ]
    })

    with st.expander("📊 Outliers and Anomalies"):

        # Display the Summary Table
        st.table(outlier_summary)

        # Additional Notes or Visualizations
        st.markdown("""
            #### 📝 Notes:
            - Negative insulin values for Patient p12 are likely due to recording issues and will be replaced during preprocessing.
            - Extreme values for heart rate (p06, p02) should be carefully considered in a real life, as they might indicate valid clinical signals.
            - The `carbs` variable group has high missingness (98%) and may be excluded from modeling.
            """)

    # Missing Values
    st.markdown("### Missing Values")
    st.markdown("""
            Missing values were analyzed across all variable groups to ensure data completeness, a fundamental aspect of data quality.
            Handling missing values appropriately is crucial for ensuring robust model performance.
            Below is a summary of missing values for each variable group along with the proposed handling strategies:
            """)

    # Simulated summary of missing values analysis (replace with actual data as needed)
    missing_summary = pd.DataFrame({
        "Variable Group": ["bg", "insulin", "carbs", "hr", "steps", "cals", "activity", "bg+1:00"],
        "Missing Percentage": [
            "1.5% - 15.4%",
            "~5.3%",
            "~98.5%",
            "~29.1%",
            "~54%",
            "~20%",
            "~98.4%",
            "0%"
        ],
        "Handling Strategy": [
            "Interpolation and imputation during preprocessing.",
            "Interpolation and imputation during preprocessing.",
            "Likely dropped due to high missingness.",
            "Interpolation for smartwatch data; imputation for remaining values.",
            "Interpolation for smartwatch data; remaining values replaced with 0.",
            "Interpolation for smartwatch data; remaining values replaced with 0.",
            "Evaluate feature's utility for predictions; drop if insignificant.",
            "No missing values; no action required."
        ]
    })

    # Missing Values Section
    with st.expander("🚨 Missing Values"):
        # Display the Summary Table
        st.table(missing_summary)

    # Additional Notes
    st.markdown("""
            #### 📝 Notes:
            - The **bg+1:00** target variable has no missing values, requiring no further action.
            - The **carbs** and **activity** groups have the highest missing percentages (~98.5% and ~98.4%). These variables may be excluded if they do not improve model predictions.
            - For all other variables, interpolation and imputation methods will be applied during the preprocessing phase to address missing values.
            - The commonly used **IQR (Interquartile Range)** method for outlier detection was avoided due to skewed data distributions.
            """)

    # Summary
    st.markdown("### Summary")
    st.markdown("""
            This dataset presents challenges such as inconsistent time intervals, missing data, outliers, skewed distributions, and potential biases.
            The proposed solution is to resample data to 5-minute intervals while addressing missing values, outliers, and multicollinearity during preprocessing. Biases related to selection, measurement, and sampling will require careful interpretation of results.
            """)
    with st.expander("📜 Summary Key Points"):
        st.markdown("""
            1. **Data Inconsistencies**:
                - Different time intervals: Patient data recorded at either 5-minute or 15-minute intervals.
                - **Solution**: Up-sampling to 5-minute intervals with interpolation for 15-minute patients.
            2. **Missing Values**:
                - High missingness in variables like `carbs-*` and `activity-*`, which will be dropped for modeling.
                - Remaining missing values in other features will be imputed and interpolated accordingly.
            3. **Potential Biases**:
                - **Selection Bias**: Results may not generalize if participants are not diverse.
            """)

    # =======================================
    st.markdown("## Data Distribution")

    st.markdown("""
            This section provides insights into the distributions of both independent variables (features) and the target variable across the dataset.
            Visualizations highlight variability, skewness, and patterns within patient data, helping to identify key trends and relationships.
            """)

    relative_path = os.path.join("reports", "final-report", "02-exploratory-data-analysis", "04-data-distribution.ipynb")
    ntbk_path = os.path.join(project_root, relative_path)

    # Extract images from the notebook
    images = extract_notebook_images(ntbk_path)
    # Extract code cells
    code_cells = extract_code_cells(ntbk_path)

    with st.expander("🔢 Numeric Features and the Target Variable"):

        tab1, tab2 = st.tabs(["📈 Feature Distributions", "💻 Code"])

        with tab1:
            st.image(images[0])
            st.markdown("""
                    1. **Blood Glucose Levels (bg and bg+1:00)**:
                        - Variability exists across patients, with most distributions having similar central tendencies.
                        - Positively skewed distributions, clustering in lower ranges.
                    2. **Insulin and Carbs**:
                        - Right-skewed distributions with occasional outliers, e.g., patient p03 shows higher carb consumption and insulin usage.
                    3. **Heart Rate (hr)**:
                        - Relatively consistent across patients, clustering between 60–100 bpm.
                        - Slightly right-skewed but near-normal distribution.
                    4. **Steps**:
                        - Large variability, with patients like p03 and p05 showing higher median activity levels.
                        - Positively skewed distributions with finer granularity for 5 min resolution.
                    5. **Calories Burned (cals)**:
                        - Consistent across patients with fewer outliers.
                        - Most values cluster in lower ranges, with the 5 min resolution capturing finer details.
                """)

            with tab2:
                code = code_cells[2]
                st.code(code, language="python")

    with st.expander("🏃 Activity Levels"):

        tab1, tab2 = st.tabs(["🔥 Heatmap", "💻 Code"])

        with tab1:
            st.image(images[1])

            st.markdown("""
                    **Activity Patterns**:
                    - Activities like “Walk” are commonly logged across patients, with distinct patterns for individuals.
                    - Heatmap shows diverse activity patterns, e.g., patient p01 logs many activities, while p10 specializes in running.
                    """)

        with tab2:
            code = code_cells[3]
            st.code(code, language="python")
    st.markdown("""
            #### Conclusion:
            The dataset reveals significant variability and skewed distributions across features, emphasizing the need for personalized analysis.
            Finer granularity from the 5 min resolution enhances detail, while individual activity patterns underline the complexity of patient behavior.
            """)

    # =======================================
    st.markdown("## Data Correlation")

    st.markdown("""
            This section analyzes and visualizes relationships between independent variables and the target variable **``bg+1:00``**.
            Correlations help identify features with the strongest influence on predicting future blood glucose levels, guiding feature selection for modeling.
            """)

    relative_path = os.path.join("reports", "final-report", "02-exploratory-data-analysis", "05-data-correlation.ipynb")
    ntbk_path = os.path.join(project_root, relative_path)

    # Extract images from the notebook
    images = extract_notebook_images(ntbk_path)

    with st.expander("🔗 Global Correlations Between Lag Features and Target Variable"):
        st.image(images[0])

        st.markdown("#### Key Observations")
        st.markdown("""
            Heatmap of all numerical variables shows strong self-correlations (e.g., ``bg`` lag features) and weaker inter-variable relationships.
            """)

    with st.expander("🔗 Correlation of the Target Variable Against All Features"):
        st.image(images[1])

        st.markdown("#### Key Observations")
        st.markdown("""
            This heatmap focuses on ``bg+1:00``, highlighting the dominance of lagged bg features.
            """)

    with st.expander("🔗 Correlation Between All Numeric Variables as a Time Series"):
        st.image(images[4])

    # Key Observations
    st.markdown("#### Summary")
    st.markdown("""
            1. **Lagged ``bg`` Features**:
                - Strong correlations with ``bg+1:00`` highlight the temporal dependency of blood glucose levels.
            2. **Carbs and Insulin**:
                - Moderate correlations with ``bg+1:00`` reflect their role in glucose regulation.
                - A notable correlation between carbs and insulin emphasizes their dietary relationship.
            3. **Activity Metrics (Steps, Cals, HR)**:
                - Weak correlations with ``bg+1:00`` indicate an indirect or limited impact.
            """)

    # =======================================
    st.markdown("## Data Vizualization")

    # Patient Time Series Overview Section
    with st.expander("⏳ Patient Time Series Overview"):

        relative_path = os.path.join("reports", "final-report", "02-exploratory-data-analysis", "03-patient-timeseries-overview.ipynb")
        ntbk_path = os.path.join(project_root, relative_path)

        # Extract images from the notebook
        images = extract_notebook_images(ntbk_path)
        # Extract code cells
        code_cells = extract_code_cells(ntbk_path)

        st.markdown("""
            This section explores individual patients' time series data to uncover daily and weekly patterns,
            providing insights into fluctuations and trends in key metrics such as blood glucose, heart rate, and activity levels.
            """)

        tab1, tab2 = st.tabs(["👤📊 Individual Trends", "💻 Code"])

        with tab1:
            # Visualization Section
            st.markdown("#### Example Visualization: Patient p01 - Day 21")
            st.markdown("""
                The following plot illustrates the relationships between blood glucose levels, carbs and insulin intake,
                calories burned, and heart rate for Patient p01 on Day 21.
                """)

            st.image(images[0])

            st.markdown("#### Key Observations")
            st.markdown("""
                1. **Blood Glucose (BG):**
                    - Significant fluctuations observed throughout the day.
                    - Spikes in BG levels align with meal times, managed through insulin doses and carbohydrate intake.
                2. **Carbs and Insulin:**
                    - Insulin closely aligns with carbohydrate intake to stabilize post-meal glucose levels.
                    - Reflects typical diabetes management practices.
                3. **Calories and Heart Rate:**
                    - Morning activity shows increased calories burned and elevated heart rate.
                    - Suggests physical activity contributes to glucose stability.
                """)

            st.markdown("#### Summary")
            st.markdown("""
                - Blood glucose fluctuations highlight the complexity of daily glucose management.
                - Close alignment between insulin doses and carbohydrate intake reflects effective diabetes management.
                - Physical activity, as indicated by calorie burn and heart rate, contributes to glucose stability.
                """)

        with tab2:
            st.code(code_cells[1], language="python")
            st.code(code_cells[2], language="python")

    with st.expander("📅 Interactive Plot for Spotting Daily BG Trends"):
        # Inputs for filtering
        # 1. Dropdown for p_num
        available_patients = list(patients_d['p_num'].unique()) + ["all"]  # Get all unique patient IDs and additional 'all' option
        patient_id = st.selectbox("Select Patient ID", options=available_patients, index=0)

        # 2. Calendar widget for 'specific_day'
        min_date = patients_d['pseudo_datetime'].min().date()  # Earliest date in the dataset
        max_date = patients_d['pseudo_datetime'].max().date()  # Latest date in the dataset
        specific_day = st.date_input("Select a Day", value=min_date, min_value=min_date, max_value=max_date)

        # Filter the data
        filtered_data = FakeDateCreator.filter_patient_data(patients_d, patient_id, specific_day)

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
