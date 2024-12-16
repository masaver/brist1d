import streamlit as st
import os

from src.features.helpers.streamlit_helpers import extract_notebook_images

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
report_path = os.path.join(project_root, 'reports', 'final-report')


def display_page():
    st.title("Modelling")

    st.markdown("""
    As team we decided to investigate individually different approaches to the problem. To handle data consistently we developed common helpers and tools for data preprocessing, feature engineering, and modelling.
    """)

    st.markdown("## Tools")

    expand = st.expander("### ⚙️ **Custom Transformers and Pipelines**")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Preprocessing**:
            - DayPhaseTransformer
            - DateTimeHourTransformer
            - DropColumnsTransformer
            - FillPropertyNaNsTransformer
            - PropertyOutlierTransformer
            """)

        with col2:
            st.markdown("""
            **Normalization**:
            - MinMaxScalerTransformer
            - StandardScalerTransformer
            - GetDummiesTransformer
            - Log1PTransformer
            """)

        st.code("""
        import ... # import custom transformers

        preprocessing_pipeline = Pipeline(steps=[
            ('date_time', DateTimeHourTransformer(time_column='time', result_column='hour', type='sin_cos', drop_time_column=True)),
            ('drop_parameter_cols', DropColumnsTransformer(starts_with=['activity', 'carbs'])),
            ('drop_others', DropColumnsTransformer(columns_to_delete=['time'])),
            ('fill_properties_nan_bg', FillPropertyNaNsTransformer(parameter='bg', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
            ('fill_properties_nan_insulin', FillPropertyNaNsTransformer(parameter='insulin', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=4)),
            ('fill_properties_nan_cals', FillPropertyNaNsTransformer(parameter='cals', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
            ('fill_properties_nan_hr', FillPropertyNaNsTransformer(parameter='hr', how=['interpolate', 'median'], interpolate=3, ffill=1, bfill=1, precision=1)),
            ('fill_properties_nan_steps', FillPropertyNaNsTransformer(parameter='steps', how=['zero'], interpolate=3, ffill=1, bfill=1, precision=1)),
            ('drop_outliers', PropertyOutlierTransformer(parameter='insulin', filter_function=lambda x: x < 0, fill_strategy='zero')),
            ('extract_features', ExtractColumnsTransformer(columns_to_extract=columns_to_extract)),
        ])

        standardization_pipeline = Pipeline(steps=[
            ('get_dummies', GetDummiesTransformer(columns=['hour', 'p_num'])),
            ('standard_scaler', StandardScalerTransformer(columns=columns_to_extract[3:-1]))
        ])

        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),
            ('standardization', standardization_pipeline)
        ])
        """)

    expand = st.expander("### 📈 **Hyper Parameter Optimizations**")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Bayesian Optimization**:
            - XGBHyperparameterTuner
            - LGBHyperparameterTuner
            - HistGradientBoostingHyperparameterTuner
            - KNNHyperparameterTuner
            """)

        with col2:
            st.markdown("""
            **Search Spaces**:
            - wide
            - medium
            - narrow
            """)

        st.code("""
        xgb_tuner = XGBHyperparameterTuner(search_space='wide')
        xgb_tuner.fit(X=X, y=y)
        xgb_model = xgb_tuner.best_model
        """)

    expand = st.expander("### 🔀 **Custom Splitter**")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - Split data based on test distribution and size
            - Avoid bias in cross-validation
            """)

        st.code("""
        splitter = CustomSplitter(test_size=0.2, n_splits=n_splits)
        splitter.fit(X_train, groups=[0] * len(X_train) + [1] * len(X_augmented))

        search_cv = BayesSearchCV(cv=splitter, ...)
        search_cv.fit(X=pd.concat([X_train, X_augmented]), ...)
        """)

    st.markdown("## Modelling Steps")

    st.markdown("""
    From the four models we developed, the two best models were selected for this presentation.
    """)

    # Header =======================================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### MaverickSensor (Model 1)")
    with col2:
        st.markdown("#### RapidJuxtapose (Model 2)")

    # Data Augmentation =======================================
    col1, col2 = st.columns(2)
    with col2:
        expand = st.expander(" ⏫ Data Augmentation")
        with expand:
            st.markdown("""
            - Use test data for unbiased model training
            - Create 175.000 new data points for all patients
            """)

    # Data Cleaning and Feature Engineering =======================================
    expand = st.expander(" ❌ Data Cleaning and Feature Engineering")
    with expand:
        st.markdown("""
        - **Data Cleaning**: Remove parameters with high missing values (Carbs, Activity).
        - **Data Imputation**: Fill missing values using interpolation, mean, median, zero.
        - **Outlier Detection**: Remove outliers from the data.
        - **Feature Engineering**: Extract day phase/circular encoding from time column.
        """)

    # Data Preliminary Modelling =======================================
    expand = st.expander(" 📝 Preliminary Modelling (Lazy Predict)")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Model Selection**
            | Model       | RMSE | Time Taken |
            | ----------- | ---- | ---------- |
            | XGBRegressor | 1.82 | 5.80s |
            | HGBRegressor | 1.90 | 9.92s |
            | LGBMRegressor | 1.90 | 7.96s |
            | Lasso | 2.06 | 2.93s |
            | Ridge | 2.07 | 1.25s |
            | KNNRegressor | 2.10 | 10.85s |

            """)
        with col2:
            st.markdown("""
            **Model Selection**
            | Model       | RMSE | Time Taken |
            | ----------- | ---- | ---------- |
            | HGBRegressor | 1.82 | 5.80s |
            | LassoLarsIC | 1.90 | 9.92s |
            | KNNRegressor | 1.90 | 7.96s |
            | XGBRegressor | 2.06 | 2.93s |
            """)

    # Feature Selection =======================================
    col1, col2 = st.columns(2)
    with col1:
        expand = st.expander(" ☑️ Feature Selection (by SHAP)")
        with expand:
            img_outputs_02 = extract_notebook_images(os.path.join(report_path, '04-modelling', '01-model-1', '02-feature-importance-with-SHAP.ipynb'))
            st.image(img_outputs_02[-3])
            st.markdown("""
            - `bg-0:00`, `bg-0:10`, `bg-0:15`
            - `hr-0:00`
            - `insulin-0:00`
            - `day_phase_evening`, `day_phase_night`
            """)
    with col2:
        expand = st.expander(" ☑️ Feature Selection (by importance)")
        with expand:
            img_outputs_02 = extract_notebook_images(os.path.join(report_path, '04-modelling', '02-model-2', '04-feature-selection.ipynb'))
            st.image(img_outputs_02[0])
            st.markdown("""
            - `bg-0:00` - `bg-1:00`
            - `cals-0:00` - `cals-1:00`
            - `hr-0:00` - `hr-1:00`
            - `insulin-0:00` - `insulin-1:00`
            - `steps-0:00` - `steps-1:00`
            - `patient number`
            - `time_sin`, `time_cos`
            """)

    # Modelling =======================================
    expand = st.expander(" 🔮 Modelling")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **VotingRegressor**
            - Lasso
            - Ridge
            - HGBRegressor
            - LGBMRegressor
            - XGBRegressor
            """)
        with col2:
            st.markdown("""
            **StackingRegressor (Ridge)**
            - HistGradientBoostingRegressor
            - LassoLarsIC
            - KNNRegressor
            - XGBRegressor
            - KerasRegressor
            """)

    # Evaluation =======================================
    expand = st.expander(" ✅ Evaluation (Prediction and Submission)")
    with expand:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Kaggle Score**: 2.56
            """)
        with col2:
            st.markdown("""
            **Kaggle Score**: 2.36 (4th on PLB)
            """)
            st.image(os.path.join(current_dir, 'images', 'private-leaderboard.png'))
