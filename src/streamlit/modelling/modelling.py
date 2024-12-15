import streamlit as st
import os

from src.features.helpers.streamlit_helpers import extract_notebook_images, extract_code_cells, extract_html_outputs, extract_text_outputs

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
report_path = os.path.join(project_root, 'reports', 'final-report')


def display_page():
    # st.title("Modelling")

    ### as group we decided to go individual modelling pathways
    ### so everybody could investigate different models and techniques
    ### so we developed a common toolset to be used by everyone in different stages of modelling

    st.markdown("## Tooling")
    st.markdown("### Data Imputation Transformers")

    st.markdown("""
    - **FillPropertyNaNsTransformer**
        - Fills missing values for a specified parameter (52 columns)
        - Interpolation, forward fill, backward fill (with or without limit)
        - Mean, median, zero, custom value
        - Chainable operations passable as a list

    - **PropertyOutlierTransformer**
        - Handles outliers for a parameter specified by the user (52 columns)
        - Outlier detection used by filter function (e.g. z-score, IQR, custom)
        - Outlier handling used by replace function (e.g. mean, median, custom)
    """)

    st.markdown("### Feature Engineering Transformers")
    st.markdown("""
    - **DayPhaseTransformer**
        - Extracts the day phase from a given time column into 6 categories
        - Categories: Morning, Noon, Afternoon, Evening, Late Evening, Night

    - **DateTimeHourTransformer**
        - Extracts the hour from a given time column into bins or cyclic encoding
        - Number of bins can be specified
        - Cyclic encoding into a sine and cosine component can be enabled

    - **DropColumnsTransformer**
        - Drops columns based on name patterns or specific columns
    """)

    st.markdown("### Normalization Transformers")

    st.markdown("""
        - **StandardScalerTransformer**: Scales numerical columns using StandardScaler.
        - **GetDummiesTransformer**: Converts categorical columns into dummy variables.
        - **LogTransformer**: Applies log transformation to specified numerical columns.
    """)

    st.markdown("### Hyper Parameter Optimizing")
    st.markdown("""
    - for a bunch of models we used hyperparameter optimization to find the best parameters for our models
    - uses Bayesian Optimization to find the best parameters
    - each provides 3 levels of parametersets (wide, medium, narrow) to be used for optimization speed
    - uses the scikit optimize library
    """)

    st.markdown("### Other Tools")
    st.markdown("""

    **CustomSplitters**
    - we developed series of custom splitters to to egt better cross validation results with biased data

    **CustomVisualizers**
    - we developed a series of visualizers to help understand cross validation results and model performance

    **KaggleSubmission**
    - a helper to create kaggle submissions from our models
    """)

    # =======================================
    st.markdown("## Modelling Steps")
    st.markdown("""
        **Data Preprocessing**:
        - **Data Cleaning**: Remove parameters with high missing values (Carbs, Activity).
        - **Data Imputation**: Fill missing values using interpolation, mean, median, zero.
        - **Feature Engineering**: Extract day phase from time column.
        - **Normalization**: Scale numerical features and convert categorical features to dummy variables.
        **Preliminary Modelling**:
        - **Lazy Predict**: Evaluate multiple regression models for best performance.
        **Feature Selection**:
        - **Feature Importance**: Identify key features influencing predictions.
        **Modelling**:
        - **Model Selection**: Select the best performing models for hyperparameter tuning.
        - **Hyperparameter Tuning**: Optimize models parameters for accuracy.
        - **Modelling**: Ensemble/Stack models for improved performance.
        **Evaluation**:
        - **Prediction**: Forecast blood glucose levels one hour ahead.
        - **Submission**: Create Kaggle submission for model evaluation.
        """)

    # =======================================
    st.markdown("### Model 1 -- **MaverickSensor** --")

    text_outputs_01 = extract_text_outputs(os.path.join(report_path, '04-modelling', '01-model-1', '01-prelim-modelling_with-LP.ipynb'))

    st.markdown("""
        **Feature Engineering**:
        - **Day Phase**: Extract day phase from time column for time-dependent predictions
        - **Lag Features**: Take values from the last hour for preliminary modelling
        - **Feature Selection**: The 7 most relevant features determined by shap values (bg-0:00, bg-0:15, bg-0:10, hr-0:00, insulin-0:00, day_phase_evening, day_phase_night)
        - **Model Selection**: VotingRegressor (Lasso, Ridge, HistGradientBoostingRegressor, XGBRegressor, LGBMRegressor)
        - **Hyperparameter Optimization**: Tune model parameters for best performance
        """)

    st.markdown("#### Preliminary Modelling with Lazy Predict")
    st.markdown("The following table shows the results of the preliminary modelling using Lazy Predict:")
    st.text(text_outputs_01[-1])

    # =======================================
    img_outputs_02 = extract_notebook_images(os.path.join(report_path, '04-modelling', '01-model-1', '02-feature-importance-with-SHAP.ipynb'))
    st.markdown("#### Feature Importance")
    st.markdown("The following plots are showing the the most important features for the model:")
    st.markdown("**XGBoost Feature Importance**")
    st.image(img_outputs_02[-3])
    st.markdown("**LGBM Feature Importance**")
    st.image(img_outputs_02[-2])
    st.markdown("**HistGradientBoosting Feature Importance**")
    st.image(img_outputs_02[-1])

    # =======================================

    image_outputs_04 = extract_notebook_images(os.path.join(report_path, '04-modelling', '01-model-1', '04-model-selection-and-HyperParameterTuning.ipynb'))
    text_outputs_04 = extract_text_outputs(os.path.join(report_path, '04-modelling', '01-model-1', '04-model-selection-and-HyperParameterTuning.ipynb'))
    st.markdown("#### Model Selection and Hyperparameter Optimization")

    st.markdown("""
    **Model Selection**:
    - **XGBRegressor**: Extreme Gradient Boosting model
    - **LGBMRegressor**: Light Gradient Boosting model
    - **HistGradientBoostingRegressor**: Histogram-based Gradient Boosting model
    - **Lasso**: Lasso Linear Regression model
    - **Ridge**: Ridge Linear Regression model
    - **KNNRegressor**: K-Nearest Neighbors model
    """)

    st.markdown("**XGBRegressor**")
    st.code("""
    xgb_tuner = XGBHyperparameterTuner(search_space='wide')
    xgb_tuner.fit(X=X, y=y)
    xgb_model = xgb_tuner.best_model
    """)
    st.image(image_outputs_04[0])

    st.markdown("**KNNRegressor**")
    st.code("""
    knn_tuner = KNNHyperparameterTuner(search_space='wide')
    knn_tuner.fit(X=X, y=y)
    knn_model = knn_tuner.best_model
    """)
    st.image(image_outputs_04[5])


    st.markdown("**Model Evaluation**")
    st.text(text_outputs_04[-2])

    st.markdown("#### Voting Regressor Model")
    st.image(os.path.join(current_dir, 'images', 'model-architecture.png'))

    st.markdown("""
    ## Kaggle Score Results

    For each of the tested models, predictions were generated and submitted to the competition on Kaggle. Below are the **expected RMSE scores** (from cross-validation) and the **Kaggle leaderboard scores**:

    | Model       | Expected RMSE | Kaggle Score |
    | ----------- | ------------- | ------------ |
    | Lasso      | 2.08       | 2.62 |
    | Ridge   | 2.08        | 2.62 |
    | XGBoost   | 2.11        | 2.59 |
    | VotingRegressor   | 2.05        | 2.56 |
    """)

    # =======================================
    st.markdown("### Model 2 -- **RapidJuxtapose** --")
    st.markdown("""
        **Data Augmentation**:
        - Extract data from test data time series for model training
        - Data from each patient now available for unbiased model training
        - We could extract 175000 lines of data from the test data
        **Feature Engineering**:
        - **Time**: Circular encoding of time feature into sine and cosine components.
        - **Lag Features**: Take values from the last two hours for preliminary modelling
        - **Patient number**: One-hot encoding of patient number for model training
        **Feature Selection**: Choose the most relevant features for model training.
            - 'bg' until -1hs
            - 'cals' until -1hs
            - 'hr' until -1hs
            - 'insulin' until -1hs
            - 'steps' until -1hs
        **Model Selection**: StackingRegressor (Ridge as final estimator)
            - **HistGradientBoostingRegressor**: Gradient boosting model
            - **LassoLarsIC**: Linear regression with L1 regularization
            - **KNNRegressor**: K-nearest neighbors model
            - **XGBRegressor**: Extreme gradient boosting model
            - **DNNRegressor**: Deep neural network model (keras)
        **Hyperparameter Optimization**: Tune parameters for each model for best performance
        **Model Evaluation**: Kaggle Score: 2.36 (4. in public leaderboard)
        """)

    # =======================================
    st.markdown("### Model Comparison")
    st.markdown("""
        **Model 1**:
        - **Kaggle Score**: 2.48
        **Model 2**:
        - **Kaggle Score**: 2.36
        **Model 2** outperforms Model 1 by 0.12 in Kaggle score.
        """)
