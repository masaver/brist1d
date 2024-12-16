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
# from src.features.helpers.streamlit_helpers import load_markdown, display_notebook
from src.features.helpers.streamlit_helpers import extract_notebook_images, extract_code_cells

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
file_path = os.path.join(project_root, "data", "raw", "train.csv")


# load the dataset
@st.cache_data
def display_page():
    # Anchor at the top
    st.markdown("<a name='top'></a>", unsafe_allow_html=True)

    st.markdown("# Conclusions")

    st.markdown(
        """In this app, we present the results of our approach for predicting blood glucose levels one hour ahead using six hours of participant data. Our work involved a comprehensive workflow, including data exploration, preprocessing, feature engineering as well as model training. Furthermore, a custom cross-validation strategy was implemented to evaluate model performance and optimize its hyperparameters.
Two Modelling Approaches

We developed two different modelling approaches. The first approach utilized the training data as provided, while the second one augmented the training dataset by generating additional data from the test data, shifting it by one to four hours. This data augmentation strategy significantly improved the model’s performance by providing additional training examples.

Our continuous trials with different types of models achieved a Kaggle Score of 2.45 on the public leaderboard, which was a significant improvement over the baseline model. So we kept further validations using a custom cross-validation strategy, which showed consistent results across different folds. **Of note, our final solution was ranked in the top 1% of the Kaggle public Leaderboard**."""
    )

    st.markdown("# Future Directions")

    st.markdown(
        """This project successfully demonstrated the utility of data augmentation and ensemble modelling. To enhance the model’s performance and generalization, the following areas could be explored deeper in the future:
* **Feature Reduction**: We could explore further feature reduction techniques to identify the most important features and reduce the dimensionality of the dataset.
* **Noise Reduction**: We could explore techniques to reduce noise in the data and improve the quality of the predictions.
* **Cross-Validation**: We could explore different cross-validation techniques which reflect better the Kaggle score.
* **Deep Learning Approaches**: Although our attempt to integrate a basic Deep Neural Network (DNN) into the Stacking Ensemble model did not yield significant improvements, more sophisticated deep learning architectures and strategies could be explored in the future to unlock additional potential.

By addressing these areas, the model could achieve even greater accuracy and reliability, providing more value to users and competitive performance in future challenges."""
    )
