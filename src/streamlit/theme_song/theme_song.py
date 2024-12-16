import streamlit as st
import os

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))


def display_page():
    st.title("Project Theme Song")

    st.markdown("""
    This page is dedicated to the project theme song. We have created a great piece of music that captures the essence of our journey.
    """)

    st.markdown("## Modelling Chronicles")

    song_file = os.path.join(current_dir, "01-modelling-chronicles.mp3")
    st.audio(song_file, format='audio/mp3')

    # =======================================
    st.markdown("## Transcript")

    st.markdown("""
    **"Modelling Chronicles” :microphone:**

    Yo, welcome to the page, it’s **Modelling time**,<br>
    Streamlit on the mic, let me drop this rhyme.<br>
    We started with a plan, made it nice and tight,<br>
    Individual pathways to model the night.

    **Tooling in the spotlight**, let me break it down,<br>
    **Transformers for imputation**, wearing the crown.<br>
    Fill those NaNs, interpolate the gaps,<br>
    Median, mean, custom flows in our traps.

    Outlier handling with a custom vibe,<br>
    Z-score, IQR, keep the data alive.<br>
    Feature engineering? We got your fix,<br>
    From *DayPhase* categories to *time column tricks*.

    We’re scaling up models, hyper-optimizing flows,<br>
    Bayesian tunes, watch how the accuracy grows.<br>
    Streamlit’s the stage, and the data’s the show,<br>
    From preprocessing moves to predictions that glow.

    Next up, normalization, keep it clean and mean,<br>
    **StandardScaler**, logs, and dummies in the scene.<br>
    Cross-validation splitters, custom to the core,<br>
    Visualize performance, yeah, we want more!

    Data cleaned up, imputed with care,<br>
    Missing values? Man, they don’t stand a prayer.<br>
    **Feature selection’s** tight, shap values in play,<br>
    Seven features poppin’, leading the way.

    Lazy Predict comes next, models on blast,<br>
    Regression’s the game, voting ensemble’s fast.<br>
    **XGB, Ridge, Lasso on the track,**<br>
    HistGradientBoosting, let’s bring it back!

    From the **Kaggle submissions** to leaderboard fights,<br>
    Optimization runs burning all night.<br>
    Model 1 stepped in, Maverick in style,<br>
    **RMSE** scores made the judges smile.

    But then came Model 2, **RapidJuxtapose**,<br>
    Time features circled, the performance arose.<br>
    Stacking up models, Ridge led the pack,<br>
    DNNs flexing, no turning back.

    We’re scaling up models, hyper-optimizing flows,<br>
    Bayesian tunes, watch how the accuracy grows.<br>
    Streamlit’s the stage, and the data’s the show,<br>
    From preprocessing moves to predictions that glow.

    **Model 2 wins**, a **2.36 score**,<br>
    Climbing up the leaderboard, aiming for more.<br>
    From data pipelines to the Kaggle fight,<br>
    Streamlit keeps it moving, day to night.

    Microphone **Modelling Chronicles**, we out the door,<br>
    Streamlit in the game, data’s the roar!
""", unsafe_allow_html=True)
