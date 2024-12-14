import streamlit as st
import os
from src.features.helpers.streamlit_helpers import load_markdown

current_dir = os.path.dirname(os.path.abspath(__file__))
st.markdown(load_markdown(os.path.join(current_dir, "markdown", "intro.md")))