import streamlit as st
import os

from src.features.helpers.streamlit_helpers import load_markdown
from importlib import import_module

# define paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# setup the menu structure
st.sidebar.title("BrisT1D Blood Glucose Prediction")
pages=["Introduction", "Data Exploration", "Data Visualization", "Modelling", "Predictions", "Conclusion & Perspectives"]
page=st.sidebar.radio("Menu", pages)

# Define a mapping for pages and their corresponding module files
page_modules = {
    #"Kaggle Challenge": "???",  
    "Exploratory Data Analysis": "eda", 
    #"Modelling": "???", 
    #"Predictions": "???", 
    #"Conclusion & Perspectives": "???"
}

st.sidebar.info(load_markdown(os.path.join(current_dir, "markdown", "team.md")))

# Cache the dynamic module loader
@st.cache_resource
def load_module(module_name):
    return import_module(module_name)

# Dynamically load and display the selected page
if selected_page in page_modules:
    # Import the module dynamically
    module_name = page_modules[selected_page]
    #module = load_module(module_name)
    module = import_module(module_name)

    # Call the display_page function from the module
    if hasattr(module, "display_page"):
        module.display_page()
else:
    st.markdown(f"### {selected_page}")
    st.markdown("Content for this page is under construction.")


# # Contect and Scope
# if selected_page == pages[0] : 
  
#   st.markdown(load_markdown(os.path.join(current_dir, "markdown", "01-kaggle-competition.md")))
  
# # Exploratory Data Analysis
# if selected_page == pages[1] : 
#   module = import_module(page_modules[selected_page])

#   # Call the display_page function from the module
#   if hasattr(module, "display_page"):
#         module.display_page()
 
# if selected_page == pages[2] : 
#   st.write("# Modelling")


# if selected_page == pages[3] : 
#   st.write("# Predictions")

# if selected_page == pages[4] : 
#   st.write("# Conclusions & Perspectives")
  
