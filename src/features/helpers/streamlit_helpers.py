import streamlit as st
import nbformat

# Function to load Markdown content
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to load and display a Jupyter Notebook
def display_notebook(notebook_path):
    # Read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Loop through cells and display content
    for cell in notebook.cells:
        if cell['cell_type'] == 'markdown':
            # Render Markdown cells
            st.markdown(cell['source'])
        elif cell['cell_type'] == 'code':
            # Render Code cells
            st.code(cell['source'])
            
            # Display outputs if available
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        st.text(output['text'])
                    if 'data' in output:
                        # Render rich outputs like images or HTML
                        if 'text/html' in output['data']:
                            st.markdown(output['data']['text/html'], unsafe_allow_html=True)
                        if 'image/png' in output['data']:
                            import base64
                            image_data = base64.b64decode(output['data']['image/png'])
                            st.image(image_data)
