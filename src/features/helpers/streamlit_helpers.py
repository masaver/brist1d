import base64
import io
from PIL import Image
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
                            image_data = base64.b64decode(output['data']['image/png'])
                            st.image(image_data)


# Function to extract images from the notebook
def extract_notebook_images(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    images = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", {}):
                    # Decode the base64 image data
                    image_data = base64.b64decode(output["data"]["image/png"])
                    # Open the image using PIL and append it to the list
                    image = Image.open(io.BytesIO(image_data))
                    images.append(image)

    return images


def extract_html_outputs(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    html_outputs = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if "text/html" in output.get("data", {}):
                    html_outputs.append(output["data"]["text/html"])

    return html_outputs


def extract_text_outputs(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    text_outputs = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if "text/plain" in output.get("data", {}):
                    text_outputs.append(output["data"]["text/plain"])

    return text_outputs


# Function to extract images from the notebook
def extract_code_cells(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    code_cells = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells.append(cell.source)
    return code_cells
