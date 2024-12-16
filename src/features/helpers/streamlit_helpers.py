import base64
import io
from PIL import Image
import streamlit as st
import nbformat
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# function to load model data
def load_model_data( model_name ):

    '''
    This function loads the previosly trained model and the target and feature variables used during training
    model_id can be  1 or 2 
    '''
    #import libraries
    import pandas as pd 
    import joblib
    import os

    # Get the model id
    if model_name == "Model 1 - MaverickSensor":
        model_id = 1
    elif model_name == "Model 2 - RapidJuxtapose":
        model_id = 2

    # read data
    base_dir = os.path.dirname(__file__)
    train_file_name = os.path.join(base_dir,f'../../models/model_{model_id}/data/X_train.csv')
    test_file_name = os.path.join(base_dir,f'../../models/model_{model_id}/data/y_train.csv')
    timestamps_file_name = os.path.join(base_dir,f'../../models/model_{model_id}/data/timestamps_ids.csv')

    features = pd.read_csv( train_file_name , index_col=0 )
    target = pd.read_csv( test_file_name , index_col=0 )
    time_stamps_df = pd.read_csv( timestamps_file_name , index_col=0 )

    # load the trained model
    model_pkl = os.path.join(base_dir,f'../../models/model_{model_id}/model/model.pkl')
    model = joblib.load(model_pkl)

    return model , features , target, time_stamps_df

# Function to display features and predictions for a random time stamp
def display_rand_row( features , target , model ):

    import numpy as np

    dfs = features.sample(1)
    ix = np.where( features.index.tolist() == dfs.index.tolist() )[0]
    bg_pred = model.predict( dfs )

    dfs['bg+1:00_real'] = target.loc[ dfs.index.tolist() , ]
    dfs['bg+1:00_pred'] =  bg_pred

    return dfs

# Function to create a pseudotime column
def create_pseudo_datetime(patients_df):
        
        import pandas as pd

        # Define a fake start date
        fake_start_date = '2020-01-01'

        # Convert 'time' to timedelta (duration since midnight)
        patients_df['time_delta'] = pd.to_timedelta(patients_df['time'])

        # Function to create pseudo datetime
        def transform_group(group):
            # Start from the same fake date
            group['pseudo_datetime'] = pd.to_datetime(fake_start_date) + group['time_delta']
            
            # Calculate the days elapsed when time resets (new day)
            group['days_elapsed'] = group['time_delta'].diff().apply(lambda x: 1 if x < pd.Timedelta(0) else 0).cumsum()
            
            # Add days_elapsed to the pseudo_datetime to simulate continuous time
            group['pseudo_datetime'] = group['pseudo_datetime'] + pd.to_timedelta(group['days_elapsed'], unit='D')
            
            return group

        # Apply the transformation
        transformed_df = patients_df.groupby('p_id', group_keys=False).apply(transform_group).reset_index(drop=True)
        
        return transformed_df

# Function to generate a global data frames of bg+1:00 real and predicted values
def global_predictions( output_dir ):

    import pandas as pd
    import os

    # File name for the output table
    output_name = os.path.join( output_dir , 'global_predictions.csv' )

    #iRead or create a .csv file with global predictions
    if os.path.exists( output_name ):
        common_preds_df = pd.read_csv( output_name )
        common_preds_df['pseudo_datetime'] = pd.to_datetime( common_preds_df['pseudo_datetime'] )
    else:
        # Read the data from models 1 & 2
        model_1 , features_1 , target_1 , timeStamps_1 = load_model_data( model_name = "Model 1 - MaverickSensor" )
        model_2 , features_2 , target_2 , timeStamps_2 = load_model_data( model_name = "Model 2 - RapidJuxtapose"  )

        # Parse the data from the models & generate predictions
        # NOTE: here you can only join points from the train.csv, because augmented points are not present in Model 1

        pred_df_1 = target_1.copy()
        pred_df_1['bg+1:00_pred'] = model_1.predict( features_1 )
        pred_df_1['abs_error'] = abs(pred_df_1['bg+1:00_pred']-pred_df_1['bg+1:00'])
        pred_df_1 = pred_df_1.join( timeStamps_1 )
        pred_df_1.index = pd.MultiIndex.from_arrays([pred_df_1.index, pred_df_1['time']], names=['id', 'time'])
        pred_df_1 = pred_df_1.drop( 'time' , axis = 1 )

        pred_df_2 = target_2.copy()
        pred_df_2['bg+1:00_pred'] = model_2.predict( features_2 )
        pred_df_2['abs_error'] = abs(pred_df_2['bg+1:00_pred']-pred_df_2['bg+1:00'])
        pred_df_2 = pred_df_2.join( timeStamps_2 )
        pred_df_2.index = pd.MultiIndex.from_arrays([pred_df_2.index, pred_df_2['time']], names=['id', 'time'])
        pred_df_2 = pred_df_2.drop( 'time' , axis = 1 )

        # Join predictions
        common_preds_df = pred_df_1.join( pred_df_2 , lsuffix='_m1', rsuffix='_m2' )

        # If needed, add a column with patient ids
        p_ids = common_preds_df.index.get_level_values('id').tolist()
        p_ids = [id.split('_')[0] for id in p_ids]
        common_preds_df['p_id'] = p_ids

        # bring the time back from the index
        common_preds_df = common_preds_df.reset_index(level=1)

        # subset cols 
        common_preds_df = common_preds_df[['p_id','time','bg+1:00_m1','bg+1:00_pred_m1','bg+1:00_pred_m2']]

        #Create PseudoDaytime
        common_preds_df = create_pseudo_datetime( common_preds_df )

        #Save the table
        common_preds_df.to_csv( output_name , index=False )

    return common_preds_df

# Function to plot a random daily profile for a patient
def rand_daily_profile( p_id , common_preds_df ):

    import numpy as np
    import math 
    import matplotlib.pyplot as plt

    dfs = common_preds_df[ common_preds_df['p_id'] == p_id ]
    unique_days = dfs["pseudo_datetime"].dt.date.unique()
    random_day = np.random.choice(unique_days)
    filtered_df = dfs[dfs["pseudo_datetime"].dt.date == random_day]

    max_val = dfs[['bg+1:00_m1','bg+1:00_pred_m1','bg+1:00_pred_m2']].max().max()
    min_val = dfs[['bg+1:00_m1','bg+1:00_pred_m1','bg+1:00_pred_m2']].min().min()

    max_val = math.ceil( max_val )
    min_val = math.floor( min_val )

    fig,ax = plt.subplots( figsize=(12,4) )
    plt.plot( filtered_df["pseudo_datetime"] , filtered_df['bg+1:00_m1'] , 'dodgerblue' , label = 'bg+1:00 real vals')
    plt.plot( filtered_df["pseudo_datetime"] , filtered_df['bg+1:00_pred_m1'] , 'palevioletred' , label = 'Model 1 bg+1:00 pred vals')
    plt.plot( filtered_df["pseudo_datetime"] , filtered_df['bg+1:00_pred_m2'] , 'yellowgreen' , label = 'Model 2 bg+1:00 pred vals')
    plt.ylim(min_val,max_val)
    plt.legend( loc = 'best' )
    
    # Format time stamps
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Use streamlit to show the plot
    return fig,ax
