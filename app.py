from collections import defaultdict
import os

import hiplot as hp
import pandas as pd
import streamlit as st

import data_management as dm

import shap
import xgboost

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

import re

shap.plots._labels.labels['VALUE'] = 'Influence on the optimized column'

st.set_page_config(page_title='HyperViz', layout='wide')

# custom css to make labels more readable
st.write('''<style>
label {
    font-size: 1rem !important;
}
input[type="radio"] + div {
    font-size: 0.9rem !important;
}
</style>''', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True, ttl=3600)
def get_cached_experiment(df: pd.DataFrame):
    exp = hp.Experiment.from_dataframe(df)
    exp._compress = True
    exp.colorby = df.columns[0]
    return exp.to_streamlit(key='hiplot')

@st.cache(allow_output_mutation=True, ttl=3600)
def calc_shap_summary_plot(df: pd.DataFrame, column_to_optimize, param_columns, cmap):

    df = df[[column_to_optimize]+param_columns]
    encoder = {col: {(val if val == val else 'nan'): encoded for encoded, val in enumerate(pd.unique(df[col]))} for col in param_columns}

    for col in param_columns:
        df[col] = df[col].map(lambda x: (encoder[col][x] if x == x else encoder[col]['nan']))

    X, y = df.loc[:,df.columns != column_to_optimize], df[column_to_optimize]
    model = xgboost.train({"learning_rate": 0.1}, xgboost.DMatrix(X, label=y), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    order = (-np.mean(np.abs(shap_values), axis=0)).argsort()
    ordered_encoder = {}
    for i in order:
        col = param_columns[i]
        ordered_encoder[col] = encoder[col]

    fig = plt.figure()

    shap.summary_plot(shap_values, X, cmap=cmap, show=False)

    return fig, ordered_encoder

@st.cache(allow_output_mutation=True, ttl=3600)
def create_legend(encoder, cmap):

    legend = {}
    for col, row in encoder.items():
        for val, encoded in row.items():
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            legend.setdefault(encoded, {})[col] = val

    def color_table(row):
        val_count = row.size -row.isna().sum()
        if val_count == 1:
            fractions = [0]
        else:
            fractions = [i/(val_count-1) for i in range(val_count)]
        background_colors = cmap(fractions)
        def get_foreground_color(color):
            if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 0.5:
                return '#000000' 
            else:
                return '#ffffff'

        css = [f'background-color: {matplotlib.colors.rgb2hex(background_colors[i])}; color: {get_foreground_color(background_colors[i])}' if val == val else 'color: #ffffff' for i, val in enumerate(row)]
        return pd.Series(css)

    return pd.DataFrame.from_dict(legend).style.apply(color_table, axis=1)

def shap_viz(df: pd.DataFrame):
    with st.expander('How does this work?'):
        st.write('Your data is used to train a simple __xgboost-model__, with the "column to optimize" as label. This model is trying to __predict the performance__ of your model, based on the parameters. After the training is complete, the __SHAP values__ are calculated, which are used to explain the behaviour of this xgboost model. So, as a result, the SHAP values can be used to __explain the influence of a parameter__ on the performance.')
    
    st.write('#')
    column_to_optimize = st.selectbox('Please select the column to optimze:', df.columns)
    other_colums = [col for col in df.columns if col != column_to_optimize]

    param_columns = st.multiselect('Which columns are the parameters to be analyzed?', other_colums, [col for col in other_colums if col.startswith('param_')])

    draw = st.checkbox('Auto-refresh the plot?')
    if not draw:
        draw = st.button('Draw plot')
    if draw:
        with st.expander('What can I see here?'):
            st.write("For each column you chose above you can see a lot of dots; __one dot for each row__ in your dataset. The __color shows the value__ the dot represents and it's __left-right-position shows the impact__ it has on the selected \"column to optimize\". Underneath is legend to explain, which color stands for which value.")
            st.write("__Example:__ If a parameter only contains two options, you might see a distinct cluster of gray dots and a distinct cluster of black dots. If the gray cluster is further left than the black one, that means that all runs with the \"gray\" value performed worse than the others. Which value that is, can be read from the legend below the plot.")
        cmap = matplotlib.cm.get_cmap('nipy_spectral')
        shap.plots._labels.labels['VALUE'] = '[SHAP summary plot] Influence on the selected column'
        fig, encoder = calc_shap_summary_plot(df, column_to_optimize, param_columns, cmap)
        st.write(fig)
        legend = create_legend(encoder, cmap)
        st.write(legend)

def sidebar():
    st.sidebar.write('# Select and filter you\'re data here:')
    models = dm.get_models()
    project = st.sidebar.selectbox('Select your project', list(models.keys()))
    if project:
        models = st.sidebar.multiselect('Select the models to compare:', models[project])

    if not models:
        return 

    data = dm.read_files(project, models)

    # Filters

    st.sidebar.write('### Here you can apply some basic filters before the data is passed to the vizes on the right.')
    st.sidebar.write('#### This can help if your data is to big to be displayed by the Hiplot component.')
    st.sidebar.write('####')

    if st.sidebar.checkbox('Remove columns with only one unique value?'):
        for col in data.columns:
            if len(data[col].unique()) == 1:
                data.drop(col,inplace=True,axis=1)

    col_list = list(data.columns)
    columns_to_keep = st.sidebar.multiselect('Which columns should be displayed?', col_list, col_list)
    data = data[columns_to_keep]

    return data

def center_uploader():
    st.write('## Upload your files here:')
    models = dm.get_models()
    project = st.selectbox('Please select a project:', ['New project']+list(models.keys()))
    project_valid = True
    if project == 'New project':
        project = st.text_input('Please enter a name for the project:')
        project_valid = not re.search(r'[^A-Za-z0-9_\-]',project)
        if not project_valid:
            st.warning('Project names may only contain alphanumeric characters, _ and -')
        model = st.text_input('Please name the model:')
    else:
        model = st.selectbox('Please select the model:', ['New model']+models[project])
        if model == 'New model':
            model = st.text_input('Please name the model:')
    
    model_valid = not re.search(r'[^A-Za-z0-9_\-]',model)
    if not model_valid:
        st.warning('Model names may only contain alphanumeric characters, _ and -')

    files = st.file_uploader('Upload your .csv file(s) here. You can upload multiple files, of e.g. different GridSearch runs', accept_multiple_files=True)

    tag = st.text_input('Provide a tag for the uploaded data. This might be a date, a number or any other string:','default')

    delimiter = [';', ','][st.radio('Select the delimiter of the csv file(s):', [0, 1], format_func=lambda x: ['; - Semicolon - GridSearchCV default', ', - Comma - normal CSV'][x])]
    decimal = ['.', ','][st.radio('Select the decimal point of the csv file(s):', [0, 1], format_func=lambda x: ['. - Dot - English', ', - Comma - German'][x])]

    df = dm.read_files_to_df(files, delimiter, decimal)

    mean_cols = []
    std_cols = []
    param_cols = []

    for col in df.columns:
        if col.startswith('mean_test_'):
            mean_cols.append(col)
        elif col.startswith('std_test_'):
            std_cols.append(col)
        elif col.startswith('param_'):
            param_cols.append(col)
            
    columns_suggestion = sorted(mean_cols)+sorted(std_cols)+sorted(param_cols)
    if len(columns_suggestion) == 0:
        columns_suggestion = list(df.columns)

    columns_to_keep = st.multiselect('Select which columns you want to keep:', list(df.columns), columns_suggestion)

    if st.button('Upload'):
        if not model_valid or not project_valid:
            st.warning('Please fix above warnings!')
        elif len(model) < 1 or len(project) < 1:
            st.warning('Both, project name and model name must not be empty!')
        elif len(files) < 1:
            st.warning("You haven't uploaded any file!")
        elif len(tag) < 1:
            st.warning('The tag may not be empty!')
        else:
            with st.spinner('Processing...'):
                dm.process_data(df, project, model, tag, columns_to_keep)
            st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

def center_delete():
    st.write('## Delete data')
    st.warning(f'This action is __irreversible__! Proceed with caution.')
    models = dm.get_models()
    project = st.selectbox('Select the project:', list(models.keys()))
    if st.button('Delete entire project'):
        if not project:
            st.warning('No project selected')
        else:
            dm.delete_project(project)
            st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
    if project:
        models_to_delete = st.multiselect('Select the model(s) to delete:', models[project] if project in models else [])
        if st.button('Delete selected models'):
            if len(models_to_delete) == 0:
                st.warning('No models selected')
            else:
                for model in models_to_delete:
                    dm.delete_model(project, model)
                    st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
    
def center_data_management():
    st.write('The goal of this app is to help with the __easy visualization of hyperparameters__ of machine learning models and their __influence on the performance__ of the model. It was originally developed to display results of the sklearn GridSearchCV method, however as long your data can be represented in a __.csv format__, it can be analyzed here. __Simply upload your file and start anaylzing.__')
    st.write('A __project__ can contain multiple __models__, which can be compared against each other.')
    st.write('A __model__ contains all the data for one model. Should you upload __multiple files__ at once, they will be __concatenated__. Should you want to __compare different files__ for one model, you can upload them one after another, providing __tags__ for different files. The data will be appended and the tags will be visible in the analysis once multiple tags have been created.')
    
    center_uploader()
    st.write('#')
    center_delete()

def center_viz(df):
    ''''''
    '''
    ## Parallel plot
    ### This plot allows you to visualize higher-dimensional data and analyze it at a glance.
    '''
    get_cached_experiment(df).display()
    '''
    #
    ## SHAP
    ### This plot helps you discover more details about the influence of individual parameters on the performance of your model.
    '''
    shap_viz(df)

def main():
    st.title('HyperViz - Analysing Hyperparameters Made Easy')
    
    data = sidebar()
    if data is not None:
        center_viz(data)
    else:
        center_data_management()

if __name__ == '__main__':
    main()
