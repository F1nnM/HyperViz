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


shap.plots._labels.labels['VALUE'] = 'Influence on the optimized column'

st.set_page_config(page_title='HyperViz', layout='wide')

@st.cache(allow_output_mutation=True, ttl=3600)
def get_cached_experiment(df):
    exp = hp.Experiment.from_dataframe(df)
    exp._compress = True
    return exp.to_streamlit(key='hiplot')

def calc_shap_summary_plot(df, column_to_optimize, param_columns, cmap):

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

    shap.summary_plot(shap_values, X, cmap=cmap)

    return fig, ordered_encoder

def create_legend(encoder, cmap):

    legend = {}
    for col, row in encoder.items():
        for val, encoded in row.items():
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            legend.setdefault(encoded, {})[col] = val

    def color_table(row):
        val_count = row.size -row.isna().sum()
        background_colors = cmap([i/(val_count-1) for i in range(val_count)])

        def get_foreground_color(color):
            if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 0.5:
                return '#000000' 
            else:
                return '#ffffff'

        css = [f'background-color: {matplotlib.colors.rgb2hex(background_colors[i])}; color: {get_foreground_color(background_colors[i])}' if val == val else 'color: #ffffff' for i, val in enumerate(row)]
        return pd.Series(css)

    return pd.DataFrame.from_dict(legend).style.apply(color_table, axis=1)


def shap_viz(df):
    column_to_optimize = st.selectbox('Please select the column to optimze:', df.columns)

    param_columns = st.multiselect('Which columns are the parameters to be analyzed?', list(df.columns), [col for col in df.columns if col.startswith('param_')])

    draw = st.checkbox('Auto-refresh the plot?')
    if not draw:
        draw = st.button('Draw plot')
    if draw:
        cmap = matplotlib.cm.get_cmap('nipy_spectral')
        shap.plots._labels.labels['VALUE'] = '[SHAP summary plot] Influence on the select column '
        fig, encoder = calc_shap_summary_plot(df, column_to_optimize, param_columns, cmap)
        st.write(fig)
        legend = create_legend(encoder, cmap)
        st.write(legend)


def sidebar():
    st.sidebar.title('Select and filter you\'re data here:')
    models = dm.get_models()
    project = st.sidebar.selectbox('Select your project', list(models.keys()))
    if project:
        models = st.sidebar.multiselect('Select the models to compare:', models[project])

    if not models:
        return 

    data = dm.read_files(project, models)

    # Filters

    return data

def center_uploader():
    st.write('## Upload your files here:')
    models = dm.get_models()
    project = st.selectbox('Please select a project for the files to be saved under:', ['New project']+list(models.keys()))
    if project == 'New project':
        project = st.text_input('Please enter a name for the project:')
        model = st.text_input('Please name the model this data belongs to:')
    else:
        model = st.selectbox('Please select the model this data belongs to:', ['New model']+models[project])
        if model == 'New model':
            model = st.text_input('Please name the model this data belongs to:')

    files = st.file_uploader('Upload your .csv file(s) here. You can upload multiple files, of e.g. different training runs', accept_multiple_files=True)

    delimiter = [';', ','][st.radio('Select the delimiter of the csv file(s):', [0, 1], format_func=lambda x: ['; - Semicolon - GridsearchCV default', ', - Comma - normal CSV'][x])]
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

    columns_to_keep = st.multiselect('Select which columns you want to keep:', list(df.columns), columns_suggestion)

    if st.button('Upload'):
        with st.spinner("Processing..."):
            dm.process_data(df, project, model, columns_to_keep)
        st.write('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

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
    st.title('HyperViz')
    st.write('Blablabla visualize stuff bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla bla')
    data = sidebar()
    if data is not None:
        center_viz(data)
    else:
        center_uploader()

if __name__ == '__main__':
    main()
