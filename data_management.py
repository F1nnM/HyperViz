import os
import pandas as pd

from pathlib import Path

import shutil

def process_data(df, project, model, tag, columns_to_keep):
    # dataprep

    mean_cols = []
    std_cols = []
    param_cols = []
    others = []

    for col in columns_to_keep:
        if col.startswith('mean_test_'):
            mean_cols.append(col)
        elif col.startswith('std_test_'):
            std_cols.append(col)
        elif col.startswith('param_'):
            param_cols.append(col)
        else:
            others.append(col)
            
    columns_ordered = sorted(mean_cols)+sorted(std_cols)+sorted(param_cols)+sorted(others)

    df = df[columns_ordered]



    df['tag'] = tag

    path = Path(f'./data/{project}')
    path.mkdir(parents=True, exist_ok=True)

    previous_data = None

    try:
        previous_data = pd.read_csv(f'./data/{project}/{model}.csv')
    except:
        pass

    if not previous_data is None:
        df = previous_data.append(df, ignore_index=True)

    df.to_csv(f'./data/{project}/{model}.csv', index=False)

def read_files_to_df(files, delimiter, decimal):

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(file, sep=delimiter, decimal=decimal), ignore_index=True)

    return df


def get_models():
    return {folder: [file[:-4] for file in os.listdir(f'./data/{folder}')] for folder in os.listdir('./data')}

def read_files(project, models):
    df = pd.DataFrame()

    for model in models:
        df = df.append(pd.read_csv(f'./data/{project}/{model}.csv'), ignore_index=True)

    return df

def delete_project(project):
    shutil.rmtree(f'./data/{project}')

def delete_model(project, model):
    os.remove(f'./data/{project}/{model}.csv')
