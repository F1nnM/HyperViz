import os
import pandas as pd

from pathlib import Path

def process_data(df, project, model, columns_to_keep):
    # dataprep

    df = df[columns_to_keep]

    path = Path(f'./data/{project}')
    path.mkdir(parents=True, exist_ok=True)

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

    # dataprep

    return df
