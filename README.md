# HyperViz - Analysing Hyperparameters Made Easy

This app allows you to easily analyse hyperparameters with visual tools. No more looking at endless rows of numbers!

## Demo-Version
The demo version is available as an official streamlit sharing app [here](https://hyperviz.mfinn.de). 

Uploaded files will be public and might get deleted without warning!

## GridsearchCV example:

For a little example on how to use GridsearchCV and how to generate a .csv file from that, look [here](https://github.com/F1nnM/HyperViz/blob/main/example_titanic_rf.py).

## Run your own version with Docker:
```
docker run -p <local port>:8501 f1nnm/hyperviz:latest
```
It's accessible under `https://0.0.0.0:<local port>`

## Run from source:
1. Clone this repo.
2. Navigate to where you saved it.
3. Install the python packages: `pip install -r requirements.txt`
4. Run `streamlit run app.py`
