import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import dash  # (version 1.9.1) pip install dash==1.9.1
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from app import app

from ML_Predictions.ML_XGBoost import ML_XGBoost_LO
from ML_Predictions.ML_LSTM_Single_Layer import LSTM_SL_LO
from ML_Predictions.ML_LSTM_Multi_Layer import LSTM_ML_LO

ml_model_list = [
    {'label': '01 XGBoost Classifier with Seasonality Decomposition', 'value': '1'},
    {'label': '02 Single layer LSTM Recurring Neural Network', 'value': '2'},
    {'label': '03 Multi layer LSTM Recurring Neural Network', 'value': '3'},
]

ML_tab_LO = html.Div([

    html.Div([

        html.H4('Next Day Stock Price Prediction using Machine Learning (ML):'),

        html.P('From the dropdown menu below, users can select various ML models, stocks, input features, and model parameters to predict the magnitude or direction of stock price changes for the next day.'),
        html.Br(),
        html.P('Note of Caution: Predicting future stock prices is an extremely challenging task. Often, stock prices are non-stationary, meaning they can change unpredictably over time. Numerous factors, including geo-political events, industrial accidents, and significant news events, can influence stock prices. Many of these factors are random and difficult to model accurately.', className='note'),

        html.Label(['Select the ML Model to predict from the options below: '],
                   style={'font-weight': 'bold'}),

        dcc.Dropdown(id='ml_model',
                     options=ml_model_list,
                     optionHeight=35,
                     # value='1',  # dropdown value selected automatically when page loads
                     disabled=False,  # disable dropdown value selection
                     multi=False,  # allow multiple dropdown values to be selected
                     searchable=True,  # allow user-searching of dropdown values
                     # gray, default text shown when no option is selected
                     placeholder='Please select a Trading Strategy',
                     clearable=True,  # allow user to removes the selected value
                     className='dropdown_box',  # activate separate CSS document in assets folder
                     ),

        html.P("Note: Options 2 and 3 are Neural Networks, which require a lot of computing power. The results might take a brief moment.", className='note'),
        html.Br(),

        html.Div([], className='content_divider'),



        html.Div(id='ml_prediction_content', children=[]),


    ], className='main_container'),

])


@app.callback(Output('ml_prediction_content', 'children'),
              [Input('ml_model', 'value')]
              )
def update_ml_tab(ml_model):
    if ml_model == '1':
        return ML_XGBoost_LO
    elif ml_model == '2':
        return LSTM_SL_LO
    elif ml_model == '3':
        return LSTM_ML_LO
