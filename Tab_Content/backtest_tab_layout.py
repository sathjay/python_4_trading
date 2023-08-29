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

from Backtest_Dropdown_Content.backtest_SMA import SMA_LO
from Backtest_Dropdown_Content.backtest_EMA import EMA_LO
from Backtest_Dropdown_Content.backtest_DEMA import DEMA_LO
from Backtest_Dropdown_Content.backtest_MACD import MACD_LO
from Backtest_Dropdown_Content.backtest_Mean_Reversion import MR_LO
from Backtest_Dropdown_Content.backtest_RSI import RSI_LO
from Backtest_Dropdown_Content.backtest_ADX import ADX_LO
from Backtest_Dropdown_Content.backtest_MFI import MFI_LO
from Backtest_Dropdown_Content.backtest_OBV import OBV_LO

trading_strategy_list = [
    {'label': '01 Simple Moving Average(SMA)', 'value': '1'},
    {'label': '02 Exponential Moving Average(EMA)', 'value': '2'},
    {'label': '03 Double Exponential Moving Average(DEMA)', 'value': '3'},
    {'label': '04 Moving Average Convergence Divergence(MACD)', 'value': '4'},
    {'label': '05 Mean Reversion', 'value': '5'},
    {'label': '06 Relative Strength Indicator(RSI)', 'value': '6'},
    {'label': '07 Average Directional Index(ADX)', 'value': '7'},
    {'label': '08 Money Flow Index(MFI)', 'value': '8'},
    {'label': '09 On Balance Volume(OBV)', 'value': '9'},
]


backtest_tab_LO = html.Div([

    html.Div([

        html.H4('Backtesting Trading Strategy:'),

        html.P(['Backtesting allows traders to simulate a trading strategy using historical data, generating results and analyzing risk before committing any actual capital. The underlying theory is that a strategy that worked well in the past is likely to succeed in the future. Conversely, a strategy that performed poorly in the past is likely to do so in the future.',
                html.Br(),
                "However, it's crucial to understand that past successes do not guarantee future success. Some strategies are tailored for specific stocks, time durations, or market conditions, and their effectiveness can vary accordingly."]),

        html.Label(['Select the Trading Strategy to be tested: '],
                   style={'font-weight': 'bold'}),

        dcc.Dropdown(id='trading_strategy',
                     options=trading_strategy_list,
                     optionHeight=35,
                     value='1',  # dropdown value selected automatically when page loads
                     disabled=False,  # disable dropdown value selection
                     multi=False,  # allow multiple dropdown values to be selected
                     searchable=True,  # allow user-searching of dropdown values
                     # gray, default text shown when no option is selected
                     placeholder='Please select a Trading Strategy',
                     clearable=True,  # allow user to removes the selected value
                     className='dropdown_box',  # activate separate CSS document in assets folder
                     ),

        html.Div([], className='content_divider'),

        html.Div(id='backtest_tab_content', children=[]),

    ], className='main_container'),




])


@app.callback(Output('backtest_tab_content', 'children'),
              [Input('trading_strategy', 'value')]
              )
def update_backtest_tab(trading_strategy):

    if trading_strategy == '1':
        return SMA_LO
    elif trading_strategy == '2':
        return EMA_LO
    elif trading_strategy == '3':
        return DEMA_LO
    elif trading_strategy == '4':
        return MACD_LO
    elif trading_strategy == '5':
        return MR_LO
    elif trading_strategy == '6':
        return RSI_LO
    elif trading_strategy == '7':
        return ADX_LO
    elif trading_strategy == '8':
        return MFI_LO
    elif trading_strategy == '9':
        return OBV_LO
