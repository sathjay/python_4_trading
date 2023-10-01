import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import dash  # (version 1.9.1) pip install dash==1.9.1
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import numpy as np
from datetime import datetime as dt
import datetime
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
from app import app

stock_list = pd.read_csv('data/Company_Name_List.csv')
stock_list.set_index('Symbol', inplace=True)
# The reason why Symbol is set as index is that in the next step, fetching the name of company is easy.
stock_dropdown_list = []
for symbol in stock_list.index:
    stock_dropdown_list.append(
        {'label': '{} - {}'.format(symbol, stock_list.loc[symbol]['Name']), 'value': symbol})
now = datetime.datetime.now()
start_date = '2012-01-02'
end_date = datetime.datetime.now()
dt = pd.to_datetime(start_date, format='%Y-%m-%d')
dt1 = pd.to_datetime(end_date, format='%Y-%m-%d')
num_of_days_btw_start_and_end = (dt1-dt).days
number_of_years = num_of_days_btw_start_and_end/365


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, plot_bgcolor='rgb(186, 228, 242)',
                      paper_bgcolor='rgb(186, 228, 242)')
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


days_dropdown_list = [
    {'label': '1 Year', 'value': '252'},
    {'label': '2 Year', 'value': '504'},
    {'label': '3 Year', 'value': '756'},
    {'label': '5 Year', 'value': '1260'},
    {'label': '7 Year', 'value': '1764'},
    {'label': 'All', 'value': 'All'}]


def price_chart(fig, df, row, column=1):
    """Return a graph object figure containing price, EMA chart, buy and sell signal"""

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Close Price'],
                             name='Price',
                             mode='lines'),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Buy'],
                             name='Buy Signal',
                             mode='markers',
                             marker=dict(
        size=11,
        color='green',
        symbol='triangle-up',
        line={'width': 2})),
        row=row,
        col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Sell'],
                             name='Sell Signal',
                             mode='markers',
                             marker=dict(
        size=11,
        color='red',
        symbol='triangle-down',
        line={'width': 2})),
        row=row,
        col=column)

    fig.update_yaxes(title_text='Price and Signal', row=row, col=column)

    return fig


def plot_ADX(fig, df, row, column=1):
    """Return a graph object figure containing the DMI and ADX indicator"""

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['ADX'],
                             name='ADX',
                             line=dict(color='darkorange', width=2)),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['plusDI'],
                             name='+ve DMI',
                             line=dict(color='green', width=2)),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MinusDM'],
                             name='-ve DMI',
                             line=dict(color='red', width=2)),
                  row=row,
                  col=column)

    fig.update_yaxes(
        title_text='ADX and +/- DMI Lines', row=row, col=column)

    return fig


def pnl_strategy(df, selected_stock):

    pnl_chart_data = []

    trace1 = go.Scatter(x=df['Date'],
                        y=df['c_return'],
                        name='Buy and Hold',
                        mode='lines',
                        opacity=.8)
    pnl_chart_data.append(trace1)

    trace2 = go.Scatter(x=df['Date'],
                        y=df['c_system_return'],
                        name='Strategy Return',
                        mode='lines',
                        opacity=.8)
    pnl_chart_data.append(trace2)

    trace3 = go.Scatter(x=df['Date'],
                        y=df['c_long_return'],
                        name='Long only',
                        mode='lines',
                        opacity=.8)
    pnl_chart_data.append(trace3)

    pnl_layout = go.Layout(
        title='PNL of the Strategy for {}'.format(selected_stock),
        yaxis=dict(title='Cummilative Return',
                   showline=True,
                   linecolor='black',
                   showgrid=True),
        xaxis=dict(title='Date',
                   showline=True,
                   linecolor='black',
                   showgrid=True),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='h',
            x=.45,
            y=1.12,
            traceorder='normal',
            borderwidth=1)
    )

    fig = go.Figure(data=pnl_chart_data, layout=pnl_layout)

    return fig


ADX_LO = html.Div([

    html.H5('Average Directional Index (ADX) Strategy:',
            className='content_title'),

    html.P(['Purpose: The ADX measures the strength of a trend irrespective of its direction.',
            html.Br(),
            'Calculation Basis: The ADX is calculated using a moving average of price range expansion, typically over a default period of 14 days, although this can be adjusted.',
            html.Br(),
            'Value Range: ADX values range from 0 to 100. A higher value (typically above 25 or 30) indicates a stronger trend, while values below 20 often suggest a weak or non-existent trend.',
            html.Br(),
            'Directional Indicators: The ADX is often plotted alongside two other indicators: the Positive Directional Movement Indicator (+DMI) and the Negative Directional Movement Indicator (-DMI). Their relative positions indicate the direction of the trend:',
            html.Li(['If +DMI is above -DMI, then the trend is up.']),
            html.Li(['If -DMI is above +DMI, then the trend is down.']),
            html.Li(
                ['If +DMI and -DMI are close together, then there is no trend.']),
            'By gauging the strength and direction of trends, the ADX can be a valuable tool for traders looking to capitalize on strong directional movements in the market'
            ]),

    html.Label(['Select Company/EFT from dorp down:'],
               style={'font-weight': 'bold'}),
    html.Br(),
    dcc.Dropdown(id='selected_stock',
                 options=stock_dropdown_list,
                 optionHeight=35,
                 value='QQQ',
                 disabled=False,  # disable dropdown value selection
                 multi=False,  # allow multiple dropdown values to be selected
                 searchable=True,  # allow user-searching of dropdown values
                 # gray, default text shown when no option is selected
                 placeholder='Please select...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',  # activate separate CSS document in assets folder
                 ),

    html.Div([

        html.Label(['Select the Average True Range moving average days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=30,
                   id='ATR_period',
                   step=1,
                   value=14,
                   marks={
                       5: '5',
                       30: '30'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),

    html.Br(),
    html.Label(['Select the number of days in past to Test the Strategy'],
               style={'font-weight': 'bold'}),
    dcc.Dropdown(id="backtest_days",
                 multi=False,
                 disabled=False,
                 value='504',
                 options=days_dropdown_list,
                 # gray, default text shown when no option is selected
                 placeholder='Please select...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',
                 ),

    html.Button('Submit', id='BT_ADX',
                className='button', n_clicks=0),
    html.Div([], className='content_divider'),
    dcc.Loading(children=[

        html.Label(id='Selected_Value_ADX'),

        html.Br(),
        dcc.Graph(id='entry_and_exit_ADX', figure=blank_fig(),
                  config={'displayModeBar': False}),
        html.Br(),
        dcc.Graph(id='pnl_chart_ADX', figure=blank_fig(),
                  config={'displayModeBar': False}),
        html.Br(),
    ], type="circle", fullscreen=True),

])


@app.callback(
    [Output('Selected_Value_ADX', 'children'),
     Output('entry_and_exit_ADX', component_property='figure'),
     Output('pnl_chart_ADX', component_property='figure')],
    [Input('BT_ADX', 'n_clicks')],
    [
        State('selected_stock', 'value'),
        State('ATR_period', 'value'),
        State('backtest_days', 'value'),
    ],
    prevent_initial_call=True)
def ADX_payoff(n_clicks, selected_stock, ATR_period, backtest_days):
    ATR_period = int(ATR_period)

    message = html.P(
        "You have selected {} days as ADX time period".format(ATR_period))

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)

    if backtest_days == 'All':
        backtest_days = len(df)
    else:
        backtest_days = int(backtest_days)

    df.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

    dff1 = df.iloc[-backtest_days:]

    dff1['HL'] = dff1['High'] - dff1['Low']
    dff1['absHC'] = abs(dff1['High'] - dff1['Close'].shift(1))
    dff1['absLC'] = abs(dff1['Low'] - dff1['Close'].shift(1))
    dff1['TR'] = dff1[['HL', 'absHC', 'absLC']].max(axis=1)
    dff1['ATR'] = dff1['TR'].rolling(window=ATR_period).mean()

    dff1['UpMove'] = dff1['High'] - dff1['High'].shift(1)
    dff1['DownMove'] = dff1['Low'].shift(1) - dff1['Low']
    dff1['Zero'] = 0

    dff1['PlusDM'] = np.where((dff1['UpMove'] > dff1['DownMove']) & (
        dff1['UpMove'] > dff1['Zero']), dff1['UpMove'], 0)
    dff1['MinusDM'] = np.where((dff1['UpMove'] < dff1['DownMove']) & (
        dff1['DownMove'] > dff1['Zero']), dff1['DownMove'], 0)

    dff1['plusDI'] = 100 * (dff1['PlusDM']/dff1['ATR']).ewm(span=ATR_period,
                                                            min_periods=0, adjust=True, ignore_na=False).mean()
    dff1['minusDI'] = 100 * (dff1['MinusDM']/dff1['ATR']).ewm(
        span=ATR_period, min_periods=0, adjust=True, ignore_na=False).mean()

    dff1['ADX'] = 100 * (abs((dff1['plusDI'] - dff1['minusDI'])/(dff1['plusDI'] + dff1['minusDI']))
                         ).ewm(span=ATR_period, min_periods=0, adjust=True, ignore_na=False).mean()

    dff1.dropna(inplace=True)

    # Trader always invested

    dff1['combined_signal'] = np.where(dff1['plusDI'] > dff1['minusDI'], 1, 0)
    dff1['combined_signal'] = np.where(
        dff1['plusDI'] < dff1['minusDI'], -1, dff1['combined_signal'])

    dff1['return'] = np.log(dff1['Close Price']).diff().shift(-1)
    dff1['system_return'] = dff1['combined_signal']*dff1['return']

    dff1['c_return'] = np.exp(dff1['return']).cumprod()
    dff1['c_system_return'] = np.exp(dff1['system_return']).cumprod()

    # Trader Long only invested

    dff1['long_signal'] = np.where(dff1['plusDI'] > dff1['minusDI'], 1, 0)
    dff1['long_return'] = dff1['long_signal']*dff1['return']
    dff1['c_long_return'] = np.exp(dff1['long_return']).cumprod()

    # Buy and Sell Signal

    dff1['signal'] = np.where(dff1['plusDI'] > dff1['minusDI'], 1, 0)
    dff1['position'] = dff1['signal'].diff()

    dff1['Buy'] = np.where(dff1['position'] == 1, dff1['Close Price'], np.nan)
    dff1['Sell'] = np.where(dff1['position'] == -1,
                            dff1['Close Price'], np.nan)

    # dff1.to_csv('08_ADX.csv',mode = 'w',sep = '|')
    dff1.reset_index(inplace=True)

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.025,
                        row_width=[0.2, 0.3])
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        showgrid=True)
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        showgrid=True)
    fig = price_chart(fig, dff1, row=1)
    fig = plot_ADX(fig, dff1, row=2)

    fig.update_layout(
        title='ADX Strategy for {}'.format(selected_stock),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='v',
            x=.93,
            y=1.18,
            traceorder='normal',
            borderwidth=1)
    )

    fig1 = pnl_strategy(dff1, selected_stock)

    return message, fig, fig1
