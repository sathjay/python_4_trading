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


def plot_OBV(fig, df, row, column=1):
    """Return a graph object figure containing the OBV, OBV EMA and a histogram in the specified row."""

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['OBV'],
                             name='OBV',
                             line=dict(color='darkorange', width=2)),
                  secondary_y=False,
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['OBV_EMA'],
                             name='OBV EMA',
                             line=dict(color='green', width=2)),
                  secondary_y=False,
                  row=row,
                  col=column)

    fig.add_trace(go.Bar(x=df['Date'],
                         y=df['Volume'],
                         name='Volume',
                         showlegend=True),
                  secondary_y=True,
                  row=row,
                  col=column)

    fig.update_yaxes(title_text='OBV', row=row, col=column, secondary_y=False,)
    fig.update_yaxes(title_text='Volume', row=row,
                     col=column, secondary_y=True,)

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


OBV_LO = html.Div([

    html.H5('On Balance Volume Trading Strategy:', className='content_title'),

    html.P('On-balance volume (OBV) is a simple accumulation-distribution tool that tallies up and down volume creating a smooth indicator line to predict when major market moves might occur based on changes in relative trading volume.'),
    html.P("On-balance volume provides a running total of an asset's trading volume and indicates whether this volume is flowing in or out of a given security or currency pair. The OBV is a cumulative total of volume (positive and negative)."),
    html.Br(),

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
        html.Label(['Select the On Balance Volumne (OBV) Exponential Moving Average days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=30,
                   id='OBV_EMA_period',
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

    html.Button('Submit', id='BT_OBV',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),

    dcc.Loading(children=[

        html.Label(id='Selected_Value_OBV'),

        html.Div(id='OBV_note', children=[]),

        html.Br(),
        dcc.Graph(id='entry_and_exit_OBV', figure=blank_fig(),
                  config={'displayModeBar': False}),
        html.Br(),
        dcc.Graph(id='pnl_chart_OBV', figure=blank_fig(),
                  config={'displayModeBar': False}),
        html.Br(),

    ], type="circle", fullscreen=True),
])


@app.callback(
    [Output('Selected_Value_OBV', 'children'),
     Output('OBV_note', 'children'),
     Output('entry_and_exit_OBV', component_property='figure'),
     Output('pnl_chart_OBV', component_property='figure')],
    [Input('BT_OBV', 'n_clicks')],
    [
        State('selected_stock', 'value'),
        State('OBV_EMA_period', 'value'),
        State('backtest_days', 'value'),
    ],
    prevent_initial_call=True)
def OBV_payoff(n_clicks, selected_stock, OBV_EMA_period, backtest_days):
    OBV_EMA_period = int(OBV_EMA_period)

    message = html.P(
        "You have selected {} days as OBV time period".format(OBV_EMA_period))

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)

    OBV_note = html.P(
        'Assumpution: In this OBV strategy, Buy signal is generated if the OBV line is higher than the OBV EMA line and sell signal is generated if OBV line is lower than OBV EMA line', className='note'),

    if backtest_days == 'All':
        backtest_days = len(df)
    else:
        backtest_days = int(backtest_days)

    dff = df[['Adj Close', 'Volume']]
    dff.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

    # Calculate OBV
    # dff.to_csv('OBV_Input.csv',mode = 'w',sep = '|')
    OBV = []
    OBV.append(0)

    for i in range(1, len(dff)):
        if dff['Close Price'][i] > dff['Close Price'][i-1]:
            OBV.append(OBV[-1] + dff['Volume'][i])
        elif dff['Close Price'][i] < dff['Close Price'][i-1]:
            OBV.append(OBV[-1] - dff['Volume'][i])
        else:
            OBV.append(OBV[-1])

    dff['OBV'] = OBV
    dff['OBV_EMA'] = dff['OBV'].ewm(span=OBV_EMA_period).mean()

    dff1 = dff.iloc[-backtest_days:]

    dff1.dropna(inplace=True)

    # Trader always invested

    dff1['combined_signal'] = np.where(dff1['OBV'] > dff1['OBV_EMA'], 1, 0)
    dff1['combined_signal'] = np.where(
        dff1['OBV'] < dff1['OBV_EMA'], -1, dff1['combined_signal'])

    dff1['return'] = np.log(dff1['Close Price']).diff().shift(-1)
    dff1['system_return'] = dff1['combined_signal']*dff1['return']

    dff1['c_return'] = np.exp(dff1['return']).cumprod()
    dff1['c_system_return'] = np.exp(dff1['system_return']).cumprod()

    # Trader Long only invested

    dff1['long_signal'] = np.where(dff1['OBV'] > dff1['OBV_EMA'], 1, 0)
    dff1['long_return'] = dff1['long_signal']*dff1['return']
    dff1['c_long_return'] = np.exp(dff1['long_return']).cumprod()

    # Buy and Sell Signal

    dff1['signal'] = np.where(dff1['OBV'] > dff1['OBV_EMA'], 1, 0)
    dff1['position'] = dff1['signal'].diff()

    dff1['Buy'] = np.where(dff1['position'] == 1, dff1['Close Price'], np.nan)
    dff1['Sell'] = np.where(dff1['position'] == -1,
                            dff1['Close Price'], np.nan)

    # dff1.to_csv('OBV.csv',mode = 'w',sep = '|')

    dff1.reset_index(inplace=True)

    fig = make_subplots(rows=2,
                        cols=1,
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": True}]],
                        shared_xaxes=True,
                        vertical_spacing=0.025,
                        row_width=[0.2, 0.3])

    fig = price_chart(fig, dff1, row=1)
    fig = plot_OBV(fig, dff1, row=2)
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        showgrid=True)
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        showgrid=True)

    fig.update_layout(

        title='OBV Strategy for {}'.format(selected_stock),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='v',
            x=.92,
            y=1.14,
            traceorder='normal',
            borderwidth=1)
    )

    fig1 = pnl_strategy(dff1, selected_stock)

    return message, OBV_note, fig, fig1
