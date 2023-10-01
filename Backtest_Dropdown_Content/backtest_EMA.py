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
    {'label': '3 Month', 'value': '66'},
    {'label': '6 Month', 'value': '126'},
    {'label': '1 Year', 'value': '252'},
    {'label': '2 Year', 'value': '504'},
    {'label': '3 Year', 'value': '756'},
    {'label': '5 Year', 'value': '1260'},
    {'label': 'All', 'value': 'All'}]


EMA_LO = html.Div([

    html.H5('Exponential Moving Average (EMA) Crossover Strategy:',
            className='content_title'),

    html.P('In the EMA Crossover strategy:'),
    html.Li('A long position is initiated when the fast EMA (short duration) crosses above the slow EMA (long duration).'),
    html.Li('Conversely, a short position is taken when the fast EMA crosses below the slow EMA.'),
    html.P('In essence, this strategy capitalizes on momentum changes indicated by the crossover of two moving averages of different durations.'),


    html.Label(['Select Company/EFT from dorp down:'],
               style={'font-weight': 'bold'}),

    html.Br(),

    dcc.Dropdown(id='my_selected_stock',
                 options=stock_dropdown_list,
                 optionHeight=35,
                 value='SPY',
                 disabled=False,  # disable dropdown value selection
                 multi=False,  # allow multiple dropdown values to be selected
                 searchable=True,  # allow user-searching of dropdown values
                 # gray, default text shown when no option is selected
                 placeholder='Please select...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',  # activate separate CSS document in assets folder
                 ),


    html.Div([

        html.Label(['Select the Fast(short-duration) moving average days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=50,
                   id='FMA',
                   step=1,
                   value=10,
                   marks={
                       5: '5',
                       50: '50'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),


    html.Br(),
    html.Div([
        html.Label(['Select the Slow(long-duration) moving average days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=10,
                   max=100,
                   id='SMA',
                   step=1,
                   value=21,
                   marks={
                       10: '10',
                       100: '100'
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

    html.Button('Submit', id='BT_EMA',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),

    dcc.Loading(children=[

        html.Label(id='Selected_Value_EMA'),

        html.Div(id='EMA_note', children=[]),

        html.Hr(),
        dcc.Graph(id='entry_and_exit_EMA', figure=blank_fig(),
                  config={'displayModeBar': False}),


        html.Hr(),
        dcc.Graph(id='pnl_chart_EMA', figure=blank_fig(),
                  config={'displayModeBar': False}),

    ], type="circle", fullscreen=True),


])


@app.callback(
    [Output('Selected_Value_EMA', 'children'),
     Output('EMA_note', 'children'),
     Output('entry_and_exit_EMA', component_property='figure'),
     Output('pnl_chart_EMA', component_property='figure')],
    [Input('BT_EMA', 'n_clicks')],
    [
        State('my_selected_stock', 'value'),
        State('FMA', 'value'),
        State('SMA', 'value'),
        State('backtest_days', 'value'),
    ],
    prevent_initial_call=True)
def cal_Expo_Moving_Avg_payoff(n_clicks, selected_stock, FMA, SMA, backtest_days):

    FMA = int(FMA)
    SMA = int(SMA)

    if SMA <= FMA:
        message = html.P("The Slow moving average days should be greater than fast moving average",
                         style={'color': 'red', 'font-size': '16px'})
        fig = blank_fig()
        fig1 = blank_fig()
        EMA_note = ''
        return message, EMA_note, fig, fig1

    else:
        message = html.P("You have selected {} days for Fast moving average and {} days for Slow moving average".format(FMA, SMA),
                         style={'color': 'black', 'font-size': '18px'})

        EMA_note = html.P(["Assumption: At start, if the Fast moving average is greater than the slow moving average then trader takes the long postion and if the Fast moving average is less than the slow moving average then trader takes the short postion."], className='note'
                          )

        df = yf.download(selected_stock, period='1d',
                         start=start_date, end=end_date)

        day = np.arange(1, len(df)+1)
        df['Day'] = day

        dff = df[['Day', 'Adj Close']]
        dff.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

        FMA = int(FMA)
        SMA = int(SMA)

        dff['Fast_EMA'] = dff['Close Price'].ewm(span=FMA, adjust=False).mean()
        dff['Slow_EMA'] = dff['Close Price'].ewm(span=SMA, adjust=False).mean()

        dff.dropna(inplace=True)

        if backtest_days == 'All':
            backtest_days = len(dff)

        else:
            backtest_days = int(backtest_days)

        dff1 = dff.iloc[-backtest_days:]

        # Trader always invested

        dff1['combined_signal'] = np.where(
            dff1['Fast_EMA'] > dff1['Slow_EMA'], 1, 0)
        dff1['combined_signal'] = np.where(
            dff1['Fast_EMA'] < dff1['Slow_EMA'], -1, dff1['combined_signal'])

        dff1['return'] = np.log(dff1['Close Price']).diff()
        dff1['system_return'] = dff1['combined_signal']*dff1['return']

        dff1['c_return'] = np.exp(dff1['return']).cumprod()
        dff1['c_system_return'] = np.exp(dff1['system_return']).cumprod()

        # Trader Long only invested

        dff1['long_signal'] = np.where(
            dff1['Fast_EMA'] > dff1['Slow_EMA'], 1, 0)
        dff1['long_return'] = dff1['long_signal']*dff1['return'].shift(-1)
        dff1['c_long_return'] = np.exp(dff1['long_return']).cumprod()

        # Buy and Sell Signal

        dff1['signal'] = np.where(dff1['Fast_EMA'] > dff1['Slow_EMA'], 1, 0)
        dff1['position'] = dff1['signal'].diff()

        dff1['Buy'] = np.where(dff1['position'] == 1,
                               dff1['Close Price'], np.nan)
        dff1['Sell'] = np.where(dff1['position'] == -1,
                                dff1['Close Price'], np.nan)

        # Trader Initial Position
        if dff1['Fast_EMA'][1] > dff1['Slow_EMA'][1]:
            dff1['Buy'][1] = dff1['Close Price'][1]
        else:
            dff1['Sell'][1] = dff1['Close Price'][1]

        # dff1.to_csv('simple_moving_avg.csv',mode = 'w',sep = '|')

        column_list = ['Close Price', 'Fast_EMA', 'Slow_EMA']
        price_chart_data = []
        dff1.reset_index(inplace=True)

        # Price and entry and exit chart

        for column in column_list:
            trace = go.Scatter(x=dff1['Date'],
                               y=dff1[column],
                               mode='lines',
                               name=column)
            price_chart_data.append(trace)

        trace_buy_signal = go.Scatter(x=dff1['Date'],
                                      y=dff1['Buy'],
                                      mode='markers',
                                      name='Buy',
                                      marker=dict(
            size=11,
            color='green',
            symbol='triangle-up',
            line={'width': 2}))

        price_chart_data.append(trace_buy_signal)

        trace_sell_signal = go.Scatter(x=dff1['Date'],
                                       y=dff1['Sell'],
                                       mode='markers',
                                       name='Sell',
                                       marker=dict(
            size=11,
            color='red',
            symbol='triangle-down',
            line={'width': 2}))

        price_chart_data.append(trace_sell_signal)

        price_chart_layout = go.Layout(
            title='Price Chart Entry and Exit for {}'.format(selected_stock),
            yaxis=dict(title='Price',
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

        fig = go.Figure(data=price_chart_data, layout=price_chart_layout)

        # Pay Off Chart

        pnl_chart_data = []

        trace_pnl_1 = go.Scatter(x=dff1['Date'],
                                 y=dff1['c_return'],
                                 mode='lines',
                                 name='Buy and Hold',
                                 opacity=.8)

        pnl_chart_data.append(trace_pnl_1)

        trace_pnl_2 = go.Scatter(x=dff1['Date'],
                                 y=dff1['c_system_return'],
                                 mode='lines',
                                 name='Fully Invested',
                                 line=dict(color='firebrick', width=2),
                                 opacity=.8
                                 )

        pnl_chart_data.append(trace_pnl_2)

        trace_pnl_3 = go.Scatter(x=dff1['Date'],
                                 y=dff1['c_long_return'],
                                 mode='lines',
                                 name='Long Only',
                                 line=dict(color='royalblue', width=2),
                                 opacity=.8
                                 )

        pnl_chart_data.append(trace_pnl_3)

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

        fig1 = go.Figure(data=pnl_chart_data, layout=pnl_layout)

    return message, EMA_note, fig, fig1
