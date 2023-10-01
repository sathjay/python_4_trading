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
                             y=df['Fast_EMA'],
                             name='Fast EMA',
                             mode='lines'),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Slow_EMA'],
                             name='Slow EMA',
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


def plot_MACD(fig, df, row, column=1):
    """Return a graph object figure containing the MACD indicator, the signal line, and a histogram in the specified row."""
    df['Hist-Color'] = np.where(df['Histogram'] < 0, 'red', 'green')
    fig.add_trace(go.Bar(x=df['Date'],
                         y=df['Histogram'],
                         name='Histogram',
                         marker_color=df['Hist-Color'],
                         showlegend=True),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['MACD_Line'],
                             name='MACD',
                             line=dict(color='darkorange', width=2)),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Signal_line'],
                             name='Signal',
                             line=dict(color='cyan', width=2)),
                  row=row,
                  col=column)

    fig.update_yaxes(title_text='MACD', row=row, col=column)

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
            borderwidth=1))

    fig = go.Figure(data=pnl_chart_data, layout=pnl_layout)

    return fig


MACD_LO = html.Div([

    html.H5('Moving Average Convergence Divergence (MACD) Crossover Strategy:',
            className='content_title'),

    html.P('The MACD indicator works using three components: the MACD line, the signal line, and a histogram.'),
    html.Ol("1) MACD Line: This is the difference between two exponential moving averages (EMAs) of a security's price, typically using 12-day and 26-day EMAs."),
    html.Ol("2) Signal Line: This is a 9-day EMA of the MACD line itself."),
    html.Ol("3) Histogram: Represents the difference between the MACD line and the signal line. It gives a visual representation of the distance between the two lines. If the MACD line is above the signal line, the histogram will be above the zero line and vice versa."),

    html.P('Key Points:'),
    html.Li("Convergence: When the MACD line and the signal line come closer together or cross, they are said to be 'converging.' This often indicates a weakening trend."),
    html.Li("Divergence: When the two lines move away from each other, they are 'diverging,' often indicating a strengthening trend."),
    html.Li("Trading Signals: The primary trading signal occurs when the MACD line crosses the signal line. A bullish signal (buy) is generated when the MACD line crosses above the signal line, and a bearish signal (sell) is generated when the MACD line crosses below the signal line."),




    html.Label(['Select Company/EFT from dorp down:'],
               style={'font-weight': 'bold'}),

    html.Br(),

    dcc.Dropdown(id='selected_stock',
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
                   max=20,
                   id='FMA',
                   step=1,
                   value=12,
                   marks={
                       5: '5',
                       20: '20'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),


    html.Br(),
    html.Div([
        html.Label(['Select the Slow(long-duration) moving average days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=10,
                   max=36,
                   id='SMA',
                   step=1,
                   value=26,
                   marks={
                       10: '10',
                       36: '36'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),

    html.Br(),
    html.Div([
        html.Label(['Select the Signal Line Moving Average Days'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=20,
                   id='SIG',
                   step=1,
                   value=9,
                   marks={
                       5: '5',
                       20: '20'
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
                 value='252',
                 options=days_dropdown_list,
                 # gray, default text shown when no option is selected
                 placeholder='Please select...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',
                 ),

    html.Button('Submit', id='BT_MACD',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),

    dcc.Loading(children=[

        html.Label(id='Selected_Value_MACD'),

        html.Div(id='MACD_note', children=[]),

        html.Hr(),
        dcc.Graph(id='entry_and_exit_MACD', figure=blank_fig(),
                  config={'displayModeBar': False}),


        html.Hr(),
        dcc.Graph(id='pnl_chart_MACD', figure=blank_fig(),
                  config={'displayModeBar': False}),

    ], type="circle", fullscreen=True),

])


@app.callback(
    [Output('Selected_Value_MACD', 'children'),
     Output('MACD_note', 'children'),
     Output('entry_and_exit_MACD', component_property='figure'),
     Output('pnl_chart_MACD', component_property='figure')],
    [Input('BT_MACD', 'n_clicks')],
    [
        State('selected_stock', 'value'),
        State('FMA', 'value'),
        State('SMA', 'value'),
        State('SIG', 'value'),
        State('backtest_days', 'value'),
    ],
    prevent_initial_call=True
)
def caluculate_moving_average_payoff(n_clicks, selected_stock, FMA, SMA, SIG, backtest_days):

    FMA = int(FMA)
    SMA = int(SMA)
    SIG = int(SIG)

    if SMA <= FMA:
        message = html.P("The Slow moving average days should be greater than fast moving average",
                         style={'color': 'red', 'font-size': '16px'})
        fig = blank_fig()
        fig1 = blank_fig()
        MACD_note = ''
        return message, MACD_note, fig, fig1

    else:
        message = html.P("You have selected {} days for Slow moving exp. average and {} days for Fast moving exp average. \
                        Signal Line moving average days is {}".format(SMA, FMA, SIG))

        MACD_note = html.P(["Assumption: At start, Buy signal is generated if the MACD line is above the signal line and Sell signal is generated if the MACD line is below the sigmal line."], className='note'
                           )

        df = yf.download(selected_stock, period='1d',
                         start=start_date, end=end_date)

        day = np.arange(1, len(df)+1)
        df['Day'] = day

        dff = df[['Day', 'Adj Close']]
        dff.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

        dff['Fast_EMA'] = dff['Close Price'].ewm(span=FMA, adjust=False).mean()
        dff['Slow_EMA'] = dff['Close Price'].ewm(span=SMA, adjust=False).mean()

        # MACD Line
        dff['MACD_Line'] = dff['Fast_EMA']-dff['Slow_EMA']

        # Signal Line
        dff['Signal_line'] = dff['MACD_Line'].ewm(
            span=SIG, adjust=False).mean()

        # MACD minus Signal line for Historgram
        dff['Histogram'] = dff['MACD_Line']-dff['Signal_line']

        dff.dropna(inplace=True)

        if backtest_days == 'All':
            backtest_days = len(dff)

        else:
            backtest_days = int(backtest_days)

        dff1 = dff.iloc[-backtest_days:]

        # Trader always invested

        dff1['combined_signal'] = np.where(
            dff1['MACD_Line'] > dff1['Signal_line'], 1, 0)
        dff1['combined_signal'] = np.where(
            dff1['MACD_Line'] < dff1['Signal_line'], -1, dff1['combined_signal'])

        dff1['return'] = np.log(dff1['Close Price']).diff().shift(-1)
        dff1['system_return'] = dff1['combined_signal']*dff1['return']

        dff1['c_return'] = np.exp(dff1['return']).cumprod()
        dff1['c_system_return'] = np.exp(dff1['system_return']).cumprod()

        # Trader Long only invested

        dff1['long_signal'] = np.where(
            dff1['MACD_Line'] > dff1['Signal_line'], 1, 0)
        dff1['long_return'] = dff1['long_signal']*dff1['return']
        dff1['c_long_return'] = np.exp(dff1['long_return']).cumprod()

        # Buy and Sell Signal

        dff1['signal'] = np.where(
            dff1['MACD_Line'] > dff1['Signal_line'], 1, 0)
        dff1['position'] = dff1['signal'].diff()

        dff1['Buy'] = np.where(dff1['position'] == 1,
                               dff1['Close Price'], np.nan)
        dff1['Sell'] = np.where(dff1['position'] == -1,
                                dff1['Close Price'], np.nan)

        # Trader Initial Position
        if dff1['MACD_Line'][1] > dff1['Signal_line'][1]:
            dff1['Buy'][1] = dff1['Close Price'][1]
        else:
            dff1['Sell'][1] = dff1['Close Price'][1]

        # dff1.to_csv('MACD.csv',mode = 'w',sep = '|')

        dff1.reset_index(inplace=True)

        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_width=[0.2, 0.3])

        fig = price_chart(fig, dff1, row=1)
        fig = plot_MACD(fig, dff1, row=2)
        fig.update_xaxes(
            showline=True,
            linecolor='black',
            showgrid=True)
        fig.update_yaxes(
            showline=True,
            linecolor='black',
            showgrid=True)

        fig.update_layout(title='MACD Strategy for {}'.format(selected_stock),
                          showlegend=True,
                          hovermode='x unified',
                          plot_bgcolor='white',
                          legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='v',
            x=.95,
            y=1.12,
            traceorder='normal',
            borderwidth=1)
        )

        fig1 = pnl_strategy(dff1, selected_stock)

        return message, MACD_note, fig, fig1
