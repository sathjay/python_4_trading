import pandas as pd
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

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Close Price'],
                             mode='lines',
                             name='Price'),
                  row=row,
                  col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Buy_price'],
                             mode='markers',
                             name='Buy',
                             marker=dict(
        size=10,
        color='green',
        symbol='triangle-up',
        line={'width': 2})),
        row=row,
        col=column)

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Sell_price'],
                             mode='markers',
                             name='Sell',
                             marker=dict(
        size=10,
        color='red',
        symbol='triangle-down',
        line={'width': 2})),
        row=row,
        col=column)

    fig.update_yaxes(title_text='Price and Signal', row=row, col=column)

    return fig


def plot_MFI(fig, df, lower_MFI, upper_MFI, row, column=1):

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Money_Flow_Index'],
                             mode='lines',
                             name='MFI'),
                  row=row,
                  col=column)

    fig.add_hline(y=lower_MFI, line_width=3,
                  line_dash="solid", line_color="green",
                  annotation_text="{} lower MFI ".format(lower_MFI),
                  annotation_font_size=20, annotation_font_color="black",
                  row=row,
                  col=column)

    fig.add_hline(y=upper_MFI, line_width=3,
                  line_dash="solid", line_color="red",
                  annotation_text="{} upper MFI".format(upper_MFI),
                  annotation_font_size=20, annotation_font_color="black",
                  row=row,
                  col=column)

    fig.update_yaxes(title_text='Money Flow Index', row=row, col=column)

    return fig


def pnl_strategy(df, selected_stock):

    payoff_data = []

    buy_hold = go.Scatter(x=df['Date'],
                          y=df['BH_Cum_Return'],
                          mode='lines',
                          name='Buy and Hold')
    payoff_data.append(buy_hold)

    MFI_ret = go.Scatter(x=df['Date'],
                         y=df['Strategy_Cum_Return'],
                         mode='lines',
                         name='MFI')
    payoff_data.append(MFI_ret)

    payoff_data_layout = go.Layout(
        title='Buy and Hold vs. MFI for {}'.format(selected_stock),
        yaxis=dict(title='Cummilative Return',
                   showline=True,
                   linecolor='black',
                   showgrid=True),
        xaxis=dict(title='Date',
                   showline=True,
                   linecolor='black',
                   showgrid=True),
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

    fig = go.Figure(data=payoff_data, layout=payoff_data_layout)

    return fig


MFI_LO = html.Div([

    html.H5('Money Flow Index Strategy:', className='content_title'),

    html.P(['The Money Flow Index (MFI) is a technical oscillator that uses price and volume data for identifying overbought or oversold signals in an asset. \
                           It can also be used to spot divergences which warn of a trend change in price. The oscillator moves between 0 and 100.',
            html.Br(),
            'Unlike conventional oscillators such as the Relative Strength Index (RSI), the Money Flow Index incorporates both price and volume data,\
                             as opposed to just price. For this reason, some analysts call MFI the volume-weighted RSI.',
            html.Br(),
            'An MFI reading above 80 is considered overbought and an MFI reading below 20 is considered oversold, although levels of 90 and 10 are also used as thresholds.',
            html.Br(),
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
        html.Label(['Select the Money Flow Index time window (days):'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=40,
                   id='MFI_Period',
                   step=1,
                   value=14,
                   marks={
                       5: '5',
                       40: '40'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),

    html.Div([

        html.Label(['Select the MFI Upper value'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=60,
                   max=95,
                   id='upper_MFI',
                   step=1,
                   value=80,
                   marks={
                       60: '60',
                       95: '95'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),

    html.Div([

        html.Label(['Select the MFI Lower value'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=40,
                   id='lower_MFI',
                   step=1,
                   value=20,
                   marks={
                       5: '5',
                       40: '40'
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

    html.Button('Submit', id='BT_MFI',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),

    dcc.Loading(children=[

        html.Label(id='Selected_Value_MFI'),

        html.Div(id='MFI_note', children=[]),

        html.Br(),
        dcc.Graph(id='entry_and_exit_MFI', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Br(),
        dcc.Graph(id='pnl_chart_MFI', figure=blank_fig(),
                  config={'displayModeBar': False}),
        html.Br(),

    ], type="circle", fullscreen=True),

])


@app.callback(
    [Output('Selected_Value_MFI', 'children'),
     Output('MFI_note', 'children'),
     Output('entry_and_exit_MFI', component_property='figure'),
     Output('pnl_chart_MFI', component_property='figure')],
    [Input('BT_MFI', 'n_clicks')],
    [
        State('selected_stock', 'value'),
        State('MFI_Period', 'value'),
        State('upper_MFI', 'value'),
        State('lower_MFI', 'value'),
        State('backtest_days', 'value'),
    ],
    prevent_initial_call=True
)
def MFI_update(n_clicks, selected_stock, MFI_Period, upper_MFI, lower_MFI, backtest_days):

    MFI_Period = int(MFI_Period)
    upper_MFI = int(upper_MFI)
    lower_MFI = int(lower_MFI)

    message = html.P("You have selected {} MFI period. {} is the upper MFI limit and  \
                    {} is the lower MFI limit".format(MFI_Period, upper_MFI, lower_MFI))

    MFI_note = html.P(["Assumption: In this MFI strategy, a sell signal is generated when the MFI value crosses the upper limit. Conversely, a buy signal is generated when the MFI value falls below the lower percentile."], className='note'
                      )

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)
    if backtest_days == 'All':
        backtest_days = len(df)
    else:
        backtest_days = int(backtest_days)

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)

    df.rename(columns={'Adj Close': 'Close Price'}, inplace=True)
    dff = df[['High', 'Low', 'Close Price', 'Volume']]
    dff['Typical_Price'] = (dff['High']+dff['Low']+dff['Close Price'])/3
    dff['Money_flow'] = dff['Typical_Price']*dff['Volume']

    # Get positive and negative money flow

    positive_flow = []
    negative_flow = []
    positive_flow.append(np.nan)  # First row value cannot be calculated
    negative_flow.append(np.nan)

    for i in range(1, len(dff['Typical_Price'])):
        if dff['Typical_Price'][i] > dff['Typical_Price'][i-1]:
            positive_flow.append(dff['Money_flow'][i-1])
            negative_flow.append(0)
        elif dff['Typical_Price'][i] < dff['Typical_Price'][i-1]:
            negative_flow.append(dff['Money_flow'][i-1])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    dff['Positive_flow'] = positive_flow
    dff['Negative_flow'] = negative_flow

    # Money flow Ratio

    dff['MFR'] = dff['Positive_flow'].rolling(window=MFI_Period, center=False).sum(
    )/dff['Negative_flow'].rolling(window=MFI_Period, center=False).sum()

    dff['Money_Flow_Index'] = 100 - 100/(1+dff['MFR'])

    dff.dropna(inplace=True)

    dff1 = dff.iloc[-backtest_days:]

    # Long and Short Position
    dff1['Position'] = np.where(
        dff1['Money_Flow_Index'] > upper_MFI, -1, np.nan)
    dff1['Position'] = np.where(
        dff1['Money_Flow_Index'] < lower_MFI, 1, dff1['Position'])
    dff1['Position'].ffill(inplace=True)
    dff1['Position'] = dff1['Position'].fillna(0)

    # Long / short signal
    dff1['Buy_Sell'] = dff1['Position'].diff()
    dff1['Buy_Sell'] = dff1['Buy_Sell'].fillna(0)

    # Getting the buy and sell price

    dff1['Buy_price'] = [dff1['Close Price'][i] if ((x == 1) or (
        x == 2)) else np.nan for i, x in enumerate(dff1['Buy_Sell'])]
    dff1['Sell_price'] = [dff1['Close Price'][i] if (
        (x == -1) or (x == -2)) else np.nan for i, x in enumerate(dff1['Buy_Sell'])]

    # dff1.to_csv('7_MFI.csv',mode = 'w',sep = '|')

    # Getting values for slicing DF for display and payoff calculation
    dff1.reset_index(inplace=True)
    first_valid_buy_position = dff1['Buy_price'].first_valid_index()
    first_valid_sell_position = dff1['Sell_price'].first_valid_index()

    if first_valid_buy_position < first_valid_sell_position:
        op_filter = first_valid_buy_position
    else:
        op_filter = first_valid_sell_position

    if op_filter > 10:
        start = op_filter-10
    else:
        start = op_filter

    df_pay_off_chart = dff1.iloc[op_filter:]
    df_price_chart = dff1.iloc[start:]

    # Strategy Return
    df_pay_off_chart['Return'] = np.log(
        df_pay_off_chart['Close Price']).diff().shift(-1)
    df_pay_off_chart['Strategy_Returns'] = df_pay_off_chart['Return'] * \
        dff1['Position']

    # Buy and Hold Return
    df_pay_off_chart['BH_Cum_Return'] = np.exp(
        df_pay_off_chart['Return']).cumprod()
    df_pay_off_chart['Strategy_Cum_Return'] = np.exp(
        df_pay_off_chart['Strategy_Returns']).cumprod()

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

    fig = price_chart(fig, df_price_chart, row=1)
    fig = plot_MFI(fig, df_price_chart, lower_MFI, upper_MFI, row=2)

    fig.update_layout(

        title='Stochastic MFI for {}'.format(selected_stock),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='v',
            x=.92,
            y=1.15,
            traceorder='normal',
            borderwidth=1)
    )

    fig1 = pnl_strategy(df_pay_off_chart, selected_stock)

    return message, MFI_note, fig, fig1
