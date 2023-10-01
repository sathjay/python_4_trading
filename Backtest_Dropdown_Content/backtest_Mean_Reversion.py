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


def SMA(df, mean_days, column='Close Price'):
    return df[column].rolling(window=mean_days).mean()


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
        size=11,
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
        size=11,
        color='red',
        symbol='triangle-down',
        line={'width': 2})),
        row=row,
        col=column)

    fig.update_yaxes(title_text='Price and Signal', row=row, col=column)

    return fig


def plot_Mean_Reversion(fig, df, lower_limit, lower_percentile, upper_limit, upper_percentile, row, column=1):

    fig.add_trace(go.Scatter(x=df['Date'],
                             y=df['Ratio'],
                             mode='lines',
                             name='Ratio'),
                  row=row,
                  col=column)

    fig.add_hline(y=lower_limit, line_width=3,
                  line_dash="solid", line_color="green",
                  annotation_text="{} percentile".format(lower_percentile),
                  annotation_font_size=15, annotation_font_color="black",
                  row=row,
                  col=column)

    fig.add_hline(y=upper_limit, line_width=3,
                  line_dash="solid", line_color="red",
                  annotation_text="{} percentile".format(upper_percentile),
                  annotation_font_size=15, annotation_font_color="black",
                  row=row,
                  col=column)

    fig.update_yaxes(title_text='Mean Rev. Ratio', row=row, col=column)

    return fig


def pnl_strategy(df, selected_stock):

    payoff_data = []

    buy_hold = go.Scatter(x=df['Date'],
                          y=df['BH_Cum_Return'],
                          mode='lines',
                          name='Buy and Hold')
    payoff_data.append(buy_hold)

    mean_reversion_ret = go.Scatter(x=df['Date'],
                                    y=df['Strategy_Cum_Return'],
                                    mode='lines',
                                    name='Mean Reversion')
    payoff_data.append(mean_reversion_ret)

    payoff_data_layout = go.Layout(
        title='Buy and Hold vs. Mean Reversion Strategy pay off for {}'.format(
            selected_stock),
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
            orientation='v',
            x=.9,
            y=1.15,
            traceorder='normal',
            borderwidth=1)
    )

    fig = go.Figure(data=payoff_data, layout=payoff_data_layout)

    return fig


MR_LO = html.Div([

    html.H5('Mean Reversion Trading Strategy:', className='content_title'),

    html.P("Mean reversion is a financial theory suggesting that asset prices and historical returns will, over time, revert to their long-term average or mean. This concept is rooted in the belief that short-term fluctuations are temporary and that assets tend to stabilize around their historical averages. The mean reversion strategy involves: "),
    html.Li("Buying: Investing in assets that have performed worse than their historical average, under the assumption that they will increase in value."),

    html.Li("Selling: Offloading assets that have performed better than their historical average, predicting that they will decrease in value."),

    html.P("In essence, mean reversion traders aim to capitalize on extreme price deviations from historical norms. They anticipate that prices will bounce back (or revert) to their historical averages or means. It's important to note that while mean reversion can be a powerful trading strategy, no approach is foolproof. Prices can and do stay at extreme levels for extended periods, and other external factors can influence market dynamics."),



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

        html.Label(['Select the Mean rolling window days:'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=10,
                   max=40,
                   id='mean_days',
                   step=1,
                   value=20,
                   marks={
                       10: '10',
                       40: '40'
                   },
                   tooltip={"placement": "bottom", "always_visible": True},
                   className='days_slider'),

    ], className='slider_container'),

    html.Div([

        html.Label(['Select the Upper percentile for Mean Return:'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=60,
                   max=95,
                   id='upper_percentile',
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

        html.Label(['Select the Lower percentile for Mean Return:'],
                   style={'font-weight': 'bold'}),
        dcc.Slider(min=5,
                   max=40,
                   id='lower_percentile',
                   step=1,
                   value=15,
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

    html.Button('Submit', id='BT_MR',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),

    dcc.Loading(children=[

        html.Label(id='Selected_Value_Mean_Reversion'),

        html.Div(id='MR_note', children=[]),

        html.Br(),
        dcc.Graph(id='mean_revertion_plot', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Br(),
        dcc.Graph(id='payoff_plot_MR', figure=blank_fig(),
                  config={'displayModeBar': False}),

    ], type="circle", fullscreen=True),
])


@app.callback([Output('Selected_Value_Mean_Reversion', 'children'),
               Output('MR_note', 'children'),
               Output('mean_revertion_plot', component_property='figure'),
               Output('payoff_plot_MR', component_property='figure')],
              [Input('BT_MR', 'n_clicks')],
              [State('selected_stock', 'value'),
              State('mean_days', 'value'),
              State('upper_percentile', 'value'),
              State('lower_percentile', 'value'),
              State('backtest_days', 'value')],
              prevent_initial_call=True
              )
def mean_reversion_strategy(n_clicks, selected_stock, mean_days, upper_percentile, lower_percentile, backtest_days):

    mean_days = int(mean_days)
    upper_percentile = int(upper_percentile)
    lower_percentile = int(lower_percentile)

    message = html.P("You have selected {} Moving Average Days. {} is the upper percentile limit and  \
                    {} is the lower percentile limit".format(mean_days, upper_percentile, lower_percentile))

    MR_note = html.P(["Assumption: In this mean reversion strategy, a sell signal is generated when the Close price to Moving Average Ratio crosses the upper percentile. Conversely, a buy signal is generated when the Close price to Moving Average Ratio falls below the lower percentile."], className='note'
                     )

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)
    day = np.arange(1, len(df)+1)
    df['Day'] = day
    dff = df[['Day', 'Adj Close']]
    dff.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

    if backtest_days == 'All':
        backtest_days = len(dff)
    else:
        backtest_days = int(backtest_days)

    dff1 = dff.iloc[-backtest_days:]

    dff1['SMA'] = SMA(dff1, mean_days)
    dff1['Ratio'] = dff1['Close Price']/dff1['SMA']

    percentiles = [5, 10, 15, 20, 25, 30, 35,
                   40, 60, 65, 70, 75, 80, 85, 90, 95]

    dff1.dropna(inplace=True)

    ratios = dff1['Ratio']
    # Getting the mean ratio for corresponding percentile
    percentiles_value = np.percentile(ratios, percentiles)

    # getting the buy and sell limit based on user input
    upper_limit = percentiles_value[percentiles.index(upper_percentile)]
    lower_limit = percentiles_value[percentiles.index(lower_percentile)]

    # Long and Short Position
    dff1['Position'] = np.where(dff1['Ratio'] > upper_limit, -1, np.nan)
    dff1['Position'] = np.where(
        dff1['Ratio'] < lower_limit, 1, dff1['Position'])
    dff1['Position'].ffill(inplace=True)
    dff1['Position'] = dff1['Position'].fillna(0)

    # Long / shot position
    dff1['Buy_Sell'] = dff1['Position'].diff()
    dff1['Buy_Sell'] = dff1['Buy_Sell'].fillna(0)

    # Getting the buy and sell price

    dff1['Buy_price'] = [dff1['Close Price'][i] if ((x == 1) or (
        x == 2)) else np.nan for i, x in enumerate(dff1['Buy_Sell'])]
    dff1['Sell_price'] = [dff1['Close Price'][i] if (
        (x == -1) or (x == -2)) else np.nan for i, x in enumerate(dff1['Buy_Sell'])]

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

    # dff1.to_csv('Mean_Reversion_Analysis1 after Buy_Sell.csv',mode = 'w',sep = '|')
    dff1.reset_index(inplace=True)

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_width=[0.2, 0.3])

    fig = price_chart(fig, df_price_chart, row=1)
    fig = plot_Mean_Reversion(fig, df_price_chart, lower_limit,
                              lower_percentile, upper_limit, upper_percentile, row=2)

    fig.update_xaxes(
        showline=True,
        linecolor='black',
        showgrid=True)
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        showgrid=True)

    fig.update_layout(

        title='Mean Reversion Strategy for {}'.format(selected_stock),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='v',
            x=.96,
                        y=1.12,
            traceorder='normal',
            borderwidth=1))

    fig1 = pnl_strategy(df_pay_off_chart, selected_stock)

    return message, MR_note, fig, fig1
