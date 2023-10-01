import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import numpy as np
import yfinance as yf
from datetime import datetime as dt
import datetime
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from app import app

SP500 = pd.read_csv('data/Company_Name_List.csv')
SP500.set_index('Symbol', inplace=True)
# The reason why Symbol is set as index is that in the next step, fetching the name of company is easy.
stock_dropdown_list = []
for symbol in SP500.index:
    stock_dropdown_list.append(
        {'label': '{} - {}'.format(symbol, SP500.loc[symbol]['Name']), 'value': symbol})

# Default start date to get the price quotes
start_date = dt(2018, 1, 1).date()
# Current date is the last day of price quotes.
end_date = datetime.datetime.now().date()


def get_Data(ticker_list, start, end):

    df = yf.download(ticker_list, period='1d', start=start, end=end)

    dt = pd.to_datetime(start, format='%Y-%m-%d')
    dt1 = pd.to_datetime(end, format='%Y-%m-%d')
    num_of_days_btw_start_and_end = (dt1-dt).days
    number_of_years = num_of_days_btw_start_and_end/365
    df_adj_close = df['Adj Close']  # Selecting only Adjusted Close
    # The above df_adj_close more columns are added to calculate cummilative return.
    df_close = df['Adj Close']

    # Selecting the first row and last row which are initial price and final price
    initial_price_of_assets = df_adj_close.iloc[[0]]
    initial_price_of_assets.reset_index(inplace=True)
    initial_price_of_assets = initial_price_of_assets.drop(['Date'], axis=1)
    initial_price_of_assets = initial_price_of_assets.T
    initial_price_of_assets.rename(columns={0: 'Initial_Price'}, inplace=True)
    final_price_of_assets = df_adj_close.iloc[[-1]]
    final_price_of_assets.reset_index(inplace=True)
    final_price_of_assets = final_price_of_assets.drop(['Date'], axis=1)
    final_price_of_assets = final_price_of_assets.T
    final_price_of_assets.rename(columns={0: 'Final_Price'}, inplace=True)

    frames = [initial_price_of_assets, final_price_of_assets]
    return_summary = pd.concat(frames, axis=1, join='inner')
    return_summary['Cummilatuive Return %'] = 100 * \
        (return_summary['Final_Price'] - return_summary['Initial_Price']
         )/return_summary['Initial_Price']
    return_summary['CAGR'] = (pow(
        (return_summary['Final_Price']/return_summary['Initial_Price']), (1/number_of_years))-1)*100

    stock_daily_return = df_adj_close.pct_change()

    a = stock_daily_return.mean()*100
    b = stock_daily_return.std(axis=0)*100
    a = a.to_frame(name='Daily Mean Return in %')
    b = b.to_frame(name='Daily Return Std Dev in %')
    return_summary = pd.concat([return_summary, a, b], axis=1, join='inner')
    return_summary = return_summary.round(2)

    # Caluculating the Daily Percentage change in the df_adj_close datafram
    for i in df_adj_close.columns:
        df_adj_close[i+'_DPR'] = df_adj_close[i].pct_change()
        df_adj_close['CR_'+i] = ((1 + df_adj_close[i+'_DPR']).cumprod()-1)*100

    # Converting Daily Percentage change in 100

    cols = [col for col in df_adj_close.columns if '_DPR' in col]
    for col in cols:
        df_adj_close[col] = df_adj_close[col]*100
    df_adj_close = df_adj_close.round(3)

    # Getting columns that start with CR_
    stock_cummilative_return = df_adj_close[df_adj_close.columns[pd.Series(
        df_adj_close.columns).str.startswith('CR_')]]
    stock_cummilative_return.rename(
        columns=lambda x: x.strip('CR_'), inplace=True)
    # Calculating Annualized Standard Deviation of stocks
    log_stock_daily_returns = np.log(df_close/df_close.shift(1)).dropna()
    daily_std = log_stock_daily_returns.std()
    annualized_std = daily_std * np.sqrt(252)*100
    annualized_std = annualized_std.round(2)

    # Another formula for standard deviation of each stock.
    # df_close.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))*100

    # Combining Return with Standard Deviation
    annualized_std = annualized_std.to_frame(name='Annualized StdDev in %')
    frames_2 = [return_summary, annualized_std]
    return_summary = pd.concat(frames_2, axis=1, join='inner')

    return return_summary, df_close, stock_daily_return, stock_cummilative_return, annualized_std, log_stock_daily_returns


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, plot_bgcolor='rgb(186, 228, 242)',
                      paper_bgcolor='rgb(186, 228, 242)')
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


stock_perf_comp_LO = html.Div([



    html.H4('Compare Performance of Four Stock Returns:'),

    html.P('For the four stocks selected from the dropdown below, users can compare metrics including: daily return distribution, \
        60-day rolling volatility, 60-day rolling Sharpe ratio, 60-day Sortino volatility, and 60-day rolling Sortino ratio.'),

    html.P('The Sharpe ratio is a measure of risk-adjusted return. It is the average return earned in excess of the risk-free rate per unit of volatility. On the other hand, Sortino volatility measures the volatility of negative asset returns and the Sortino ratio is a variation of the Sharpe ratio that only factors in downside risk. This is particularly useful when investors are more concerned about potential losses than gains.'),

    html.Label(['From the Dropdown Select 4 stocks for comparision:'],
               style={'font-weight': 'bold'}),

    dcc.Dropdown(id='my_ticker_symbol',
                 options=stock_dropdown_list,
                 optionHeight=35,
                 # dropdown value selected automatically when page loads
                 value=['AAPL', 'GOOG', 'PEP', 'XOM'],
                 disabled=False,  # disable dropdown value selection
                 multi=True,  # allow multiple dropdown values to be selected
                 searchable=True,  # allow user-searching of dropdown values
                 # gray, default text shown when no option is selected
                 placeholder='Please select 4 stocks...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',  # activate separate CSS document in assets folder
                 ),


    html.Button('Submit', id='stock_perf_comp',
                className='button', n_clicks=0),

    html.Div([], className='content_divider'),
    html.Br(),

    dcc.Loading(children=[

        html.H5(id='output', className='content_title'),


        dcc.Graph(id='return_distribution', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='stock_return_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='rolling_volatility_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='sharpe_ratio_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='sortino_vol_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='sortino_ratio_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

    ], type="circle", fullscreen=True),


], className='main_container')


@app.callback([Output('output', 'children'),

              Output('return_distribution', 'figure'),
              Output('stock_return_graph', 'figure'),
              Output('rolling_volatility_graph', 'figure'),
              Output('sharpe_ratio_graph', 'figure'),
              Output('sortino_vol_graph', 'figure'),
              Output('sortino_ratio_graph', 'figure')
               ], [Input('stock_perf_comp', 'n_clicks')],
              [State('my_ticker_symbol', 'value')],
              prevent_initial_call=True
              )
def process_selected(n_clicks, ticker_list):
    # Defining these empty return objects, so that when the page loads for the first time no half filled graphs are displayed.
    message = []
    fig = blank_fig()
    stock_cum_ret_fig = blank_fig()
    rolling_vol_fig = blank_fig()
    rolling_SR_fig = blank_fig()
    rolling_sortino_vol_fig = blank_fig()
    rolling_sortino_ratio_fig = blank_fig()

    if len(ticker_list) != 4:
        message = html.P('Please select 4 stocks from the dropdown list above', style={
                         'color': 'red', 'font-size': '15px'})

        return message, fig, stock_cum_ret_fig, rolling_vol_fig, rolling_SR_fig, rolling_sortino_vol_fig, rolling_sortino_ratio_fig

    else:
        message = ['You selected ' +
                   ', '.join(str(stock) for stock in ticker_list)]
        return_summary, df_close, stock_daily_return, stock_cummilative_return, annualized_std, log_stock_daily_returns = get_Data(
            ticker_list, start_date, end_date)

        return_summary = return_summary.reset_index(level=0)
        return_summary.rename(columns={'index': 'Stock'}, inplace=True)

        fig = make_subplots(rows=2, cols=2)
        traces = []
        stock_list = log_stock_daily_returns.columns.values
        for stock in stock_list:
            trace = go.Histogram(
                x=log_stock_daily_returns[stock]*100, name=stock)
            traces.append(trace)

        fig.append_trace(traces[0], 1, 1)
        fig.append_trace(traces[1], 1, 2)
        fig.append_trace(traces[2], 2, 1)
        fig.append_trace(traces[3], 2, 2)

        fig.update_layout(autosize=False,
                          title=dict(
                              text='Frequency of Daily log returns in %'),

                          plot_bgcolor='white',
                          xaxis=dict(showline=True, linecolor='black', showgrid=True,
                                     title=str(stock_list[0])+' Annualized Volatility in %: ' + str(annualized_std.iloc[0]['Annualized StdDev in %'])),
                          xaxis2=dict(showline=True, linecolor='black', showgrid=True,
                                      title=str(stock_list[1])+' Annualized Volatility in %: ' + str(annualized_std.iloc[1]['Annualized StdDev in %'])),
                          xaxis3=dict(showline=True, linecolor='black', showgrid=True,
                                      title=str(stock_list[2])+' Annualized Volatility in %: ' + str(annualized_std.iloc[2]['Annualized StdDev in %'])),
                          xaxis4=dict(showline=True, linecolor='black', showgrid=True,
                                      title=str(stock_list[3])+' Annualized Volatility in %: ' + str(annualized_std.iloc[3]['Annualized StdDev in %'])),

                          yaxis=dict(showline=True,
                                     linecolor='black', showgrid=True),
                          yaxis2=dict(showline=True,
                                      linecolor='black', showgrid=True),
                          yaxis3=dict(showline=True,
                                      linecolor='black', showgrid=True),
                          yaxis4=dict(showline=True,
                                      linecolor='black', showgrid=True)


                          )

        # Cummilative return plotting

        daily_cum_ret_data = []
        for stock in stock_list:

            trace = go.Scatter(x=stock_cummilative_return.index.values,
                               y=stock_cummilative_return[stock],
                               mode='lines',
                               name=stock)
            daily_cum_ret_data.append(trace)

        cum_layout = go.Layout(
            title=dict(text='Cummilative Return of Each Stock', x=.05, y=.95),
            yaxis=dict(title='Cummilative Return (%)',
                       showline=True,
                       linecolor='black',
                       showgrid=True,
                       ),
            xaxis=dict(title='Date',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                orientation='h',
                x=.6,
                y=1.12,
                traceorder='normal',
                borderwidth=1)
        )

        stock_cum_ret_fig = go.Figure(
            data=daily_cum_ret_data, layout=cum_layout)

        # Calculating 60 day rolling volatility:

        TRADING_DAYS = 60
        volatility = log_stock_daily_returns.rolling(
            window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)*100
        volatility.tail()
        rolling_vol_data = []

        for stock in stock_list:
            trace = go.Scatter(x=volatility.index.values,
                               y=volatility[stock],
                               mode='lines',
                               name=stock)
            rolling_vol_data.append(trace)

        vol_layout = go.Layout(
            title=str(TRADING_DAYS) + ' Day Rolling Volatility Data',
            yaxis=dict(title='Volatility (%)',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            xaxis=dict(title='Date',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                orientation='h',
                x=.6,
                y=1.12,
                traceorder='normal',
                borderwidth=1)
        )

        rolling_vol_fig = go.Figure(data=rolling_vol_data, layout=vol_layout)

        # 60 Day Rolling Sharpe ratio

        risk_free_rate = 4
        Rf = (risk_free_rate/100)/255
        sharpe_ratio = (log_stock_daily_returns.rolling(
            window=TRADING_DAYS).mean() - Rf)*TRADING_DAYS / volatility*100

        rolling_SR_data = []

        for stock in stock_list:
            trace = go.Scatter(x=sharpe_ratio.index.values,
                               y=sharpe_ratio[stock],
                               mode='lines',
                               name=stock)
            rolling_SR_data.append(trace)

        rolling_SR_layout = go.Layout(
            title=str(TRADING_DAYS) + ' Day Rolling Sharpe Ratio',
            yaxis=dict(title='Rolling Sharpe Ratio',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            xaxis=dict(title='Date',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                orientation='h',
                x=.6,
                y=1.12,
                traceorder='normal',
                borderwidth=1)
        )

        rolling_SR_fig = go.Figure(
            data=rolling_SR_data, layout=rolling_SR_layout)

        # 60 day rolling Sortino volatility

        sortino_vol = log_stock_daily_returns[log_stock_daily_returns < 0].rolling(
            window=TRADING_DAYS, center=True, min_periods=10).std()*np.sqrt(TRADING_DAYS)*100
        sortino_ratio = (log_stock_daily_returns.rolling(
            window=TRADING_DAYS).mean() - Rf)*TRADING_DAYS / sortino_vol*100

        rolling_sortino_vol_data = []

        for stock in stock_list:
            trace = go.Scatter(x=sortino_vol.index.values,
                               y=sortino_vol[stock],
                               mode='lines',
                               name=stock)
            rolling_sortino_vol_data.append(trace)

        rolling_sortino_vol_layout = go.Layout(
            title=str(TRADING_DAYS) + ' Day Rolling Sortino Volatility',
            yaxis=dict(title='Sortino Volatility (%)',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            xaxis=dict(title='Date',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                orientation='h',
                x=.6,
                y=1.12,
                traceorder='normal',
                borderwidth=1)
        )

        rolling_sortino_vol_fig = go.Figure(
            data=rolling_sortino_vol_data, layout=rolling_sortino_vol_layout)

        # 60 day rolling Sortino Ratio

        rolling_sortino_ratio_data = []

        for stock in stock_list:
            trace = go.Scatter(x=sortino_ratio.index.values,
                               y=sortino_ratio[stock],
                               mode='lines',
                               name=stock)
            rolling_sortino_ratio_data.append(trace)

        rolling_sortino_ratio_layout = go.Layout(
            title=str(TRADING_DAYS) + ' Day Rolling Sortino Ratio',
            yaxis=dict(title='Sortino Ratio',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            xaxis=dict(title='Date',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                orientation='h',
                x=.6,
                y=1.12,
                traceorder='normal',
                borderwidth=1)
        )

        rolling_sortino_ratio_fig = go.Figure(
            data=rolling_sortino_ratio_data, layout=rolling_sortino_ratio_layout)

        return message, fig, stock_cum_ret_fig, rolling_vol_fig, rolling_SR_fig, rolling_sortino_vol_fig, rolling_sortino_ratio_fig
