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
import scipy as sc
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.offline as pyo
import plotly.graph_objs as go
import os
from plotly.subplots import make_subplots
from app import app


stock_list = pd.read_csv('data/Company_Name_List.csv')
stock_list.set_index('Symbol', inplace=True)
# The reason why Symbol is set as index is that in the next step, fetching the name of company is easy.
stock_dropdown_list = []
for symbol in stock_list.index:
    stock_dropdown_list.append(
        {'label': '{} - {}'.format(symbol, stock_list.loc[symbol]['Name']), 'value': symbol})


# Default start date to get the price quotes
start_date = dt(2020, 1, 1).date()
# Current date is the last day of price quotes.
end_date = datetime.datetime.now().date()


def get_Data(ticker_list, start, end):

    df = yf.download(ticker_list, period='1d',
                     start=start, end=end, actions=True)

    global weights
    dt = pd.to_datetime(start, format='%Y-%m-%d')
    dt1 = pd.to_datetime(end, format='%Y-%m-%d')
    num_of_days_btw_start_and_end = (dt1-dt).days
    number_of_years = num_of_days_btw_start_and_end/365
    df_adj_close = df['Adj Close']  # Selecting only Adjusted Close
    num_of_assets = len(ticker_list)
    equal_weight = (100/num_of_assets)/100
    weights = np.repeat(equal_weight, num_of_assets)

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

    # Getting Dividend Information:
    df_Dividends = df['Dividends']
    df_Dividends.reset_index(drop=True)
    dividend_recieved_by_assets = df_Dividends.sum()
    dividend_recieved_by_assets = pd.DataFrame(
        dividend_recieved_by_assets.values, dividend_recieved_by_assets.index)
    dividend_recieved_by_assets.rename(
        columns={0: 'Dividend_Recieved'}, inplace=True)
    frames = [initial_price_of_assets,
              final_price_of_assets, dividend_recieved_by_assets]
    return_summary = pd.concat(frames, axis=1, join='inner')
    return_summary['Cummilatuive Return %'] = 100 * \
        (return_summary['Final_Price'] - return_summary['Initial_Price']
         )/return_summary['Initial_Price']
    return_summary['CAGR'] = (pow(
        (return_summary['Final_Price']/return_summary['Initial_Price']), (1/number_of_years))-1)*100
    return_summary['Cummilatuive Return % with Div.'] = (
        ((return_summary['Final_Price']+return_summary['Dividend_Recieved'])/return_summary['Initial_Price'])-1)*100
    return_summary['CAGR with Div'] = (pow(
        ((return_summary['Final_Price']+return_summary['Dividend_Recieved'])/return_summary['Initial_Price']), (1/number_of_years))-1)*100

    stock_daily_return = df_adj_close.pct_change()
    mean_stock_daily_return = stock_daily_return.mean()
    covMatrix = stock_daily_return.cov()
    dividend_return = return_summary['Dividend_Recieved'] / \
        return_summary['Initial_Price']
    Mean_Daily_Dividend_Return = dividend_return/num_of_days_btw_start_and_end
    meanReturns = mean_stock_daily_return+Mean_Daily_Dividend_Return
    return_summary = return_summary.round(2)

    return meanReturns, covMatrix, return_summary, df_adj_close, stock_daily_return, weights


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(
        np.dot(weights.T, np.dot(covMatrix, weights))
    )*np.sqrt(252)
    return returns, std


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    '''We are using negative sharpe ration because minimization of negative sharpe ratio will be equal 
    to maximizing of sharpe ratio'''

    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate)/pStd


def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    '''
    constraintSet: It is the range of the weightage that a stock in the porfolio can have. (.02,1) means the minimum
    that each asset should have is 2% and maximum weightage if could have is 20%.
    numAssets is the number of assests in the portfolio
    'fun' x: np.sum(x) - 1 This equality constraint means that sum of all the portfolio weights should be equal to 1
    bounds is in the form of tuple providing the range for each assests in the portfolio
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negativeSR,  # This is the minimization function
                      weights,   # This is parameters that could be varied
                      args=args,
                      method='SLSQP',  # This is method for alogorithm
                      bounds=bounds,
                      constraints=constraints)
    return result


def Bounded_maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0.02, .25)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(negativeSR,  # This is the minimization function
                      weights,   # This is parameters that could be varied
                      args=args,
                      method='SLSQP',  # This is method for alogorithm
                      bounds=bounds,
                      constraints=constraints)
    return result


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0, 1)):
    """Minimize the portfolio variance by altering the 
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = minimize(portfolioVariance,
                      weights,
                      args=args,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result


def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]


def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0, 1)):
    """Identifying a portfolio for which has minimium variance for specific target return
       Return Target is iterated through for multiple values to get efficient frontier line
    """

    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - (returnTarget/100)},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = minimize(portfolioVariance,
                      weights,
                      args=args,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return effOpt


def random_portfolios(num_portfolios, meanReturns, covMatrix, risk_free_rate=0):
    results = np.zeros((3, num_portfolios))
    numAssets = len(meanReturns)
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(numAssets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std_dev = portfolioPerformance(
            weights, meanReturns, covMatrix)
        results[0, i] = portfolio_std_dev*100
        results[1, i] = portfolio_return*100
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def calculatedResults(meanReturns, covMatrix, riskFreeRate=0):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(
        maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_returns, maxSR_std = round(
        maxSR_returns*100, 2), round(maxSR_std*100, 2)
    maxSR_allocation = pd.DataFrame(
        maxSR_Portfolio['x'], index=meanReturns.index, columns=['Max_SR_Alloc'])
    maxSR_allocation.Max_SR_Alloc = [
        round(i*100, 0) for i in maxSR_allocation.Max_SR_Alloc]

    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(
        minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_returns, minVol_std = round(
        minVol_returns*100, 2), round(minVol_std*100, 2)
    minVol_allocation = pd.DataFrame(
        minVol_Portfolio['x'], index=meanReturns.index, columns=['Min_Vol_Alloc'])
    minVol_allocation.Min_Vol_Alloc = [
        round(i*100, 0) for i in minVol_allocation.Min_Vol_Alloc]

    # Bounded Portfolio Portfolio
    Bounded_Portfolio = Bounded_maxSR(meanReturns, covMatrix)
    Bounded_returns, Bounded_std = portfolioPerformance(
        Bounded_Portfolio['x'], meanReturns, covMatrix)
    Bounded_returns, Bounded_std = round(
        Bounded_returns*100, 2), round(Bounded_std*100, 2)
    Bounded_allocation = pd.DataFrame(
        Bounded_Portfolio['x'], index=meanReturns.index, columns=['Bounded_Alloc'])
    Bounded_allocation.Bounded_Alloc = [
        round(i*100, 0) for i in Bounded_allocation.Bounded_Alloc]

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns+1, 26)
    for target in targetReturns:
        efficientList.append(efficientOpt(
            meanReturns, covMatrix, target)['fun'])

    alloc = [maxSR_allocation, minVol_allocation, Bounded_allocation]
    allocation = pd.concat(alloc, axis=1, join='inner')

    return maxSR_returns, maxSR_std, minVol_returns, minVol_std, efficientList, targetReturns, allocation, Bounded_returns, Bounded_std


index_list = ['^GSPC']


def get_Index_Data(index_list, start, end):
    df = yf.download(index_list, period='1d', start=start, end=end)
    df.rename(columns={'Adj Close': 'S&P 500'}, inplace=True)
    # Getting only the Adj Close price of all the stock in the list for all dates
    df_adj_close = df['S&P 500']

    index_daily_returns = df_adj_close.pct_change()
    index_cummilative_daily_returns = (
        (1+index_daily_returns).cumprod() - 1) * 100

    return index_daily_returns, index_cummilative_daily_returns


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, plot_bgcolor='rgb(186, 228, 242)',
                      paper_bgcolor='rgb(186, 228, 242)')
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


portfolio_opt_LO = html.Div([

    html.H4('Portfolio Optimization with the Efficient Frontier:'),

    html.P(['Select a minimum of 5 stocks from the dropdown list below. This application will extract stock returns, including dividends, from January 2020 and construct portfolios with the following characteristics:', html.Br(),
            '1) Maximum Sharpe Ratio', html.Br(),
            '2) Minimum Volatility', html.Br(),
            '3) Maximum Sharpe ratio, subject to capital allocation constraints where a minimum of 2 percent and a maximum of 25 percent of capital can be invested in each stock.',
            html.Br(),
            'Additionally, the application will compare the performance of these portfolios against an equal weight portfolio and the S&P 500 index.']),

    html.Label(['From the Dropdown select minimum of 5 stocks for comparision:'],
               style={'font-weight': 'bold'}),

    dcc.Dropdown(id='ticker_list',
                 options=stock_dropdown_list,
                 optionHeight=35,
                 # dropdown value selected automatically when page loads
                 value=['AAPL', 'GOOG', 'PEP', 'XOM', 'SBUX'],
                 disabled=False,  # disable dropdown value selection
                 multi=True,  # allow multiple dropdown values to be selected
                 searchable=True,  # allow user-searching of dropdown values
                 # gray, default text shown when no option is selected
                 placeholder='Please select minimum of 5 stocks...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',  # activate separate CSS document in assets folder
                 ),



    html.Button('Submit', id='port_optimize',
                className='button', n_clicks=0),

    html.Div(id='message_PO'),

    dcc.Loading(children=[

        html.Div([], className='content_divider'),
        html.Br(),


        dcc.Graph(id='stock_return_graph_PO', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        html.Div(id='table_1_label'),
        html.Div(
            dash_table.DataTable(id='return_summary_PO',
                                 style_cell={'textAlign': 'center'},
                                 style_header={'backgroundColor': 'blue',
                                               'fontWeight': 'bold',
                                               'color': 'white',
                                               'fontSize': '16px',
                                               'whiteSpace': 'normal',
                                               'height': 'auto',
                                               },
                                 style_data={
                                     'color': 'black',
                                    'backgroundColor': 'white',
                                    'fontSize': '14px',
                                 }), className='table_container'),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='my_graph', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        html.Div([dcc.Graph(id='pie_chart', figure=blank_fig(), config={'displayModeBar': False})
                  ]),

        html.Div([], className='content_divider'),
        html.Br(),

        html.Div(id='table_2_label'),
        html.Div(
            dash_table.DataTable(id='allocation',
                                 style_cell={'textAlign': 'center'},
                                 style_header={'backgroundColor': 'blue',
                                               'fontWeight': 'bold',
                                               'color': 'white',
                                               'fontSize': '16px',
                                               'whiteSpace': 'normal',
                                               'height': 'auto',
                                               },
                                 style_data={
                                     'color': 'black',
                                    'backgroundColor': 'white',
                                    'fontSize': '14px',
                                 }), className='table_container'),

        html.Div([], className='content_divider'),
        html.Br(),

        dcc.Graph(id='portfolio_performance_graph',
                  figure=blank_fig(), config={'displayModeBar': False}),

        html.Div([], className='content_divider'),
        html.Br(),

        html.Div(id='table_3_label'),
        html.Div(
            dash_table.DataTable(id='portfolio_summary',
                                 style_cell={'textAlign': 'center'},
                                 style_header={'backgroundColor': 'blue',
                                               'fontWeight': 'bold',
                                               'color': 'white',
                                               'fontSize': '16px',
                                               'whiteSpace': 'normal',
                                               'height': 'auto',
                                               },
                                 style_data={
                                     'color': 'black',
                                    'backgroundColor': 'white',
                                    'fontSize': '14px',
                                 }), className='table_container'),

        html.Div([], className='content_divider'),
        html.Br(),

    ], type="circle", fullscreen=True),



], className='main_container')


@app.callback(
    [Output('message_PO', 'children'),
     Output('stock_return_graph_PO', 'figure'),
     Output('table_1_label', 'children'),
     Output('return_summary_PO', component_property='data'),
     Output('return_summary_PO', component_property='columns'),
     Output('my_graph', 'figure'),
     Output(component_id='pie_chart', component_property='figure'),

     Output('table_2_label', 'children'),
     Output('allocation', component_property='data'),
     Output('allocation', component_property='columns'),
     Output('portfolio_performance_graph', 'figure'),
     Output('table_3_label', 'children'),
     Output('portfolio_summary', component_property='data'),
     Output('portfolio_summary', component_property='columns'),

     ], [Input('port_optimize', 'n_clicks')],
    [State('ticker_list', 'value'),
     ],
    prevent_initial_call=True)
def update_graph(n_clicks, ticker_list):
    message = html.Div()
    stock_cum_ret_fig = blank_fig()
    table_1_label = html.Div()
    s_data = []
    s_columns = []
    fig = blank_fig()
    pie_fig = blank_fig()
    table_2_label = html.Div()
    a_data = []
    a_columns = []
    port_cum_ret_fig = blank_fig()
    table_3_label = html.Div()
    ps_data = []
    ps_columns = []

    if len(ticker_list) < 5:
        message = html.P('Please select minimum of 5 stocks from the dropdown list above', style={
                         'color': 'red', 'font-size': '15px'})

        return message, stock_cum_ret_fig, table_1_label, s_data, s_columns, fig, pie_fig, table_2_label, a_data, a_columns, port_cum_ret_fig, table_3_label, ps_data, ps_columns

    else:

        meanReturns, covMatrix, return_summary, df_adj_close, stock_daily_return, weights = get_Data(
            ticker_list, start_date, end_date)

        print('\n The Annual return of each stocks in portfolio from Training start date to Training End Date\n')
        print(return_summary)

        print('\n Mean Return')
        print(meanReturns)
        print('\n Covariance Martix')
        print(covMatrix)

        MaxSRresult = maxSR(meanReturns, covMatrix)
        maxSR1, maxWeights = MaxSRresult['fun'], MaxSRresult['x']
        print('\nMax SR')

        print(maxSR1, maxWeights)

        minVarResult = minimizeVariance(meanReturns, covMatrix)
        minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
        print('\nMinimum Risk')
        print(minVar, minVarWeights)

        global num_portfolios
        num_portfolios = 10000
        results, weights_rp = random_portfolios(
            num_portfolios, meanReturns, covMatrix, risk_free_rate=0)
        # Converting the results array to dataframe
        df = pd.DataFrame(results.T, columns=[
                          'STD_DEV', 'Return', 'Sharpe_Ratio'])

        global g_allocation

        maxSR_returns, maxSR_std, minVol_returns, minVol_std, efficientList, targetReturns, allocation, Bounded_returns, Bounded_std = calculatedResults(
            meanReturns, covMatrix, riskFreeRate=0)

        g_allocation = allocation

        print('\n*****Max Sharpe Ration Return and Max STD_Dev****')
        print(maxSR_returns, maxSR_std)
        print('\n*****Min Volatility Return and Min STD_Dev****')
        print(minVol_returns, minVol_std)
        print('\n***** Bounded portfolio Return and STD_Dev****')
        print(Bounded_returns, Bounded_std)
        print('\n ******** Portfolio Allocation Percentage **********\n')
        print(allocation)

        # Max Sharpe Ratio Point
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[maxSR_std],
            y=[maxSR_returns],
            marker=dict(color='red', size=10,
                        line=dict(width=2, color='black'))
        )

    # Port_Return_with_Random_Weights

        Random_Weights = go.Scatter(
            x=df['STD_DEV'],
            y=df['Return'],
            name='Port_Return_with_Random_Weights',
            mode='markers',
            marker=dict(
                color=df['Sharpe_Ratio'],
                colorscale='Viridis',
                showscale=True)
        )

    # Portfolio with ownership constraints
        bounded = go.Scatter(
            name='Constrainted Portfolio Sharpe Ratio',
            mode='markers',
            x=[Bounded_std],
            y=[Bounded_returns],
            marker=dict(color='blue', size=10,
                        line=dict(width=2, color='black'))
        )

    # Min Vol
        MinVol = go.Scatter(
            name='Mininium Volatility',
            mode='markers',
            x=[minVol_std],
            y=[minVol_returns],
            marker=dict(color='green', size=10,
                        line=dict(width=2, color='black'))
        )

    # Efficient Frontier
        EF_curve = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            x=[round(ef_std*100, 2) for ef_std in efficientList],
            y=[round(target, 2) for target in targetReturns],
            line=dict(color='black', width=2, dash='dashdot')
        )

        data = [MaxSharpeRatio, Random_Weights, MinVol, bounded, EF_curve]
        layout = go.Layout(
            title='Portfolio Optimisation with the Efficient Frontier',
            yaxis=dict(title='Annualised Return (%)',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            xaxis=dict(title='Annualised Volatility (%)',
                       showline=True,
                       linecolor='black',
                       showgrid=True,),
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                x=.75,
                y=1.15,
                traceorder='normal',
                borderwidth=1),

        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(coloraxis_colorbar_x=1.1)

        # Data Table of Return Summary
        return_summary = return_summary.reset_index(level=0)
        return_summary.rename(columns={'index': 'Stock'}, inplace=True)
        s_columns = [{'name': col, 'id': col}
                     for col in return_summary.columns]
        s_data = return_summary.to_dict(orient='records')

        table_1_label = html.Label(['Individial Stock Return Summary with and without Dividend:'],
                                   style={'font-weight': 'bold', }),

        # Getting Weigths of different Allocation
        columns1 = list(allocation.columns.values)
        weights_array = []
        for column in columns1:
            weights_array.append(allocation[column].values/100)
        weights_array.append(weights)  # Adding the equal weight

        pie_allocation = allocation
        pie_allocation_columns = pie_allocation.columns

        # Data Table of Different Allocation
        allocation = allocation.reset_index(level=0)
        allocation.rename(columns={'index': 'Stock'}, inplace=True)
        a_columns = [{'name': col, 'id': col} for col in allocation.columns]
        a_data = allocation.to_dict(orient='records')

        table_2_label = html.Label(['Capital allocation for each stock for respective Portfolio Objective:'],
                                   style={'font-weight': 'bold', }),

        stock_list = stock_daily_return.columns.values
        stock_cum_daily_returns = ((1 + stock_daily_return).cumprod() - 1)*100

        # Graph for Daily Cummilative Return
        daily_cum_ret_data = []

        for stock in stock_list:
            trace = go.Scatter(x=stock_cum_daily_returns.index.values,
                               y=stock_cum_daily_returns[stock],
                               mode='lines',
                               name=stock)
            daily_cum_ret_data.append(trace)

            cum_layout = go.Layout(
                title=dict(
                    text='Cummilative Return of Each Stock in Protfolio with Dividend', x=.05, y=.95),
                yaxis=dict(title='Cummilative Return (%)',
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
                    x=.5,
                    y=1.04,
                    traceorder='normal',
                    borderwidth=1)
            )

        stock_cum_ret_fig = go.Figure(
            data=daily_cum_ret_data, layout=cum_layout)
        portfolio_type = ['Max_SR_Alloc', 'Min_Vol_Alloc',
                          'Bounded_Alloc', 'Equal_Weight']

        index_daily_returns, index_cummilative_daily_returns = get_Index_Data(
            index_list, start_date, end_date)

        # Getting the cummilative return for different allocation
        i = -1
        for pt in portfolio_type:
            i = i+1
            stock_daily_return['PR '+pt] = np.average(
                stock_daily_return[stock_list], weights=weights_array[i], axis=1)
            stock_daily_return['CR ' +
                               pt] = ((1 + stock_daily_return['PR '+pt]).cumprod()-1)*100

        daily_cummilative_portfolio_ret = stock_daily_return[[
            'CR Max_SR_Alloc', 'CR Min_Vol_Alloc', 'CR Bounded_Alloc', 'CR Equal_Weight']]
        portfolio_daily_return = stock_daily_return[[
            'PR Max_SR_Alloc', 'PR Min_Vol_Alloc', 'PR Bounded_Alloc', 'PR Equal_Weight']]

        frames2 = [index_cummilative_daily_returns,
                   daily_cummilative_portfolio_ret]
        portfolio_graph_data = pd.concat(frames2, axis=1, join='inner')

        portfolio_graph_data.rename(columns={
            'CR Max_SR_Alloc': 'Max_SR_Alloc Portfolio',
            'CR Min_Vol_Alloc': 'Min_Vol_Alloc Portfolio',
            'CR Bounded_Alloc':  'Bounded_Alloc Portfolio',
            'CR Equal_Weight': 'Equal_Weight Portfolio'},
            inplace=True)

        # Plotting Portfolio Performance against S&P 500
        pg_data = []
        pg_columns = portfolio_graph_data.columns.values
        for column in pg_columns:
            trace = go.Scatter(x=portfolio_graph_data.index.values,
                               y=portfolio_graph_data[column],
                               mode='lines',
                               name=column)
            pg_data.append(trace)

        pg_layout = go.Layout(
            title='Portfolio Performance vs Index Return',
            yaxis=dict(title='Cummilative Return (%)',
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
                x=0.01, y=1,
                traceorder='normal',
                bgcolor='white',
                bordercolor='black',
                borderwidth=2),
        )

        port_cum_ret_fig = go.Figure(data=pg_data, layout=pg_layout)

        pie_fig = make_subplots(rows=1, cols=3,
                                specs=[
                                    [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
                                subplot_titles=[
                                    'Maximum SR allocation', 'Minimum Volatility Allocation', 'Bounded Allocation'],
                                )

        pie_fig.add_trace(
            go.Pie(

                values=pie_allocation['Max_SR_Alloc'],
                labels=pie_allocation.index,
                name="Maximum SR allocation",
                hole=.1,
            ),
            row=1, col=1
        )

        pie_fig.add_trace(
            go.Pie(

                values=pie_allocation['Min_Vol_Alloc'],
                labels=pie_allocation.index,
                name='Minimum Volatility Allocation',
                hole=.1,
            ),
            row=1, col=2
        )

        pie_fig.add_trace(
            go.Pie(

                values=pie_allocation['Bounded_Alloc'],
                labels=pie_allocation.index,
                name='Bounded Allocation',
                hole=.1,
            ),
            row=1, col=3
        )

        pie_fig.update_layout(title='Portfolio Allocation:',
                              legend=dict(y=0.01, x=.3, orientation='h'))

        # Getting the portfolio final return and Standard deviation

        portfilio_standard_dev = portfolio_graph_data.std()
        portfilio_standard_dev = portfilio_standard_dev.to_frame(
            name='Cummilative Return Std_dev in %')

        # Calculating Max Drawdown:

        frames_dr = [index_daily_returns, portfolio_daily_return]
        portfolio_daily_return_1 = pd.concat(frames_dr, axis=1, join='inner')

        def max_drawdown(portfolio_daily_return_1):
            cumulative_returns = (portfolio_daily_return_1+1).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns/peak)-1
            return drawdown.min()
        max_drawdowns = portfolio_daily_return_1.apply(
            max_drawdown, axis=0) * 100
        max_drawdowns = max_drawdowns.to_frame(name='Max Drawdown in %')

        max_drawdowns.rename(index={'PR Max_SR_Alloc': 'Max_SR_Alloc Portfolio',
                                    'PR Min_Vol_Alloc': 'Min_Vol_Alloc Portfolio',
                                    'PR Bounded_Alloc': 'Bounded_Alloc Portfolio',
                                    'PR Equal_Weight': 'Equal_Weight Portfolio'}, inplace=True)

        last_day_return = portfolio_graph_data[-1:]
        last_day_return.reset_index(inplace=True)
        last_day_return = last_day_return.drop(['Date'], axis=1)
        last_day_return = last_day_return.T
        last_day_return = last_day_return.squeeze(axis=1)
        last_day_return = last_day_return.to_frame(
            name='Cummilative Return in %')

        print('****************$$$$$$$$$$$$$$$$$$$')
        print(last_day_return)
        print(portfilio_standard_dev)
        print(max_drawdowns)

        frames_2 = [last_day_return, portfilio_standard_dev, max_drawdowns]
        portfolio_summary_comp = pd.concat(frames_2, axis=1, join='inner')
        portfolio_summary_comp = portfolio_summary_comp.round(2)
        print('portfolio_summary_comp')
        print(portfolio_summary_comp)
        portfolio_summary_comp = portfolio_summary_comp.reset_index(level=0)
        portfolio_summary_comp.rename(
            columns={'index': 'Portfolio Type'}, inplace=True)
        ps_columns = [{'name': col, 'id': col}
                      for col in portfolio_summary_comp.columns]
        ps_data = portfolio_summary_comp.to_dict(orient='records')

        table_3_label = html.Label(['Portfolio Performance Summary from Jan 2020:'],
                                   style={'font-weight': 'bold', }),

        return message, stock_cum_ret_fig, table_1_label, s_data, s_columns, fig, pie_fig, table_2_label, a_data, a_columns, port_cum_ret_fig, table_3_label, ps_data, ps_columns
