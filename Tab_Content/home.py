import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from app import app
card_height = '10vh'

home_LO = html.Div([
    html.Div([

        html.H4("Welcome to Python4Trading.com:"),
        html.P("I created this site to showcase how an interactive web analytics application can be constructed using Python and its associated packages, including Dash, Plotly, and Scikit-learn. Navigate the tabs above to delve deeper and explore the various features."),
        html.P("For the Python code used in these demos, as well as other valuable Python resources, please refer to the 'About Me' section."),
        html.P(['Disclaimer: The financial data presented is sourced from the "yfinance" Python package, which is freely available. Please note that the reliability and accuracy of this data cannot be guaranteed. All analysis results are intended for educational purposes only and should not be construed as investment advice.'], className='note')

    ], className='intro_section'),

    html.Div([], className='content_divider'),

    html.H5("Popular Stocks- Price Dashboard:", id='dashboards_title'),

    dcc.Loading(children=[
        html.Div([

            html.Div([

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('AMZN.png'),
                             className='logo'),
                        html.H5(['Amazon'], className='card_heading'),

                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph1', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([
                    html.Div([
                        html.Img(src=app.get_asset_url('AMD.png'),
                             className='logo'),
                        html.H5(['AMD'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph2', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([
                    html.Div([
                        html.Img(src=app.get_asset_url('ADBE.png'),
                             className='logo'),
                        html.H5(['Adobe'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph3', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height, })
                ], className='card'),

            ], className='summary_board'),


            html.Div([

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('Tesla.png'),
                             className='logo'),
                        html.H5(['Tesla'], className='card_heading'),

                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph4', figure={},
                              config={'displayModeBar': False},
                              style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('NVDA.png'),
                             className='logo'),
                        html.H5(['Nvidia'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph5', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('Apple.png'),
                             className='logo'),
                        html.H5(['Apple'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph6', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height, })
                ], className='card'),

            ], className='summary_board'),

            html.Div([

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('MSFT.png'),
                             className='logo'),
                        html.H5(['Microsoft'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph7', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([

                    html.Div([
                        html.Img(src=app.get_asset_url('META.png'),
                             className='logo'),
                        html.H5(['Meta'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph8', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height})
                ], className='card'),

                html.Div([
                    html.Div([
                        html.Img(src=app.get_asset_url('Google.png'),
                             className='logo'),
                        html.H5(['Alphabet'], className='card_heading'),
                    ], className='card_head_row'),

                    dcc.Graph(id='indicator-graph9', figure={},
                              config={
                        'displayModeBar': False},
                        style={'width': '85%', 'height': card_height, })
                ], className='card'),


            ], className='summary_board'),

        ], className='market_summary'),

    ], type="circle", fullscreen=False),



    dcc.Interval(id='update', n_intervals=0, interval=1000*180)



], className='main_container')


@app.callback([
    Output('indicator-graph1', 'figure'),
    Output('indicator-graph2', 'figure'),
    Output('indicator-graph3', 'figure'),
    Output('indicator-graph4', 'figure'),
    Output('indicator-graph5', 'figure'),
    Output('indicator-graph6', 'figure'),
    Output('indicator-graph7', 'figure'),
    Output('indicator-graph8', 'figure'),
    Output('indicator-graph9', 'figure'),],
    [Input('update', 'n_intervals')])
def update_indicator(timer):

    ticker_list = ['AMZN', 'AMD', 'ADBE', 'TSLA',
                   'NVDA', 'AAPL', 'MSFT', 'META', 'GOOG']

    df = yf.download(ticker_list, period='2d')
    df1 = df['Adj Close']
    AMZN = df1[['AMZN']]
    AMD = df1[['AMD']]
    ADBE = df1[['ADBE']]
    TSLA = df1[['TSLA']]
    NVDA = df1[['NVDA']]
    AAPL = df1[['AAPL']]
    MSFT = df1[['MSFT']]
    META = df1[['META']]
    GOOG = df1[['GOOG']]

    # Amazon
    close = AMZN['AMZN'].iloc[-1]
    yes_close = AMZN['AMZN'].iloc[-2]

    delta_font = {'size': 22}
    number_font = {'size': 28}

    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig.update_traces(delta_font=delta_font, number_font=number_font)
    fig.update_layout(margin=dict(t=18, r=0, l=40, b=8))

    if close >= yes_close:
        fig.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig.update_traces(delta_decreasing_color='red')

    # Fig1 for Nasdaq
    close = AMD['AMD'].iloc[-1]
    yes_close = AMD['AMD'].iloc[-2]

    fig1 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig1.update_traces(delta_font=delta_font, number_font=number_font)
    fig1.update_layout(margin=dict(t=18, r=0, l=40, b=0))

    if close >= yes_close:
        fig1.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig1.update_traces(delta_decreasing_color='red')

    # Fig2 for FTSE
    close = ADBE['ADBE'].iloc[-1]
    yes_close = ADBE['ADBE'].iloc[-2]

    fig2 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig2.update_traces(delta_font=delta_font, number_font=number_font)
    fig2.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig2.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig2.update_traces(delta_decreasing_color='red')

   # Fig3 for TSLA
    close = TSLA['TSLA'].iloc[-1]
    yes_close = TSLA['TSLA'].iloc[-2]

    fig3 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig3.update_traces(delta_font=delta_font, number_font=number_font)
    fig3.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig3.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig3.update_traces(delta_decreasing_color='red')

     # Fig4 for TSLA
    close = NVDA['NVDA'].iloc[-1]
    yes_close = NVDA['NVDA'].iloc[-2]

    fig4 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig4.update_traces(delta_font=delta_font, number_font=number_font)
    fig4.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig4.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig4.update_traces(delta_decreasing_color='red')

     # Fig5 for AAPL
    close = AAPL['AAPL'].iloc[-1]
    yes_close = AAPL['AAPL'].iloc[-2]

    fig5 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig5.update_traces(delta_font=delta_font, number_font=number_font)
    fig5.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig5.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig5.update_traces(delta_decreasing_color='red')

    # Fig6 for MSFT
    close = MSFT['MSFT'].iloc[-1]
    yes_close = MSFT['MSFT'].iloc[-2]

    fig6 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig6.update_traces(delta_font=delta_font, number_font=number_font)
    fig6.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig6.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig6.update_traces(delta_decreasing_color='red')

    # Fig7 for META
    close = META['META'].iloc[-1]
    yes_close = META['META'].iloc[-2]

    fig7 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig7.update_traces(delta_font=delta_font, number_font=number_font)
    fig7.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig7.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig7.update_traces(delta_decreasing_color='red')

    # Fig8 for GOOG
    close = GOOG['GOOG'].iloc[-1]
    yes_close = GOOG['GOOG'].iloc[-2]

    fig8 = go.Figure(go.Indicator(
        mode="number+delta",
        value=close,
        number={"prefix": "$", 'valueformat': "00000.2f"},
        delta={'reference': yes_close, 'relative': True, 'valueformat': "0.2%"}))
    fig8.update_traces(delta_font=delta_font, number_font=number_font)
    fig8.update_layout(margin=dict(t=18, r=0, l=30, b=0))

    if close >= yes_close:
        fig8.update_traces(delta_increasing_color='green')
    elif close < yes_close:
        fig8.update_traces(delta_decreasing_color='red')

    return fig, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8
