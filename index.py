import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
import pandas as pd
import numpy as np
import datetime

from app import app

from Tab_Content.stock_daily_return import stock_daily_return_LO
from Tab_Content.stock_performance_comparision import stock_perf_comp_LO
from Tab_Content.portfolio_optimization import portfolio_opt_LO
from Tab_Content.cluster_analysis import cluster_analysis_LO
from Tab_Content.backtest_tab_layout import backtest_tab_LO
from Tab_Content.ml_tab_layout import ML_tab_LO
from Tab_Content.home import home_LO
from Tab_Content.about_me import about_me_LO

import dash_bootstrap_components as dbc

meta_tags = [{'name': 'viewport', 'content': 'width=device-width'}]
external_stylesheets = [meta_tags]

today = datetime.date.today()
year = today.year
copyright_message = f"© {year} -Sathya Jayagopi. All Rights Reserved."

server = app.server

tabs_styles = {'display': 'flex',
               'flex-direction': 'row',
               'margin-left': '2%',
               'height': '85px'

               }

tab_style = {'color': 'white',
             'backgroundColor': '#0c89cd',
             'fontSize': '15px',
             # 'padding': '1.3vh',
             # 'padding-top': '1.0vh',
             'width': '18%',
             'padding-bottom': '1.0vh',
             }

selected_tab_style = {  # 'color': 'white',
    #   'padding': '1.1vh',
    # 'padding-top': '1.0vh',
    'backgroundColor': '#0c89cd',
    'fontSize': '15px',

    #  'border-top': '1px solid white',
    #  'border-left': '1px solid white',
    #   'border-right': '1px solid white',

}


app.layout = html.Div([

    html.Div([

        html.Div([
            html.Img(src=app.get_asset_url('python.png'),
                 className='image'),
            html.H2(['Python4Trading.com'], className='title'),


            html.Div([

                dcc.Tabs(id='main_tabs', value='home',
                 children=[
                     dcc.Tab(
                         label='Home',
                         value='home',
                         style=tab_style,
                         selected_style=selected_tab_style,
                     ),

                     dcc.Tab(label='Daily Return Analysis',
                             value='return_daily_distribution',
                             style=tab_style,
                             selected_style=selected_tab_style),

                     dcc.Tab(label='Stock Return Comparision',
                             value='stock_comparision',
                             style=tab_style,
                             selected_style=selected_tab_style,
                             ),

                     dcc.Tab(label='Portfolio Optimizer',
                             value='portfolio_optimizer',
                             style=tab_style,
                             selected_style=selected_tab_style),

                     dcc.Tab(label='Cluster Analysis',
                             value='cluster_analysis',
                             style=tab_style,
                             selected_style=selected_tab_style),

                     dcc.Tab(label='Backtest Trading Strategy',
                             value='backtest',
                             style=tab_style,
                             selected_style=selected_tab_style),

                     dcc.Tab(label='ML Predictions',
                             value='ml_prediction',
                             style=tab_style,
                             selected_style=selected_tab_style),

                     dcc.Tab(label='About Me',
                             value='about_me',
                             style=tab_style,
                             selected_style=selected_tab_style),


                 ], style=tabs_styles,
                    className='tab_bar', colors={'border': None,
                                                 'primary': None,
                                                 'background': None})



            ], className='tab_container'),

        ], className='title_container'),


    ], className='header_container'),



    html.Div(id='return_tab_content', children=[]),

    html.Footer([

        html.Div([html.P(copyright_message),

                  ], className='footerContent')

    ], className='footerContainer')

], className='whole_container')


@app.callback(Output('return_tab_content', 'children'),
              [Input('main_tabs', 'value')])
def update_content(main_tabs):

    if main_tabs == 'home':
        return home_LO
    elif main_tabs == 'return_daily_distribution':
        return stock_daily_return_LO
    elif main_tabs == 'stock_comparision':
        return stock_perf_comp_LO
    elif main_tabs == 'portfolio_optimizer':
        return portfolio_opt_LO
    elif main_tabs == 'cluster_analysis':
        return cluster_analysis_LO
    elif main_tabs == 'backtest':
        return backtest_tab_LO
    elif main_tabs == 'ml_prediction':
        return ML_tab_LO
    elif main_tabs == 'about_me':
        return about_me_LO


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host="0.0.0.0", port=8080)
