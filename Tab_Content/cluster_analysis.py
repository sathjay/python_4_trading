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
from sklearn.cluster import KMeans
import plotly.offline as pyo
import plotly.graph_objs as go
import os
from plotly.subplots import make_subplots
from app import app


df = pd.read_csv('data/Dow_30_Fundamental_data_21_22.csv', sep=',', header=0)

Metrics_list = ['Debt To Equity', 'Debt To Assets',
                'Dividend Yield', 'Net Profit Margin', 'Free Cash Flow Yield',
                'Price Cash Flow Ratio', 'Price Earnings Ratio', 'Price Earnings To Growth Ratio',
                'Price Sales Ratio', 'Return On Assets', 'Return On Equity', 'Returns', 'Variance']

df.reset_index(drop=True, inplace=True)
df.set_index(["SYMBOL"], inplace=True, drop=True)


df['Dividend Yield'] = df['Dividend Yield'].fillna(0)
df = df.astype(float)
df['Year'] = df['Year'].astype(int)
df['Dividend Yield'] = df['Dividend Yield'].fillna(0)


Metrics_dropdown_list = []

for Metric in Metrics_list:
    Metrics_dropdown_list.append({'label': Metric, 'value': Metric})


year_dropdown_list = []

for year in df['Year'].unique():
    year_dropdown_list.append({'label': year, 'value': year})


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, plot_bgcolor='rgb(186, 228, 242)',
                      paper_bgcolor='rgb(186, 228, 242)')
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


cluster_analysis_LO = html.Div([

    html.H4("Cluster Analysis on Dow 30 Index Component's Fundamental Financial Data:"),

    html.P(['Cluster analysis allows investors to identify companies with similar financial traits, regardless of differences in industries, geographies, or asset classes. By leveraging this technique, investors can construct portfolios of assets that share common financial characteristics.',
            html.Br(),
            'In this tab, users can perform cluster analysis on the financial data of stocks within the Dow 30 index. They can select the input features for the analysis and visualize the resulting clusters in a 3D graph.']),


    html.Br(),
    html.Label('Select Fundamental Data Year:'),
    dcc.Dropdown(id="selected_year",
                 multi=False,
                 disabled=False,
                 options=year_dropdown_list,
                 value=2022,
                 className='dropdown_box',

                 ),

    html.Label(
        'Select the input features/metrics for Cluster Analysis Model. (Select minimum of 3 features)'),
    dcc.Dropdown(id="selected_metrics",
                 multi=True,
                 value=['Returns', 'Variance', 'Debt To Assets'],
                 disabled=False,
                 options=Metrics_dropdown_list,
                 className='dropdown_box',
                 ),

    html.Br(),
    html.Div([], className='content_divider'),
    html.Br(),

    html.Div([
        html.Div([
            html.P('Select the X axis'),
            dcc.Dropdown(id="x_axis",
                         multi=False,
                         disabled=False,
                         options=[]
                         )
        ], className='input_feature_3d_drop_down'),

        html.Div([
            html.P('Select the Y axis'),
            dcc.Dropdown(id="y_axis",
                         multi=False,
                         disabled=False,
                         options=[]
                         ),

        ], className='input_feature_3d_drop_down'),

        html.Div([
            html.P('Select the Z axis'),
            dcc.Dropdown(id="z_axis",
                         multi=False,
                         disabled=False,
                         options=[]
                         ),
        ], className='input_feature_3d_drop_down'),

    ], className='container_3d_graph_input_feature'),

    dcc.Loading(children=[
        dcc.Graph(id='graph_3d', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Br(),
        html.Div(id='cluster_analysis_table_label'),
        html.Br(),
        html.Div(
            dash_table.DataTable(id='cluster_table',
                                 filter_action="native",
                                 sort_action="native",
                                 sort_mode="multi",
                                 editable=False,
                                 style_table={'overflowY': 'scroll',
                                              'maxHeight': '600px'},
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
                                 },
                                 style_cell={'textAlign': 'center'},

                                 ), className='table_container'),

    ], type="circle", fullscreen=True),

], className='main_container')


@app.callback(  # This is added to show the drop down list option
                [Output('x_axis', 'options'),
                 Output('y_axis', 'options'),
                 Output('z_axis', 'options')],
                [Input('selected_metrics', 'value')])
def graph_axis_options(selected_metrics):
    a = selected_metrics
    return a, a, a


@app.callback(  # This call back is added to prefil the default values in dropdown.
    [Output('x_axis', 'value'),
     Output('y_axis', 'value'),
     Output('z_axis', 'value')],
    [Input('selected_metrics', 'value')])
def graph_axis_options(selected_metrics):
    a = selected_metrics
    return a[0], a[1], a[2]


@app.callback(
    [Output('graph_3d', component_property='figure'),
     Output('cluster_analysis_table_label', 'children'),
     Output('cluster_table', 'columns'),
     Output('cluster_table', 'data')],

    [
        Input('selected_year', 'value'),
        Input('selected_metrics', 'value'),
        Input('x_axis', 'value'),
        Input('y_axis', 'value'),
        Input('z_axis', 'value')
    ])
def graph_3d(selected_year, selected_metrics, x_axis, y_axis, z_axis):

    df_year = df.loc[df['Year'].isin([selected_year])]

    dff = df_year[selected_metrics]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(dff)
    labels = kmeans.labels_

    dff = dff.astype(float)
    dff = dff.round(3)

    dff['Cluster_Labels'] = labels

    fig = px.scatter_3d(
        data_frame=dff,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color="Cluster_Labels",

        opacity=0.5,
        hover_name=dff.index.values,
    )

    fig.update_traces(marker_size=8)
    fig.update_layout(title='Cluster Analysis on Dow30 Index Component:')

    dff = dff.reset_index(level=0)
    s_columns = [{'name': col, 'id': col} for col in dff.columns]
    s_data = dff.to_dict(orient='records')

    label = html.P('Cluster Analysis Summary with Cluster Labels:')

    return fig, label, s_columns, s_data,
