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

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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


seasonality_period = [
    {'label': '10', 'value': '10'},
    {'label': '20', 'value': '20'},
    {'label': '30', 'value': '30'},
    {'label': '60', 'value': '60'},
    {'label': '90', 'value': '90'},
    {'label': '120', 'value': '120'},
    {'label': '150', 'value': '150'},
    {'label': '200', 'value': '200'}

]
input_feature_list = [
    {'label': 'SMA Cross Over', 'value': 'SMA_Cross_Over'},
    {'label': 'EMA Cross Over', 'value': 'EMA_Cross_Over'},
    {'label': 'Double EMA Cross Over', 'value': 'Double_EMA_Cross_Over'},
    {'label': 'MACD Cross Over', 'value': 'MACD_Cross_Over'},
    {'label': 'RSI', 'value': 'RSI'}
]

ML_XGBoost_LO = html.Div([

    html.Br(),

    html.H5('XGBoost Classifier with Seasonality Decomposition:',
            className='content_title'),

    html.P("Here users can perform the Dickey Fuller test to determine the Stationarity of Stock price, Seasonal Decomposition and analyze how sucessful the XGBoost ML model is in predicting tomorrow's price direction (Up/Down move) through XGBoost Classifier."),

    html.Label(['Selection Company/EFT from dorp down:'],
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

    html.Label(
        'Select the period(days) to decompose the seasonality in input time series'),
    dcc.Dropdown(id="seasonality_period",
                 multi=False,
                 value='90',
                 options=seasonality_period,
                 className='dropdown_box',
                 ),

    html.Label(
        'Select the additional Input Features to be considered by the XGBoost Classifer model:'),
    dcc.Dropdown(id="additional_feature",
                 multi=True,
                 value=['SMA_Cross_Over'],
                 options=input_feature_list,
                 className='dropdown_box',
                 ),
    html.Label('By default Close price will be an input feature. If None selected, then Close Price will be input feature. '),

    html.Button('Submit', id='BT_XGB',
                className='button', n_clicks=0),

    dcc.Loading(children=[

        html.Br(),

        html.Div(id='stationary', className='stationary'),

        dcc.Graph(id='seasonality_decomposition', figure=blank_fig(),
                  config={'displayModeBar': False},),

        html.Br(),
        html.Div([], className='content_divider'),

        html.Div(id='accuracy'),
        html.Div(
            dash_table.DataTable(id='precision_SL',
                                 style_cell={
                                     'textAlign': 'center'},
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
                                 }
                                 ), className='table_container'),

        html.Br(),

        dcc.Graph(id='confusion_matrix_1', figure=blank_fig(),
                  config={'displayModeBar': False}),

        html.Hr(),
        dcc.Graph(id='direction_over_time', figure=blank_fig(),
                  config={'displayModeBar': False}),

    ], type="circle", fullscreen=True),

])


@app.callback(
    [Output('stationary', 'children'),
     Output('seasonality_decomposition', 'figure'),
     Output('accuracy', 'children'),
     Output('precision_SL', component_property='data'),
     Output('precision_SL', component_property='columns'),
     Output('confusion_matrix_1', 'figure'),
     Output('direction_over_time', 'figure')],
    [Input('BT_XGB', 'n_clicks')],
    [State('selected_stock', 'value'),
     State('additional_feature', 'value'),
     State('seasonality_period', 'value')],
    prevent_initial_call=True)
def XGBoost_model(n_clicks, selected_stock, additional_feature, seasonality_period):
    print('Call Back working')

    seasonality_period = int(seasonality_period)

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df[['Date', 'Adj Close']].copy()
    df.rename(columns={'Adj Close': 'Close Price'}, inplace=True)
    df.set_index('Date', inplace=True)

    # Dickey Fuller Test

    adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)

    if pvalue > .05:
        a = '. If p value above .05 then the data is not Stationary.'
    else:
        a = '. The data is Stationary'

    s = 'The pvalue is = ' + str(round(pvalue, 4)) + a

    df_test = html.Div([
        html.Div([], className='content_divider'),
        html.Br(),
        html.H5('Dickey Fuller Test:', className='content_title'),
        html.P(s)

    ])
    decomposed = seasonal_decompose(df['Close Price'],
                                    model='additive', period=seasonality_period)

    trend = decomposed.trend
    seasonal = decomposed.seasonal  # Cyclic behavior may not be seasonal!
    residual = decomposed.resid

    fig = make_subplots(rows=4,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02
                        )

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close Price'], name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=seasonal,
                  name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=residual,
                  name='Residual'), row=4, col=1)

    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Trend', row=2, col=1)
    fig.update_yaxes(title_text='Seasonal', row=3, col=1)
    fig.update_yaxes(title_text='Residual', row=4, col=1)

    fig.update_xaxes(
        showline=True,
        linecolor='black',
        showgrid=True)
    fig.update_yaxes(
        showline=True,
        linecolor='black',
        showgrid=True)

    fig.update_layout(

        title='Seasonality Decomposition for {}'.format(selected_stock),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            orientation='h',
            x=.55,
            y=1.10,
            traceorder='normal',
            borderwidth=1)
    )

    df['50_SMA'] = df['Close Price'].rolling(50).mean()
    df['200_SMA'] = df['Close Price'].rolling(200).mean()
    df['10_Expo_MA'] = df['Close Price'].ewm(span=10, adjust=False).mean()
    df['21_Expo_EMA'] = df['Close Price'].ewm(span=21, adjust=False).mean()

    df['10_DEMA'] = (2*df['10_Expo_MA']) - \
        df['10_Expo_MA'].ewm(span=10, adjust=False).mean()
    df['21_DEMA'] = (2*df['21_Expo_EMA']) - \
        df['21_Expo_EMA'].ewm(span=21, adjust=False).mean()

    # MACD Line
    df['MACD_Line'] = df['10_Expo_MA']-df['21_Expo_EMA']

    # Signal Line
    df['Signal_line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    # SMA Cross Over
    df['SMA_Cross_Over'] = df['50_SMA']/df['200_SMA']

    # EMA Cross Over
    df['EMA_Cross_Over'] = df['10_Expo_MA']/df['21_Expo_EMA']

    # Double EMA Cross Over
    df['Double_EMA_Cross_Over'] = df['10_DEMA']/df['21_DEMA']

    # MACD Cross Over
    df['MACD_Cross_Over'] = df['MACD_Line']/df['Signal_line']

    # Calculate RSI
    def RSI(df, RSI_period=14, column='Close Price'):

        def SMA(df, period=RSI_period, column='Close Price'):

            return df[column].rolling(window=period).mean()

        delta = df[column].diff(1)
        delta.dropna(inplace=True)
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        df['up'] = up
        df['down'] = down
        AVG_Gain = SMA(df, RSI_period, column='up')
        AVG_Loss = abs(SMA(df, RSI_period, column='down'))

        RS = AVG_Gain/AVG_Loss

        RSI = 100.0 - (100.0/(1+RS))
        df['RSI'] = RSI

        return df

    df = RSI(df, RSI_period=14, column='Close Price')

    df.dropna(inplace=True)

    input_features = ['Close Price']

    input_features.extend(additional_feature)

    data_set = df[input_features]

    print(input_features)
    print(data_set.index)

    data_set['Yesterday_to_Today_Percent_Change'] = data_set['Close Price'].pct_change() * \
        100

    data_set['Today_to_tomorrow_pct_change'] = data_set['Yesterday_to_Today_Percent_Change'].shift(
        -1)
    data_set['Yesterday_pct_change'] = data_set['Yesterday_to_Today_Percent_Change'].shift(
        +1)

    data_set['Up_or_Down'] = (
        data_set['Today_to_tomorrow_pct_change'] > 0).astype(int)
    data_set = data_set.dropna()

    data_set.dropna(inplace=True)

    X = data_set.drop(
        labels=["Close Price", "Today_to_tomorrow_pct_change", 'Up_or_Down'], axis=1)
    Y = data_set['Up_or_Down']

    test_size = 0.1

    test_split_idx = int(data_set.shape[0] * (1-test_size))

    X_train = X.iloc[:test_split_idx].copy()
    X_test = X.iloc[test_split_idx:].copy()

    y_train = Y[:test_split_idx].copy()
    y_test = Y[test_split_idx:].copy()

    print(y_train.head())
    print(X_train.info())

    print(X_train.head())

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Evaluate the classifier on test data
    from sklearn.metrics import accuracy_score
    b = round((accuracy_score(y_test, y_pred))*100, 2)

    accuracy = html.Div([
        html.Br(),
        html.P(
            'Accuracy % of XGBoost model to predict next day direction of selected instrument:'),
        html.H4(b),
        html.Br(),
        html.Label('Precision, Recall and F1 Score of XGBoost Model Prediction:'),

    ])
    print(b)
    print(type(y_pred))
    print(type(y_test))

    analysis_price_df = pd.DataFrame()
    analysis_price_df['Actual Up_or_Down Move'] = data_set['Up_or_Down'][-len(
        y_pred):]
    analysis_price_df.reset_index(inplace=True)
    analysis_price_df['Predicted Up_or_Down Move'] = pd.Series(y_pred)

    print(analysis_price_df.tail(10))

    print("Actual")
    print(analysis_price_df['Actual Up_or_Down Move'].value_counts())

    print(" Predicted")
    print(analysis_price_df['Predicted Up_or_Down Move'].value_counts())

    def up_down_move(df, column):
        if df[column] == 0:
            return 'Down Move'
        else:
            return 'Up Move'

    analysis_price_df['Actual Up or Down'] = analysis_price_df.apply(up_down_move,
                                                                     column='Actual Up_or_Down Move', axis=1)

    analysis_price_df['Predicted Up or Down'] = analysis_price_df.apply(up_down_move,
                                                                        column='Predicted Up_or_Down Move', axis=1)

    print(analysis_price_df.tail(10))

    print("Actual")
    print(analysis_price_df['Actual Up or Down'].value_counts())

    print(" Predicted")
    print(analysis_price_df['Predicted Up or Down'].value_counts())

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    x = ['Predicted Down Move', 'Predicted Up Move']
    y = ['Actual Down Move', 'Actual Up Move']

    fig1 = px.imshow(
        cm, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
    fig1.update_traces(text=cm, texttemplate="%{text}")
    fig1.update_xaxes(side="bottom")
    fig1.update_layout(title='Confusion Matrix of XGBoost prediction',
                       width=600,
                       height=400)

    report = classification_report(y_test, y_pred, output_dict=True)

    class_report = pd.DataFrame(report).transpose()
    class_report.rename(index={'0': 'Down Move',
                               '1': 'Up Move'},
                        inplace=True)

    print(class_report.index)

    print(class_report)

    updated_classification_report = class_report.drop('accuracy')

    print('Updated Classification report:')

    print(updated_classification_report)

    directional_change_data = []

    analysis_price_df.replace(to_replace=0, value=-1)

    print(analysis_price_df.tail())

    trace1 = go.Scatter(x=analysis_price_df['Date'],
                        y=analysis_price_df['Actual Up_or_Down Move'],
                        mode='lines',
                        name='Actual Directional Change',
                        opacity=.5)

    directional_change_data.append(trace1)

    trace2 = go.Scatter(x=analysis_price_df['Date'],
                        y=analysis_price_df['Predicted Up_or_Down Move'],
                        mode='lines',
                        name='Predicted Directional Change',
                        opacity=.5)

    directional_change_data.append(trace2)

    directional_change_layout = go.Layout(
        title='Daily Directional Change Predicted Vs Actual for {}'.format(
            selected_stock),
        yaxis=dict(title='Direction Change',
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
            x=.85,
            y=1.12,
            traceorder='normal',
            borderwidth=1)
    )

    fig2 = go.Figure(data=directional_change_data,
                     layout=directional_change_layout)

    updated_classification_report.reset_index(inplace=True)

    precision_col = [{'name': col, 'id': col}
                     for col in updated_classification_report.columns]
    precision_data = updated_classification_report.to_dict(orient='records')

    return df_test, fig, accuracy, precision_data, precision_col, fig1, fig2
