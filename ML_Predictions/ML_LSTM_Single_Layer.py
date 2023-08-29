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

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers import Input as Keras_Input
from sklearn.preprocessing import MinMaxScaler

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


additional_feature_list = [
    {'label': 'SMA Cross Over', 'value': 'SMA_Cross_Over'},
    {'label': 'EMA Cross Over', 'value': 'EMA_Cross_Over'},
    {'label': 'Double EMA Cross Over', 'value': 'Double_EMA_Cross_Over'},
    {'label': 'MACD Cross Over', 'value': 'MACD_Cross_Over'},
    {'label': 'RSI', 'value': 'RSI'}
]

lookback_days_list = [

    {'label': '32 Days', 'value': '32'},
    {'label': '64 Days', 'value': '64'},

]

batch_size_list = [

    {'label': '32', 'value': '32'},
    {'label': '64', 'value': '64'},
    {'label': '128', 'value': '128'},

]

activation_function_list = [
    {'label': 'Linear', 'value': 'linear'},
    {'label': 'Leaky Rectified Linear unit', 'value': 'LeakyReLU'},
    {'label': 'Rectified Linear unit', 'value': 'relu'},
    {'label': 'Scaled Exponential Linear unit', 'value': 'selu'},
    {'label': 'Sigmoid function ', 'value': 'sigmoid'},
    {'label': 'Swish function', 'value': 'swish'},
    {'label': 'Hyperbolic Tangent function', 'value': 'tanh'}
]


LSTM_SL_LO = html.Div([
    html.Br(),
    html.P("Here users can see how well a Single Layer Long Short Term Memory (LTSM) Recurring Neural Network (RNN) model\
                            can predit the stock price. User can select the Stock, Input features, Look back period,\
                            Batch Size and Activation function and see the performance of the model "),


    html.Label(['Select Company/EFT from dorp down:'],
               style={'font-weight': 'bold'}),
    dcc.Dropdown(id='selected_stock',
                 options=stock_dropdown_list,
                 optionHeight=35,
                 value='GLD',
                 disabled=False,  # disable dropdown value selection
                 multi=False,  # allow multiple dropdown values to be selected
                 searchable=True,  # allow user-searching of dropdown values
                 # gray, default text shown when no option is selected
                 placeholder='Please select...',
                 clearable=True,  # allow user to removes the selected value
                 className='dropdown_box',  # activate separate CSS document in assets folder
                 ),

    html.Label('Select the Input features for Single Layer LSTM model: (By default Close price will be an input feature. If None selected, then Close Price will be input feature.)'),
    dcc.Dropdown(id="additional_feature",
                 multi=True,
                 value=['SMA_Cross_Over'],
                 options=additional_feature_list,
                 className='dropdown_box',
                 ),

    html.Br(),
    html.H4('Select the LSTM RNN model parameters: '),

    html.Label(
        'Select the number of past days to look back for LSTM RNN Prediction:'),
    dcc.Dropdown(id="lookback_days",
                 multi=False,
                 value='32',
                 options=lookback_days_list,
                 className='dropdown_box',
                 ),

    html.Label('Select the Batch Size:'),
    dcc.Dropdown(id="batch_size",
                 multi=False,
                 value='32',
                 options=batch_size_list,
                 className='dropdown_box',
                 ),

    html.Label('Select the Activation Function'),
    dcc.Dropdown(id="activation",
                 multi=False,
                 value='relu',
                 options=activation_function_list,
                 className='dropdown_box',
                 ),

    html.Button('Submit', id='BT_LSTM_SL',
                className='button', n_clicks=0),

    html.Hr(),

    dcc.Loading(children=[
        html.Div(id='message'),

        html.Div(

            dash_table.DataTable(
                id='model_summary_SL',
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
                }
            ),
            className='table_container'),

        html.Hr(),
        dcc.Graph(id='price_chart_LSTM_Single_Layer', figure=blank_fig(),
                  style={'width': '92%', 'height': '74vh'}),

        html.Hr(),
        dcc.Graph(id='daily_price_change_LSTM_Single_Layer', figure=blank_fig(),
                  style={'width': '92%', 'height': '74vh'}),

        html.Hr(),
        html.Div(
            dash_table.DataTable(id='accuracy_SL',
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
                                 }),
            className='table_container'),

    ], type="circle", fullscreen=True),

])


@app.callback(
    [Output('message', 'children'),
     Output('model_summary_SL', component_property='data'),
     Output('model_summary_SL', component_property='columns'),
     Output('price_chart_LSTM_Single_Layer', 'figure'),
     Output('daily_price_change_LSTM_Single_Layer', 'figure'),
     Output('accuracy_SL', component_property='data'),
     Output('accuracy_SL', component_property='columns'),],
    [Input('BT_LSTM_SL', 'n_clicks')],
    [State('selected_stock', 'value'),
     State('lookback_days', 'value'),
     State('batch_size', 'value'),
     State('activation', 'value'),
     State('additional_feature', 'value')
     ],
    prevent_initial_call=True
)
def LTSM_Multivariate_Single_Layer_model(n_clicks, selected_stock, lookback_days, batch_size, activation, additional_feature):

    backcandles = int(lookback_days)
    batch_size = int(batch_size)

    print(additional_feature)

    activation = activation

    df = yf.download(selected_stock, period='1d',
                     start=start_date, end=end_date)
    df.reset_index(inplace=True)

    df = df[['Date', 'Adj Close']].copy()
    df.rename(columns={'Adj Close': 'Close Price'}, inplace=True)

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

    input_features = ['Close Price', 'Date']

    input_features.extend(additional_feature)

    print('The input Features', input_features)

    data_set = df[input_features]

    data_set['Daily_Percent_Change'] = data_set['Close Price'].pct_change()*100

    data_set['TargetNextClose'] = data_set['Close Price'].shift(-1)

    data_set.dropna(inplace=True)

    splitlimit = int(len(data_set)*0.8)
    print(splitlimit)

    df_xtest = data_set.iloc[splitlimit+backcandles+1:]

    data_set.drop(['Date'], axis=1, inplace=True)

    print(data_set.columns.values)

    sc = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = sc.fit_transform(data_set)
    print(data_set_scaled)

    # multiple feature from data provided to the model
    X = []
    print(data_set_scaled.shape[0])
    # Since the date column is added subtracting that here
    for j in range(len(input_features)-1):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i-backcandles:i, j])

    # move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])

    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    y = np.reshape(yi, (len(yi), 1))

    print(X)
    print(X.shape)
    print(y)
    print(y.shape)

    # split data into train test sets

    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    nuerons = int((backcandles) * (len(input_features)-1))

    # Date column added in input feature is subtracted.
    lstm_input = Keras_Input(
        shape=(backcandles, (len(input_features)-1)), name='Lstm_Input')
    inputs = LSTM(nuerons, name='First_Layer')(lstm_input)
    inputs = Dense(1, name='Dense_Layer')(inputs)
    output = Activation(activation, name='Output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=batch_size,
              epochs=8, validation_split=0.1)
    model.summary()

    input_features.remove('Date')

    listToStr = ', '.join([str(elem) for elem in input_features])

    a = 'The input features are: '+listToStr+'.'

    message = html.Div([
        html.H5('The Model summary and Reults:', className='content_title'),
        html.Br(),
        html.H6(a),
        html.Br(),
        html.H6('The model parameters are:'),
    ])

    y_pred = model.predict(X_test)

    print(len(y_pred))
    print(len(y_test))

    y_prediction_copies = np.repeat(y_pred, data_set.shape[1], axis=-1)
    y_future_price = sc.inverse_transform(y_prediction_copies)[:, 0]

    y_test_copies = np.repeat(y_test, data_set.shape[1], axis=-1)
    y_actual_price = sc.inverse_transform(y_test_copies)[:, 0]

    print(len(y_future_price))
    print((len(y_actual_price)))

    for i in range(len(y_actual_price)):
        print("{}|Actual: {} | Predicted: {}".format(
            i, y_actual_price[i], y_future_price[i]))

    price_chart_LSTM_Single_Layer_Data = []

    trace1 = go.Scatter(x=df['Date'],
                        y=df['Close Price'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='black', width=1),
                        opacity=.7
                        )

    price_chart_LSTM_Single_Layer_Data.append(trace1)

    trace2 = go.Scatter(x=df_xtest['Date'].values,
                        y=y_future_price,
                        mode='lines',
                        name='Y Prediction',
                        line=dict(color='red', width=1),
                        opacity=.7
                        )

    price_chart_LSTM_Single_Layer_Data.append(trace2)

    trace3 = go.Scatter(x=df_xtest['Date'].values,
                        y=y_actual_price,
                        mode='lines',
                        name='Y Actual',
                        line=dict(color='blue', width=1),
                        opacity=.7
                        )

    price_chart_LSTM_Single_Layer_Data.append(trace3)

    price_chart_LSTM_Single_Layer_Layout = go.Layout(
        title='LSTM RNN Prediction for {}'.format(selected_stock),
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
            orientation='v',
            x=.90,
            y=1.12,
            traceorder='normal',
            borderwidth=1)
    )

    fig = go.Figure(data=price_chart_LSTM_Single_Layer_Data,
                    layout=price_chart_LSTM_Single_Layer_Layout)

    analysis_price_df = pd.DataFrame()
    analysis_price_df['Target_Close_Price'] = data_set['TargetNextClose'][-len(
        y_future_price):]
    analysis_price_df.reset_index(inplace=True)

    analysis_price_df['Predicted_Close_Price'] = pd.Series(y_future_price)

    analysis_price_df['Diff_Target_Close_Price'] = analysis_price_df['Target_Close_Price'].pct_change() * \
        100

    analysis_price_df['Diff_Predicted_Close_Price'] = analysis_price_df['Predicted_Close_Price'].pct_change() * \
        100

    b = df['Date'][-len(y_future_price):]

    analysis_price_df['Date'] = b.values

    analysis_price_df.dropna(inplace=True)

    print('********')
    print(analysis_price_df)

    daily_price_change_data = []

    print(analysis_price_df['Diff_Predicted_Close_Price'])

    trace4 = go.Scatter(x=analysis_price_df['Date'],
                        y=analysis_price_df['Diff_Target_Close_Price'],
                        mode='lines',
                        name='Actual Price Change')

    daily_price_change_data.append(trace4)

    trace5 = go.Scatter(x=analysis_price_df['Date'],
                        y=analysis_price_df['Diff_Predicted_Close_Price'],
                        mode='lines',
                        name='Predicted Price Change')

    daily_price_change_data.append(trace5)

    daily_price_change_layout = go.Layout(
        title='Daily Price Change Predicted Vs Actual for {}'.format(
            selected_stock),
        yaxis=dict(title='Price Change',
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
            y=1.08,
            traceorder='normal',
            borderwidth=1))

    fig1 = go.Figure(data=daily_price_change_data,
                     layout=daily_price_change_layout)

    # Getting Accuracy

    def directional(df):
        if df['Diff_Target_Close_Price']*df['Diff_Predicted_Close_Price'] > 0:
            return 'Accurate'
        else:
            return 'In-Accurate'

    analysis_price_df['Directional_Accuracy'] = analysis_price_df.apply(
        directional, axis=1)

    b = analysis_price_df['Directional_Accuracy'].value_counts()

    Accuracy = pd.DataFrame(
        {'Directional Accuracy': b.index, 'Count': b.values})

    print(Accuracy)

    Acc_count = Accuracy.loc[Accuracy['Directional Accuracy']
                             == 'Accurate', 'Count'].values

    print(Acc_count)

    In_Acc_count = Accuracy.loc[Accuracy['Directional Accuracy']
                                == 'In-Accurate', 'Count'].values

    print(In_Acc_count)

    Total = Acc_count[0]+In_Acc_count[0]

    Accuracy_Percent = round(Acc_count[0]/Total, 3)*100
    Accuracy.loc[len(Accuracy.index)] = [
        'Accuracy Percent %', Accuracy_Percent]

    # Model Summary

    model_summary = pd.DataFrame(
        columns=["Name", "Type", "Shape", "Parameters"])

    index = 0
    for layer in model.layers:
        model_summary.loc[index] = [layer.name, layer.__class__.__name__, str(
            layer.output_shape), int(layer.count_params())]
        index += 1

    print(Accuracy)

    print(model_summary)

    model_summary_col = [{'name': col, 'id': col}
                         for col in model_summary.columns]
    model_summary_data = model_summary.to_dict(orient='records')

    accuracy_col = [{'name': col, 'id': col} for col in Accuracy.columns]
    accuracy_data = Accuracy.to_dict(orient='records')

    return message, model_summary_data, model_summary_col, fig, fig1, accuracy_data, accuracy_col
