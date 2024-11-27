import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_auth
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create a Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define authorized users and passwords
VALID_USERNAME_PASSWORD_PAIRS = [
    ('username', 'password')  # Replace with actual username and password
]

# Set up authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("Stock Market Dashboard"),
        html.P("Welcome to the Stock Market Dashboard. Enter a stock Name below to get started."),
        html.Div(id='stock-info'),
        html.Img(id='stock-logo', style={'max-height': '100px'})
    ]),
    html.Div([
        dcc.Input(id='stock-search', type='text', placeholder='Search Indian Stock...',
                  style={'width': '50%', 'margin': '10px'}),
        html.Button('Search', id='search-button', n_clicks=0,
                    style={'margin': '10px'}),
    ]),
    html.Div([
        dcc.Graph(id='intraday-chart', style={'margin': '10px'}),
        dcc.Graph(id='daily-chart', style={'margin': '10px'}),
        dcc.Dropdown(id='indicator-dropdown',
                     options=[
                         {'label': 'Moving Average', 'value': 'ma'},
                         {'label': 'Relative Strength Index', 'value': 'rsi'},
                         {'label': 'Bollinger Bands', 'value': 'bollinger'},
                         {'label': 'Average True Range', 'value': 'atr'},
                     ],
                     value=['ma'],
                     multi=True,
                     style={'width': '50%', 'margin': '10px'}),
        dcc.Interval(id='graph-update', interval=30 * 1000, n_intervals=0),
        html.Div(id='prediction-output', style={'margin': '10px'}),
        html.Div(id='buy-sell-recommendation', style={'margin': '10px'})
    ])
])

def moving_average(data, window):
    ma_series = data['Close'].rolling(window=window).mean()
    return ma_series

def relative_strength_index(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(data, window):
    ma = moving_average(data, window)
    std = data['Close'].rolling(window=window).std()
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)
    return upper_band, lower_band

def average_true_range(data, window):
    tr = pd.DataFrame(index=data.index)
    tr['h-l'] = data['High'] - data['Low']
    tr['h-pc'] = abs(data['High'] - data['Close'].shift(1))
    tr['l-pc'] = abs(data['Low'] - data['Close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr_series = tr['tr'].rolling(window=window).mean()
    return atr_series

def generate_signals(data, indicator):
    signals = np.zeros(len(data))
    if indicator == 'ma':
        short_window = 10
        long_window = 50
        data['short_ma'] = moving_average(data, short_window)
        data['long_ma'] = moving_average(data, long_window)
        signals = np.where(data['short_ma'] > data['long_ma'], 1, -1)
    elif indicator == 'rsi':
        window = 14
        data['rsi'] = relative_strength_index(data, window)
        signals = np.where(data['rsi'] < 30, 1, np.where(data['rsi'] > 70, -1, 0))
    elif indicator == 'bollinger':
        window = 20
        upper_band, lower_band = bollinger_bands(data, window)
        data['upper_band'] = upper_band
        data['lower_band'] = lower_band
        signals = np.where(data['Close'] < lower_band, 1, np.where(data['Close'] > upper_band, -1, 0))
    elif indicator == 'atr':
        window = 14
        data['atr'] = average_true_range(data, window)
    return signals

def make_prediction(data, selected_indicators):
    signals = np.zeros(len(data))
    for indicator in selected_indicators:
        signal_column = f'{indicator}_Signals'
        signals = np.logical_or(signals, data[signal_column])

    if 1 in signals:
        return "Buy"
    elif -1 in signals:
        return "Sell"
    else:
        return "Hold"

def load_historical_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def prepare_lstm_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

def build_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, batch_size=32, epochs=100):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

def predict_lstm_prices(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

def generate_recommendation(predicted_prices):
    if len(predicted_prices) < 3:
        return "Insufficient data for recommendation"

    trend = 0
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > predicted_prices[i-1]:
            trend += 1
        elif predicted_prices[i] < predicted_prices[i-1]:
            trend -= 1

    if trend > 0:
        return "Recommendation: Buy"
    elif trend < 0:
        return "Recommendation: Sell"
    else:
        return "Recommendation: Hold"

@app.callback(
    [Output('stock-info', 'children'),
     Output('stock-logo', 'src')],
    [Input('search-button', 'n_clicks')],
    [State('stock-search', 'value')]
)
def update_stock_info_and_logo(n_clicks, search_query):
    if n_clicks and search_query:
        stock_data = yf.Ticker(search_query)
        if not stock_data.history().empty:
            info = stock_data.info
            stock_info = html.Div([
                html.H2(f"{info['longName']} ({info['symbol']})"),
                html.P(f"Sector: {info.get('sector', 'N/A')}"),
                html.P(f"Industry: {info.get('industry', 'N/A')}"),
                html.P(f"Country: {info.get('country', 'N/A')}"),
                html.P(f"Market Cap: {info.get('marketCap', 'N/A')}"),
                html.P(f"52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')} {info.get('currency', 'N/A')}"),
                html.P(f"52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')} {info.get('currency', 'N/A')}")
            ])
            logo_url = f"https://logo.clearbit.com/{info.get('website', '')}"
            response = requests.get(logo_url)
            if response.status_code == 200:
                logo_src = logo_url
            else:
                logo_src = ''
            return stock_info, logo_src
    return None, None

@app.callback([Output('intraday-chart', 'figure'),
               Output('daily-chart', 'figure')],
              [Input('search-button', 'n_clicks')],
              [State('stock-search', 'value'),
               State('indicator-dropdown', 'value')])
def update_chart(n_clicks, search_query, selected_indicators):
    if n_clicks and search_query:
        stock_data = yf.Ticker(search_query)
        intraday_data = stock_data.history(interval='5m', period='1d')
        daily_data = stock_data.history(interval='1d', period='1y')

        if intraday_data.empty or daily_data.empty:
            return {}, {}

        intraday_fig = go.Figure()
        intraday_fig.add_trace(go.Candlestick(
            x=intraday_data.index,
            open=intraday_data['Open'],
            high=intraday_data['High'],
            low=intraday_data['Low'],
            close=intraday_data['Close'],
            name='Intraday Data'
        ))
        intraday_fig.update_layout(title='Intraday Chart')

        daily_fig = go.Figure()
        daily_fig.add_trace(go.Candlestick(
            x=daily_data.index,
            open=daily_data['Open'],
            high=daily_data['High'],
            low=daily_data['Low'],
            close=daily_data['Close'],
            name='Daily Data'
        ))
        daily_fig.update_layout(title='Daily Chart (Last One Year)')

        for indicator in selected_indicators:
            if indicator == 'ma':
                window = 14
                daily_data['ma'] = moving_average(daily_data, window)
                daily_fig.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=daily_data['ma'],
                    mode='lines',
                    name=f'{window}-day MA'
                ))
            elif indicator == 'rsi':
                window = 14
                daily_data['rsi'] = relative_strength_index(daily_data, window)
                daily_fig.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=daily_data['rsi'],
                    mode='lines',
                    name=f'{window}-day RSI'
                ))
            elif indicator == 'bollinger':
                window = 20
                upper_band, lower_band = bollinger_bands(daily_data, window)
                daily_fig.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=upper_band,
                    mode='lines',
                    name='Upper Bollinger Band'
                ))
                daily_fig.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=lower_band,
                    mode='lines',
                    name='Lower Bollinger Band'
                ))
            elif indicator == 'atr':
                window = 14
                daily_data['atr'] = average_true_range(daily_data, window)
                daily_fig.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=daily_data['atr'],
                    mode='lines',
                    name=f'{window}-day ATR'
                ))

        return intraday_fig, daily_fig
    return {}, {}

@app.callback(
    [Output('prediction-output', 'children'),
     Output('buy-sell-recommendation', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('stock-search', 'value')]
)
def update_prediction_and_recommendation(n_clicks, search_query):
    if n_clicks and search_query:
        stock_data = yf.Ticker(search_query)
        daily_data = stock_data.history(interval='1d', period='1y')

        if daily_data.empty:
            return "No data available for prediction.", "No recommendation available."

        X, y, scaler = prepare_lstm_data(daily_data)
        model = build_lstm_model(X.shape[1])
        train_lstm_model(model, X, y)

        future_days = 3
        X_test = X[-future_days:]
        predicted_prices = predict_lstm_prices(model, X_test, scaler)
        prediction_output = f"Next {future_days} days predicted prices: {predicted_prices.flatten()}"

        recommendation = generate_recommendation(predicted_prices)
        return prediction_output, recommendation

    return "No data available for prediction.", "No recommendation available."

if __name__ == '__main__':
    app.run_server(debug=True)
