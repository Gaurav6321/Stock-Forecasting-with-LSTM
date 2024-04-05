import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_auth
import dash_bootstrap_components as dbc  # Import Bootstrap components
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests

# Define the Alpha Vantage and Yahoo Finance API keys
ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'
YAHOO_FINANCE_API_KEY = 'YOUR_YAHOO_FINANCE_API_KEY'

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
        html.Div(id='stock-info'),  # Placeholder for stock info
        html.Img(id='stock-logo', style={'max-height': '100px'})  # Placeholder for stock logo
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
                     value=['ma'],  # Default selected indicators
                     multi=True,
                     style={'width': '50%', 'margin': '10px'}),
        dcc.Interval(id='graph-update', interval=30 * 1000, n_intervals=0),  # Update graph every 30 seconds
        html.Div(id='prediction-output', style={'margin': '10px'}),
        html.Div(id='buy-sell-recommendation', style={'margin': '10px'})  # Placeholder for recommendation
    ])
])

# Placeholder: Implement your custom indicators here
def moving_average(data, window):
    # For demonstration purposes, let's use a simple moving average
    ma_series = data['Close'].rolling(window=window).mean()
    return ma_series

def relative_strength_index(data, window):
    # Calculate the Relative Strength Index (RSI) indicator
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(data, window):
    # Implement the Bollinger Bands indicator here
    # ...
    # For demonstration purposes, let's use random upper and lower bands
    upper_band = data['Close'].apply(lambda x: x + np.random.random())
    lower_band = data['Close'].apply(lambda x: x - np.random.random())
    return upper_band, lower_band

def average_true_range(data, window):
    # Calculate the Average True Range (ATR) indicator
    tr = pd.DataFrame(index=data.index)
    tr['h-l'] = data['High'] - data['Low']
    tr['h-pc'] = abs(data['High'] - data['Close'].shift(1))
    tr['l-pc'] = abs(data['Low'] - data['Close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr_series = tr['tr'].rolling(window=window).mean()
    return atr_series

# Placeholder: Implement your buy and sell signal logic here
def generate_signals(data, indicator):
    signals = np.zeros(len(data))
    # Implement the buy and sell signal logic for the given indicator here
    # ...
    return signals

# Placeholder: Implement your buy/sell prediction logic here
def make_prediction(data, selected_indicators):
    signals = np.zeros(len(data))
    for indicator in selected_indicators:
        signal_column = f'{indicator}_Signals'
        signals = np.logical_or(signals, data[signal_column])

    # Implement your logic to generate the overall buy/sell prediction
    # based on the aggregated signals from all the selected indicators
    if 1 in signals:
        return "Buy"
    elif -1 in signals:
        return "Sell"
    else:
        return "Hold"

# Import LSTM-related libraries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load historical stock price data
def load_historical_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Prepare LSTM training data
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

# Build LSTM model
def build_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train LSTM model
def train_lstm_model(model, X_train, y_train, batch_size=32, epochs=100):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

# Predict future stock prices using LSTM model
def predict_lstm_prices(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

# Placeholder: Implement your buy/sell prediction logic here
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


# Callback to update stock information and logo
@app.callback(
    [Output('stock-info', 'children'),
     Output('stock-logo', 'src')],
    [Input('search-button', 'n_clicks')],
    [State('stock-search', 'value')]
)
def update_stock_info_and_logo(n_clicks, search_query):
    if n_clicks and search_query:
        # Fetch stock data using yfinance
        stock_data = yf.Ticker(search_query)

        if not stock_data.history().empty:
            # Get stock info
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

            # Fetch stock logo using Yahoo Finance API
            logo_url = f"https://logo.clearbit.com/{info.get('website', '')}"
            response = requests.get(logo_url)
            if response.status_code == 200:
                logo_src = logo_url
            else:
                logo_src = ''  # If logo retrieval fails, provide an empty source

            return stock_info, logo_src

    # Return default values if search is not triggered
    return None, None

# Callback to update the intraday and daily stock data and charts
@app.callback([Output('intraday-chart', 'figure'),
               Output('daily-chart', 'figure')],
              [Input('search-button', 'n_clicks')],
              [State('stock-search', 'value'),
               State('indicator-dropdown', 'value')])
def update_chart(n_clicks, search_query, selected_indicators):
    if n_clicks and search_query:
        # Fetch live data for the selected stock using yfinance
        intraday_data = yf.download(search_query, period='1d', interval='1m')
        daily_data = yf.download(search_query, period='1y')  # Fetch data for last year

        if intraday_data.empty or daily_data.empty:
            return {'data': [], 'layout': go.Layout(title='Intraday Stock Price with Indicators',
                                                    annotations=[dict(text='Error fetching data', showarrow=False)])}, \
                {'data': [], 'layout': go.Layout(title='Daily Stock Data for the Last Year',
                                                 annotations=[dict(text='Error fetching data', showarrow=False)])}

        # Placeholder: Implement your indicator calculations here using intraday_data and daily_data
        intraday_indicator_traces = []
        daily_indicator_traces = []
        for indicator in selected_indicators:
            if indicator == 'ma':
                ma_window = 10
                ma_intraday_series = moving_average(intraday_data, ma_window)
                ma_daily_series = moving_average(daily_data, ma_window)
                intraday_indicator_traces.append(go.Scatter(x=intraday_data.index, y=ma_intraday_series, mode='lines',
                                                            name='Intraday Moving Average ({}-day)'.format(ma_window)))
                daily_indicator_traces.append(go.Scatter(x=daily_data.index, y=ma_daily_series, mode='lines',
                                                         name='Daily Moving Average ({}-day)'.format(ma_window)))
                intraday_data['Moving_Average'] = ma_intraday_series
                daily_data['Moving_Average'] = ma_daily_series

            elif indicator == 'rsi':
                rsi_window = 14
                rsi_intraday_series = relative_strength_index(intraday_data, rsi_window)
                rsi_daily_series = relative_strength_index(daily_data, rsi_window)
                intraday_indicator_traces.append(
                    go.Scatter(x=intraday_data.index, y=rsi_intraday_series, mode='lines', name='Intraday RSI'))
                daily_indicator_traces.append(
                    go.Scatter(x=daily_data.index, y=rsi_daily_series, mode='lines', name='Daily RSI'))
                intraday_data['RSI'] = rsi_intraday_series
                daily_data['RSI'] = rsi_daily_series

            elif indicator == 'bollinger':
                bb_intraday_upper, bb_intraday_lower = bollinger_bands(intraday_data, window=20)
                bb_daily_upper, bb_daily_lower = bollinger_bands(daily_data, window=20)
                intraday_indicator_traces.append(go.Scatter(x=intraday_data.index, y=bb_intraday_upper, mode='lines',
                                                            name='Intraday Upper Bollinger Band'))
                intraday_indicator_traces.append(go.Scatter(x=intraday_data.index, y=bb_intraday_lower, mode='lines',
                                                            name='Intraday Lower Bollinger Band'))
                daily_indicator_traces.append(
                    go.Scatter(x=daily_data.index, y=bb_daily_upper, mode='lines', name='Daily Upper Bollinger Band'))
                daily_indicator_traces.append(
                    go.Scatter(x=daily_data.index, y=bb_daily_lower, mode='lines', name='Daily Lower Bollinger Band'))
                intraday_data['Bollinger_Upper'] = bb_intraday_upper
                intraday_data['Bollinger_Lower'] = bb_intraday_lower
                daily_data['Bollinger_Upper'] = bb_daily_upper
                daily_data['Bollinger_Lower'] = bb_daily_lower

            elif indicator == 'atr':
                atr_window = 14
                atr_intraday_series = average_true_range(intraday_data, atr_window)
                atr_daily_series = average_true_range(daily_data, atr_window)
                intraday_indicator_traces.append(
                    go.Scatter(x=intraday_data.index, y=atr_intraday_series, mode='lines', name='Intraday ATR'))
                daily_indicator_traces.append(
                    go.Scatter(x=daily_data.index, y=atr_daily_series, mode='lines', name='Daily ATR'))
                intraday_data['ATR'] = atr_intraday_series
                daily_data['ATR'] = atr_daily_series

        # Generate buy and sell signals for each selected indicator
        for indicator in selected_indicators:
            signals = generate_signals(intraday_data, indicator)
            intraday_data[f'{indicator}_Signals'] = signals

        # Create the intraday candlestick trace
        intraday_candlestick_trace = go.Candlestick(x=intraday_data.index,
                                                    open=intraday_data['Open'],
                                                    high=intraday_data['High'],
                                                    low=intraday_data['Low'],
                                                    close=intraday_data['Close'],
                                                    name='Intraday Candlesticks')

        # Combine the intraday candlestick trace with intraday indicator traces
        intraday_data_trace = [intraday_candlestick_trace] + intraday_indicator_traces

        intraday_layout = go.Layout(title='Intraday Stock Price with Indicators',
                                    xaxis=dict(title='Datetime'),
                                    yaxis=dict(title='Price'),
                                    showlegend=True)
        intraday_fig = go.Figure(data=intraday_data_trace, layout=intraday_layout)

        # Create the daily candlestick trace
        daily_candlestick_trace = go.Candlestick(x=daily_data.index,
                                                 open=daily_data['Open'],
                                                 high=daily_data['High'],
                                                 low=daily_data['Low'],
                                                 close=daily_data['Close'],
                                                 name='Daily Candlesticks')

        # Combine the daily candlestick trace with daily indicator traces
        daily_data_trace = [daily_candlestick_trace] + daily_indicator_traces

        daily_layout = go.Layout(title='Daily Stock Data for the Last Year',
                                 xaxis=dict(title='Datetime'),
                                 yaxis=dict(title='Price'),
                                 showlegend=True)
        daily_fig = go.Figure(data=daily_data_trace, layout=daily_layout)

        return intraday_fig, daily_fig

    else:
        # Return empty charts if no stock is searched
        return {'data': [], 'layout': go.Layout(title='Intraday Stock Price with Indicators')}, \
            {'data': [], 'layout': go.Layout(title='Daily Stock Data for the Last Year')}

# Callback for LSTM prediction and recommendation
@app.callback([Output('prediction-output', 'children'),
               Output('buy-sell-recommendation', 'children')],
              [Input('search-button', 'n_clicks')],
              [State('stock-search', 'value')])
def lstm_prediction_and_recommendation(n_clicks, search_query):
    if n_clicks and search_query:
        # Fetch historical stock data using yfinance
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = load_historical_data(search_query, start_date, end_date)

        # Prepare data for LSTM model
        window_size = 60
        X, y, scaler = prepare_lstm_data(data, window_size)

        # Build and train LSTM model
        model = build_lstm_model(window_size)
        train_lstm_model(model, X, y)

        # Predict future stock prices using LSTM model
        future_window = 5  # Predicting for the next 5 days
        last_window = data['Close'].values[-window_size:]
        predicted_prices = []
        for i in range(future_window):
            X_pred = last_window[-window_size:].reshape(1, window_size, 1)
            next_price = predict_lstm_prices(model, X_pred, scaler)
            predicted_prices.append(next_price[0][0])
            last_window = np.append(last_window, next_price)

        # Generate buy/sell recommendation
        recommendation = generate_recommendation(predicted_prices)

        return f"Predicted Prices for Next 5 Days: {', '.join(map(str, predicted_prices))}", recommendation

    return "", ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
