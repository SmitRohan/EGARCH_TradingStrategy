#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, Input, Output, State
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from arch import arch_model
import datetime
import webbrowser
from threading import Timer
import dash_auth

stock_list = {
    'RELIANCE': 'RELIANCE.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'INFOSYS': 'INFY.NS',
    'TCS': 'TCS.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'AXIS BANK': 'AXISBANK.NS',
    'KOTAK BANK': 'KOTAKBANK.NS',
    'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'LARSEN & TOUBRO': 'LT.NS',
    'HCL TECHNOLOGIES': 'HCLTECH.NS',
    'SUN PHARMA': 'SUNPHARMA.NS',
    'MARUTI SUZUKI': 'MARUTI.NS',
    'MAHINDRA & MAHINDRA': 'M&M.NS',
    'ULTRATECH CEMENT': 'ULTRACEMCO.NS',
    'NTPC': 'NTPC.NS',
    'BAJAJ FINANCE': 'BAJFINANCE.NS',
    'TITAN': 'TITAN.NS',
    'OIL & NATURAL GAS CORP': 'ONGC.NS',
    'ADANI PORTS': 'ADANIPORTS.NS',
    'ADANI ENTERPRISES': 'ADANIENT.NS',
    'BHARAT ELECTRONICS': 'BEL.NS',
    'POWER GRID CORP': 'POWERGRID.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
    'WIPRO': 'WIPRO.NS',
    'COAL INDIA': 'COALINDIA.NS',
    'JSW STEEL': 'JSWSTEEL.NS',
    'BAJAJ AUTO': 'BAJAJ-AUTO.NS',
    'NESTLE INDIA': 'NESTLEIND.NS',
    'ASIAN PAINTS': 'ASIANPAINT.NS',
    'TATA STEEL': 'TATASTEEL.NS',
    'TRENT': 'TRENT.NS',
    'GRASIM INDUSTRIES': 'GRASIM.NS',
    'SBI LIFE INSURANCE': 'SBILIFE.NS',
    'SHRIRAM FINANCE': 'SHRIRAMFIN.NS',
    'CIPLA': 'CIPLA.NS',
    'TATA CONSUMER PRODUCTS': 'TATACONSUM.NS',
    'DR REDDY\'S LABORATORIES': 'DRREDDY.NS',
    'APOLLO HOSPITALS': 'APOLLOHOSP.NS',
    'BHARTI AIRTEL': 'BHARTIARTL.NS',
    'TECH MAHINDRA': 'TECHM.NS',
    'DIVI\'S LABORATORIES': 'DIVISLAB.NS',
    'BAJAJ FINSERV': 'BAJAJFINSV.NS',
    'INDUSIND BANK': 'INDUSINDBK.NS',
    'EICHER MOTORS': 'EICHERMOT.NS',
    'HDFC LIFE INSURANCE': 'HDFCLIFE.NS',
    'BRITANNIA INDUSTRIES': 'BRITANNIA.NS',
    'HERO MOTOCORP': 'HEROMOTOCO.NS',
    'UPL': 'UPL.NS',
    'TATA POWER': 'TATAPOWER.NS',
    'NIFTY 50': '^NSEI'
}

# stock_list = {
#     'RELIANCE': 'RELIANCE.NS',
#     'HDFC BANK': 'HDFCBANK.NS',
#     'INFOSYS': 'INFY.NS',
#     'TCS': 'TCS.NS',
#     'ICICI BANK': 'ICICIBANK.NS',
#     'SBIN': 'SBIN.NS',
#     'AXIS BANK': 'AXISBANK.NS',
#     'KOTAK BANK': 'KOTAKBANK.NS'
# }

VALID_USERNAME_PASSWORD_PAIRS = {
    'EGARCH': 'Smit@123'
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

server = app.server

app.title = 'NSE Stock Dashboard'


auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# Global styles
container_style = {
    'maxWidth': '1000px',
    'margin': '30px auto',
    'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    'color': '#222',
    'backgroundColor': '#f9f9f9',
    'padding': '20px',
    'borderRadius': '8px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
}

header_style = {
    'textAlign': 'center',
    'color': '#0077b6',
    'marginBottom': '25px',
    'fontWeight': '700',
    'fontSize': '28px'
}

dropdown_style = {
    'width': '300px',
    'marginBottom': '20px',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
    'borderRadius': '5px'
}

button_style = {
    'backgroundColor': '#0077b6',
    'color': 'white',
    'border': 'none',
    'padding': '10px 25px',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'fontWeight': '600',
    'fontSize': '16px',
    'marginBottom': '20px',
    'transition': 'background-color 0.3s ease',
}

button_style_hover = {
    **button_style,
    'backgroundColor': '#005f86',
}

tab_style = {
    'padding': '10px',
    'fontWeight': '600',
    'fontSize': '16px',
    'color': '#0077b6'
}

tab_selected_style = {
    'backgroundColor': '#0077b6',
    'color': 'white',
    'fontWeight': '700',
    'borderRadius': '5px',
    'padding': '10px'
}

page_1_layout = html.Div([
    html.H2("Stock Trend & RSI + MACD (NSE Stocks)", style=header_style),
    dcc.Dropdown(
        id='stock-selector',
        options=[{'label': name, 'value': ticker} for name, ticker in stock_list.items()],
        value='RELIANCE.NS',
        persistence=True,
        persisted_props=['value'],
        persistence_type='memory',
        style=dropdown_style,
        clearable=False
    ),
    dcc.Graph(id='stock-trend', config={'displayModeBar': False}),
    dcc.Graph(id='rsi-plot', config={'displayModeBar': False})
], style=container_style)

page_2_layout = html.Div([
    html.H2("EGARCH-Based Daily Strategy (NSE Stocks)", style=header_style),
    dcc.Dropdown(
        id='stock-selector-egarch',
        options=[{'label': name, 'value': ticker} for name, ticker in stock_list.items()],
        value='RELIANCE.NS',
        style=dropdown_style,
        persistence=True,
        persisted_props=['value'],
        persistence_type='memory',

        clearable=False
    ),
    html.Button('Run EGARCH Strategy', id='run-egarch', n_clicks=0, style=button_style),
    dcc.Loading(
        id="loading-egarch",
        type="circle",
        children=html.Div([
            dcc.Graph(id='volatility-plot', config={'displayModeBar': False}),
            dcc.Graph(id='signal-plot', config={'displayModeBar': False}),
            html.Div(id='next-day-prediction', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': '700', 'color': '#0077b6', 'textAlign': 'center'})
        ])
    )
], style=container_style)

page_3_layout = html.Div([
    html.H2("EGARCH Signal Summary Table", style=header_style),
    dcc.Dropdown(
        id='multi-stock-dropdown',
        options=[{'label': name, 'value': ticker} for name, ticker in stock_list.items()],
        value=['RELIANCE.NS', 'HDFCBANK.NS'],
        persistence=True,
        persisted_props=['value'],
        persistence_type='memory',

        multi=True,
        style=dropdown_style
    ),
    html.Button('Generate Summary Table', id='generate-table-btn', n_clicks=0, style=button_style),
    dcc.Loading(
        id="loading-table",
        type="circle",
        children=html.Div(id='summary-table-container')
    )
], style=container_style)

app.layout = html.Div([
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Daily Trend & RSI + MACD', value='tab-1', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='EGARCH Strategy', value='tab-2', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='EGARCH Summary Table', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        ],
        style={'maxWidth': '1000px', 'margin': '20px auto'}
    ),
    html.Div(id='tabs-content', style={'maxWidth': '1000px', 'margin': '0 auto'})
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return page_1_layout
    elif tab == 'tab-2':
        return page_2_layout
    elif tab == 'tab-3':
        return page_3_layout


@app.callback(
    [Output('stock-trend', 'figure'),
     Output('rsi-plot', 'figure')],
    Input('stock-selector', 'value')
)
def update_stock_graph(ticker='HDFCBANK.NS'):
    df = yf.download(ticker, period='1y')
    if df.empty:
        return go.Figure(), go.Figure()
    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(0)

    # Calculate RSI
    rsi_indicator = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi_indicator.rsi()

    # Calculate MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Candlestick Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(x=df['Date'],
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'],
                                  name='Candlestick'))
    fig1.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price',
                       plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9',
                       font=dict(color='#222'),
                       margin=dict(l=40, r=40, t=50, b=40))

    # RSI + MACD Plot
    fig2 = go.Figure()

    # RSI
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
    fig2.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70,
                   line=dict(color="red", width=1, dash="dash"))
    fig2.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30,
                   line=dict(color="green", width=1, dash="dash"))

    # MACD
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='black')))

    fig2.update_layout(title='RSI and MACD Indicators', xaxis_title='Date', yaxis_title='Value',
                       plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9',
                       font=dict(color='#222'),
                       margin=dict(l=40, r=40, t=50, b=40))

    return fig1, fig2


@app.callback(
    [Output('volatility-plot', 'figure'),
     Output('signal-plot', 'figure'),
     Output('next-day-prediction', 'children')],
    Input('run-egarch', 'n_clicks'),
    State('stock-selector-egarch', 'value')
)
def run_egarch_model(n_clicks, ticker):
    if n_clicks == 0:
        return go.Figure(), go.Figure(), ""

    end = datetime.datetime.today()
    start = end - pd.Timedelta(days=365)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return go.Figure(), go.Figure(), "No data available for selected stock."

    df = df.dropna()
    df['Returns'] = 100 * df['Close'].pct_change()
    returns = df['Returns'].dropna()

    model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    df = df.loc[returns.index]
    df['Volatility'] = res.conditional_volatility
    forecast = res.forecast(horizon=1)
    cond_vol = np.sqrt(forecast.variance.values[-1, :])[0]  # already in %
    cond_vol_percent = cond_vol

    low_vol = df['Volatility'].quantile(0.3)
    high_vol = df['Volatility'].quantile(0.7)
    df['Signal'] = np.where(df['Volatility'] < low_vol, 'Buy',
                            np.where(df['Volatility'] > high_vol, 'Sell', 'Hold'))
    threshold = np.percentile(df['Volatility'], 75)  # same scale
    Signal = "High Volatility" if cond_vol_percent > threshold else "Low Volatility"

    # Volatility Plot
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility'], mode='lines', name='Volatility', line=dict(color='#0077b6')))
    fig_vol.add_hline(y=threshold, line_dash='dash', line_color='red', annotation_text="75th Percentile Threshold",
                     annotation_position="top right")

    fig_vol.update_layout(title=f"{ticker} - EGARCH Predicted Volatility", xaxis_title='Date', yaxis_title='Volatility',
                          plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9',
                          font=dict(color='#222'),
                          margin=dict(l=40, r=40, t=50, b=40))

    # Signal Plot
    fig_sig = go.Figure()
    colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'}
    for signal in df['Signal'].unique():
        fig_sig.add_trace(go.Scatter(
            x=df[df['Signal'] == signal].index,
            y=df[df['Signal'] == signal]['Returns'],
            mode='markers',
            name=signal,
            marker=dict(color=colors[signal], size=8)
        ))
    fig_sig.update_layout(title=f"{ticker} - EGARCH Strategy Signals", xaxis_title='Date', yaxis_title='Returns (%)',
                          plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9',
                          font=dict(color='#222'),
                          margin=dict(l=40, r=40, t=50, b=40))

    # next_day_vol = res.forecast(horizon=1).variance.values[-1][0] ** 0.5
    next_day_msg = f"Next Day Volatility Prediction: {Signal} (Volatility = {cond_vol_percent:.2f}%)"

    return fig_vol, fig_sig, next_day_msg

@app.callback(
    Output('summary-table-container', 'children'),
    Input('generate-table-btn', 'n_clicks'),
    State('multi-stock-dropdown', 'value')
)
def generate_egarch_summary(n_clicks, selected_tickers):
    if n_clicks == 0 or not selected_tickers:
        return html.Div()

    summary_data = []

    for ticker in selected_tickers:
        end = datetime.datetime.today()
        start = end - pd.Timedelta(days=365)
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return go.Figure(), go.Figure(), "No data available for selected stock."
    
        df = df.dropna()
        df['Returns'] = 100 * df['Close'].pct_change()
        returns = df['Returns'].dropna()

        model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
        res = model.fit(disp='off')
        df = df.loc[returns.index]
        df['Volatility'] = res.conditional_volatility
        forecast = res.forecast(horizon=1)
        cond_vol = np.sqrt(forecast.variance.values[-1, :])[0]  # already in %
        cond_vol_percent = cond_vol

        last_close = float(df['Close'].iloc[-1])
        last_vol = float(df['Volatility'].iloc[-1])

        low_vol = df['Volatility'].quantile(0.3)
        high_vol = df['Volatility'].quantile(0.7)

        if cond_vol_percent < low_vol:
            signal = "Buy"
            vol_cat = "Low"
        elif cond_vol_percent > high_vol:
            signal = "Sell"
            vol_cat = "High"
        else:
            signal = "Hold"
            vol_cat = "Medium"
    
        summary_data.append({
            'Ticker': ticker,
            'Last Close Price': f"{last_close:.2f}",
            'EGARCH Signal': signal,
            'Today\'s Volatility': f"{last_vol:.2f}",
            'Forecasted Volatility': f"{cond_vol:.2f}",
            'Volatility Category': vol_cat
        })

    if not summary_data:
        return html.Div("No data available for selected stocks.", style={'color': 'red'})

    df_summary = pd.DataFrame(summary_data)

    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_summary.columns])),
        html.Tbody([
            html.Tr([
                html.Td(df_summary.iloc[i][col]) for col in df_summary.columns
            ]) for i in range(len(df_summary))
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'marginTop': '20px',
        'textAlign': 'center'
    })

    return table

# if __name__ == "__main__":
#     app.run_server(debug=True)

# def open_browser():
#     webbrowser.open_new("http://127.0.0.1:8050/")


# if __name__ == "__main__":
#     Timer(1, open_browser).start()  # Delay opening slightly to let server start
#     app.run(debug=True, use_reloader=False)

