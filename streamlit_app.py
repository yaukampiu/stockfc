# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('My Stock Forecast App')

stocks = ('AAPL', 'GOOG', 'TSLA', 'TSM', 'VOO', 'BTC-USD')
selected_stock = st.selectbox('Select Stock for Prediction', stocks)

n_years = st.slider('Number of Years to Predict:', 1, 5)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Latest Stock Price')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Historical Stock Price - Trend', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast Stock Price')
st.write(forecast.tail())
    
st.write(f'Forecast Plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#st.write("Forecast Components")
#fig2 = m.plot_components(forecast)
#st.write(fig2)
