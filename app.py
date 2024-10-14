import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# Set up Streamlit
st.title('Stock Price Prediction App')
st.sidebar.header('Input Stock Ticker')
stock_ticker = st.sidebar.text_input('Enter Stock Ticker (e.g. AAPL, GOOG)', 'AAPL')

start_date = st.sidebar.text_input('Enter a start date for Ticker (format YYYY-MM-DD)', '2010-01-01')
end_date = st.sidebar.text_input('Enter an end date for Ticker (format YYYY-MM-DD)', '2024-01-01')

# Initialize session state variables
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None
if 'prediction_date' not in st.session_state:
    st.session_state['prediction_date'] = None

# Validate date format and existence
def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


# Check if the entered dates are valid
start_dt = validate_date(start_date)
end_dt = validate_date(end_date)

if start_dt is None:
    st.error("Invalid start date! Please enter a valid date in the format YYYY-MM-DD.")
elif end_dt is None:
    st.error("Invalid end date! Please enter a valid date in the format YYYY-MM-DD.")
elif start_dt >= end_dt:
    st.error("Start date must be earlier than the end date.")
else:
    # Fetch stock data from Yahoo Finance
    @st.cache
    def load_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data

    # Button to load the data
    if st.button('Load Data'):
        st.session_state['stock_data'] = load_data(stock_ticker, start_date, end_date)
        st.session_state['prediction_date'] = None  # Reset the prediction date on new data load
        st.success(f'{stock_ticker} data loaded!')

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Check if data is loaded
    if st.session_state['stock_data'] is not None:
        st.subheader(f'{stock_ticker} Stock Data')
        st.write(st.session_state['stock_data'].tail())

        # Select prediction date after data is loaded
        st.session_state['prediction_date'] = st.date_input('Select a desired date for prediction', st.session_state['stock_data'].index[-1])
        sequence_length = 10  # Set your desired sequence length here

        if st.session_state['prediction_date'] is not None:
            stock_data = st.session_state['stock_data']

            # Preprocessing and creating sequences
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

            sequence_length = 30


            # Function to calculate classification labels (up/down)
            def get_class_labels(y):
                return [1 if y[i] > y[i - 1] else 0 for i in
                        range(1, len(y))]  # 1 for price increase, 0 for price decrease


            # Function to evaluate metrics
            def evaluate_metrics(y_test, y_pred):
                # Get binary labels for both actual and predicted values
                y_test_class = get_class_labels(y_test)
                y_pred_class = get_class_labels(y_pred)

                # Calculate metrics
                accuracy = accuracy_score(y_test_class, y_pred_class)
                precision = precision_score(y_test_class, y_pred_class)
                f1 = f1_score(y_test_class, y_pred_class)

                return accuracy, precision, f1

            def create_sequences(data, sequence_length=10):
                X, y = [], []
                for i in range(sequence_length, len(data)):
                    X.append(data[i-sequence_length:i])  # Previous `sequence_length` time steps
                    y.append(data[i, 3])  # Target: Close price at current time step
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled_data, sequence_length)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Assume scaler is defined somewhere in your code
            # Initialize your scaler and your model
            if 'scaler' not in st.session_state:
                st.session_state.scaler = MinMaxScaler()

            # CNN Model
            # Model Definitions with More Filters and Layers
            def cnn_model():
                model = Sequential()
                model.add(Conv1D(filters=128, kernel_size=3, activation='relu',
                                 input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))
                return model

            # LSTM Model
            def lstm_model():
                model = Sequential()
                model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(0.3))
                model.add(LSTM(100, return_sequences=False))
                model.add(Dropout(0.3))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))
                return model

            # Hybrid CNN-LSTM Model
            def hybrid_model():
                model = Sequential()
                model.add(Conv1D(filters=128, kernel_size=3, activation='relu',
                                 input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(100, return_sequences=True))
                model.add(Dropout(0.3))
                model.add(LSTM(100, return_sequences=False))
                model.add(Dropout(0.3))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))
                return model


            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # # When the button is clicked
            # if st.button('Predict Next Day Stock Price'):
            #     # Get the last available stock data for the prediction
            #     last_data = st.session_state['stock_data'].tail(sequence_length)  # Get the last `sequence_length` rows
            #     last_data_scaled = st.session_state.scaler.transform(
            #         last_data[['Open', 'High', 'Low', 'Close', 'Volume']])
            #
            #     # Create the input sequence for prediction
            #     last_sequence = last_data_scaled.reshape((1, sequence_length, last_data_scaled.shape[1]))
            #
            #     # Make the prediction using the last sequence
            #     next_day_pred = st.session_state.cnn.predict(last_sequence)
            #
            #     # Inverse transform the prediction to get the actual price
            #     next_day_price = st.session_state.scaler.inverse_transform([[next_day_pred[0][0], 0, 0, 0, 0]])[0][3]
            #
            #     st.write(f'Predicted Stock Price for the Next Day: ${next_day_price:.2f}')


            # Train and predict
            def train_and_predict(model, X_train, y_train, X_test):
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

                # model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=1)
                y_pred = model.predict(X_test)
                y_pred = scaler.inverse_transform([[0, 0, 0, pred[0], 0] for pred in y_pred])[:, 3]
                # Calculate evaluation metrics
                accuracy, precision, f1 = evaluate_metrics(y_test, y_pred)

                return y_pred, accuracy, precision, f1

            # Plot predictions
            def plot_predictions(y_test, y_pred, title):
                plt.figure(figsize=(12, 6))
                y_test_rescaled = scaler.inverse_transform([[0, 0, 0, val, 0] for val in y_test])[:, 3]
                plt.plot(y_test_rescaled, color='blue', label='Actual Stock Price')
                plt.plot(y_pred, color='red', label='Predicted Stock Price')
                plt.title(title)
                plt.xlabel('Time')
                plt.ylabel('Stock Price')
                plt.legend()
                st.pyplot(plt)


            # 3D plot predictions using dots
            def plot_predictions_3d(y_test, y_pred, title):
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Rescale y_test back to the original values
                y_test_rescaled = scaler.inverse_transform([[0, 0, 0, val, 0] for val in y_test])[:, 3]

                # Create a time axis for the X-axis
                time_axis = range(len(y_test_rescaled))

                # Plot actual stock prices with blue dots
                ax.scatter(time_axis, y_test_rescaled, time_axis, color='blue', label='Actual Stock Price', s=10)

                # Plot predicted stock prices with red dots
                ax.scatter(time_axis, y_pred, time_axis, color='red', label='Predicted Stock Price', s=10)

                # Labels for the axes
                ax.set_xlabel('Time')
                ax.set_ylabel('Stock Price')
                ax.set_zlabel('Index')

                ax.set_title(title)
                ax.legend()

                st.pyplot(fig)


            # 3D plot predictions using Plotly (interactive)
            def plot_predictions_3d_interactive(y_test, y_pred, title):
                # Rescale y_test back to the original values
                y_test_rescaled = scaler.inverse_transform([[0, 0, 0, val, 0] for val in y_test])[:, 3]

                # Create a time axis for the X-axis
                time_axis = list(range(len(y_test_rescaled)))

                # Create the Plotly figure
                fig = go.Figure()

                # Add actual stock prices as blue dots
                fig.add_trace(go.Scatter3d(
                    x=time_axis,
                    y=y_test_rescaled,
                    z=time_axis,
                    mode='markers',
                    marker=dict(size=5, color='blue'),
                    name='Actual Stock Price'
                ))

                # Add predicted stock prices as red dots
                fig.add_trace(go.Scatter3d(
                    x=time_axis,
                    y=y_pred,
                    z=time_axis,
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Predicted Stock Price'
                ))

                # Set layout properties for the figure
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title='Time',
                        yaxis_title='Stock Price',
                        zaxis_title='Index'
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                # Render the plot in Streamlit
                st.plotly_chart(fig)


            # Model selection
            cnn = cnn_model()
            lstm = lstm_model()
            hybrid = hybrid_model()

            if st.button('Train and Predict'):
                st.write(f"Training and Predicting for {st.session_state['prediction_date']} using CNN Model...")
                cnn_pred, cnn_accuracy, cnn_precision, cnn_f1 = train_and_predict(cnn, X_train, y_train, X_test)
                st.write("CNN Prediction Results", cnn_pred)
                st.write(
                    f"CNN Model - Accuracy: {cnn_accuracy:.2f}, Precision: {cnn_precision:.2f}, F1-Score: {cnn_f1:.2f}")
                plot_predictions(y_test, cnn_pred, 'CNN Model Prediction')
                # plot_predictions_3d(y_test, cnn_pred, 'CNN Model Prediction - 3d')
                plot_predictions_3d_interactive(y_test, cnn_pred, 'CNN Model Prediction - 3d interactive')

                st.write(f"Training and Predicting for {st.session_state['prediction_date']} using LSTM Model...")
                lstm_pred, lstm_accuracy, lstm_precision, lstm_f1 = train_and_predict(lstm, X_train, y_train, X_test)
                st.write("LSTM Prediction Results", lstm_pred)
                st.write(
                    f"LSTM Model - Accuracy: {lstm_accuracy:.2f}, Precision: {lstm_precision:.2f}, F1-Score: {lstm_f1:.2f}")
                plot_predictions(y_test, lstm_pred, 'LSTM Model Prediction')
                # plot_predictions_3d(y_test, lstm_pred, 'LSTM Model Prediction - 3d')
                plot_predictions_3d_interactive(y_test, lstm_pred, 'LSTM Model Prediction - 3d interactive')

                st.write(f"Training and Predicting for {st.session_state['prediction_date']} using Hybrid CNN-LSTM Model...")
                hybrid_pred, hybrid_accuracy, hybrid_precision, hybrid_f1 = train_and_predict(hybrid, X_train, y_train, X_test)
                st.write("Hybrid CNN-LSTM Prediction Results", hybrid_pred)
                st.write(
                    f"Hybrid CNN-LSTM Model - Accuracy: {hybrid_accuracy:.2f}, Precision: {hybrid_precision:.2f}, F1-Score: {hybrid_f1:.2f}")
                plot_predictions(y_test, hybrid_pred, 'Hybrid CNN-LSTM Model Prediction')
                # plot_predictions_3d(y_test, hybrid_pred, 'Hybrid CNN-LSTM Prediction - 3d')
                plot_predictions_3d_interactive(y_test, hybrid_pred, 'Hybrid CNN-LSTM Model Prediction - 3d interactive')