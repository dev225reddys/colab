import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to forecast multiple steps ahead
def forecast(model, initial_sequence, n_steps):
    sequence = initial_sequence.copy()
    forecasts = []
    for _ in range(n_steps):
        pred = model.predict(sequence[np.newaxis, :, :], verbose=0)[0]
        forecasts.append(pred)
        sequence = np.append(sequence[1:], pred[np.newaxis, :], axis=0)
    return np.array(forecasts)

# Function to build and train LSTM model
def build_and_train_model(X_train, y_train, seq_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 3)))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

# Read the CSV file
df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date')

# Start from May 2023 to avoid initial missing values
df = df.loc['5/1/23':]

# Split data: training up to August 2024, validation from September 2024 to January 2025
train_df = df.loc[:'8/1/24']
val_df = df.loc['9/1/24':'1/1/25']

# Scale the data
scaler = MinMaxScaler()
scaler.fit(train_df)
train_scaled = scaler.transform(train_df)
val_scaled = scaler.transform(val_df)

# Try different sequence lengths for tuning
sequence_lengths = [3, 6, 9]
best_mse = float('inf')
best_model = None
best_seq_length = None
best_forecast = None

for seq_length in sequence_lengths:
    # Prepare training sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    
    # Train the model
    model = build_and_train_model(X_train, y_train, seq_length)
    
    # Forecast for validation period (5 months: Sep 2024 to Jan 2025)
    initial_sequence = train_scaled[-seq_length:]
    forecast_scaled = forecast(model, initial_sequence, n_steps=5)
    forecast_val = scaler.inverse_transform(forecast_scaled)
    val_actual = val_df.values
    
    # Compute mean squared error for validation
    mse = np.mean((val_actual - forecast_val) ** 2)
    
    # Update best model if this MSE is lower
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_seq_length = seq_length
        
        # Forecast 9 months ahead (Sep 2024 to May 2025) with the best model
        best_forecast_scaled = forecast(model, initial_sequence, n_steps=9)
        best_forecast = scaler.inverse_transform(best_forecast_scaled)

# Generate forecast dates
last_date = train_df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=9, freq='MS')

# Extract forecast for February 2025 to May 2025 (last 4 months)
forecast_feb_to_may = best_forecast[5:]
forecast_dates_feb_to_may = forecast_dates[5:]

# Print forecast results
print("Forecast from February 2025 to May 2025:")
for date, values in zip(forecast_dates_feb_to_may, forecast_feb_to_may):
    print(f"{date.strftime('%m/%d/%y')}: "
          f"MNTs left legacy apo={values[0]:.2f}, "
          f"Myplan_shift_per={values[1]:.2f}, "
          f"Churn={values[2]:.2f}")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(15, 20))
variables = ['MNTs left legacy apo', 'Myplan_shift_per', 'Churn']

for i, var in enumerate(variables):
    axes[i].plot(train_df.index, train_df[var], label='Training Data', color='blue')
    axes[i].plot(val_df.index, val_df[var], label='Validation Data', color='green')
    axes[i].plot(forecast_dates, best_forecast[:, i], label='Forecast', color='red', linestyle='--')
    axes[i].set_title(f'{var} Over Time', fontsize=14)
    axes[i].set_xlabel('Date', fontsize=12)
    axes[i].set_ylabel(var, fontsize=12)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()

# Save to PDF
with PdfPages('forecast_results.pdf') as pdf:
    pdf.savefig(fig)
plt.close()

print(f"Best sequence length chosen: {best_seq_length} with validation MSE: {best_mse:.4f}")
