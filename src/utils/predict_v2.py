import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

def predict_v2(csv_string: str):
    """
    Predict future temperatures using a Random Forest model. If the model exists, load it; otherwise, train a new one.
    
    Parameters:
    - csv_string (str): A CSV string with columns 'id', 'temp', 'created_at', and 'env_id'.
    
    Returns:
    - predictions_json (dict): JSON structure containing timestamps, predictions, and accuracy metrics.
    """
    
    # Load the dataset
    data = pd.read_csv(StringIO(csv_string))
    data['created_at'] = pd.to_datetime(data['created_at'], utc=True)
    data = data.sort_values('created_at')
    data['created_at'] = data['created_at'].apply(lambda t: t.replace(second=0, microsecond=0))
    data['temp'] = data['temp'].round(2)

    # Create lagged features
    data['temp_lag_1'] = data['temp'].shift(1)
    data['temp_lag_2'] = data['temp'].shift(2)
    data['temp_lag_3'] = data['temp'].shift(3)
    data = data.dropna()

    # Define features and target
    features = ['temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'minute', 'hour', 'day']
    data['minute'] = data['created_at'].dt.minute
    data['hour'] = data['created_at'].dt.hour
    data['day'] = data['created_at'].dt.day
    X = data[features]
    y = data['temp']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    # Train the model
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Step 6: Prepare future data
    last_timestamp = data['created_at'].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=1440,
        freq='min',
        tz='UTC'
    )

    future_data = pd.DataFrame({
        'created_at': future_timestamps,
        'minute': [ts.minute for ts in future_timestamps],
        'hour': [ts.hour for ts in future_timestamps],
        'day': [ts.day for ts in future_timestamps],
    })

    # Initialize lagged features for future data
    future_data['temp_lag_1'] = data['temp'].iloc[-1]
    future_data['temp_lag_2'] = data['temp_lag_1'].iloc[-1]
    future_data['temp_lag_3'] = data['temp_lag_2'].iloc[-1]

    # Predict future temperatures
    predictions = []
    for i in range(len(future_data)):
        current_features = future_data.iloc[i][['temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'minute', 'hour', 'day']]
        prediction = round(rf_model.predict([current_features])[0], 2)
        predictions.append(prediction)
        if i + 1 < len(future_data):
            future_data.loc[i + 1, 'temp_lag_1'] = prediction
            future_data.loc[i + 1, 'temp_lag_2'] = future_data.loc[i, 'temp_lag_1']
            future_data.loc[i + 1, 'temp_lag_3'] = future_data.loc[i, 'temp_lag_2']

    df = pd.DataFrame({'timestamp': future_timestamps, 'temperature': predictions})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Calculate the R² and MSE on test data
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate average temperature for each hour
    average = []
    for x in range(24):
        average.append({"time": x, "avg_temp": round(df[df.timestamp.dt.hour == x]['temperature'].mean(), 2)})

    last_hour = future_timestamps[0].hour

    # Create JSON structure
    predictions_json = {
        "predictions": [
            {"timestamp": ts, "temperature": temp}
            for ts, temp in zip(future_timestamps, predictions)
        ],
        'averages': [*average[last_hour:], *average[:last_hour]],
        'accuracy': {
            'mse': mse,
            'r2': r2
        }
    }

    return predictions_json
