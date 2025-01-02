import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json
from sklearn.model_selection import train_test_split
from datetime import timedelta
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from io import StringIO

def predict_v2(csv_string: str):
    """
    Predict future temperatures using a Random Forest model. If the model exists, load it; otherwise, train a new one.
    
    Parameters:
    - csv_string (str): A CSV string with columns 'id', 'temp', 'created_at', and 'env_id'.
    - output_file (str): Path to save the future predictions in JSON format.
    
    Returns:
    - predictions_json (dict): JSON structure containing timestamps and predictions.
    """
    model_path = 'saved_model.pkl'

    # Step 1: Check if the model exists
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading the model...")
        rf_model = joblib.load(model_path)
    else:
        print(f"Model not found at {model_path}. Training a new model...")
        
        # Load the dataset
        data = data = pd.read_csv(StringIO(csv_string))
        data['created_at'] = pd.to_datetime(data['created_at'])
        data = data.sort_values('created_at')
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
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

        # Train the model
        rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)

        # Save the model
        joblib.dump(rf_model, model_path)
        print(f"Model trained and saved to {model_path}")

    # Load the dataset for predictions
    data = pd.read_csv(StringIO(csv_string))
    data['created_at'] = pd.to_datetime(data['created_at'])
    print("Length: ", len(data))
    data = data.sort_values('created_at')
    data['temp'] = data['temp'].round(2)

    # Create lagged features
    data['temp_lag_1'] = data['temp'].shift(1)
    data['temp_lag_2'] = data['temp'].shift(2)
    data['temp_lag_3'] = data['temp'].shift(3)
    data = data.dropna()

    # Step 6: Prepare future data
    last_timestamp = data['created_at'].iloc[-1]
    future_timestamps = [last_timestamp + timedelta(minutes=i) for i in range(1, 1440 + 1)]

    future_data = pd.DataFrame({
        'created_at': future_timestamps,
        'minute': [ts.minute for ts in future_timestamps],
        'hour': [ts.hour for ts in future_timestamps],
        'day': [ts.day for ts in future_timestamps],
    })

    # Initialize lagged features
    future_data['temp_lag_1'] = data['temp'].iloc[-1]
    future_data['temp_lag_2'] = data['temp_lag_1'].iloc[-1]
    future_data['temp_lag_3'] = data['temp_lag_2'].iloc[-1]

    # Predict future temperatures
    predictions = []
    for i in range(len(future_data)):
        current_features = future_data.iloc[i][['temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'minute', 'hour', 'day']]
        prediction = round(rf_model.predict([current_features])[0], 2)
        # prediction = rf_model.predict([current_features])[0]
        predictions.append(prediction)
        if i + 1 < len(future_data):
            future_data.loc[i + 1, 'temp_lag_1'] = prediction
            future_data.loc[i + 1, 'temp_lag_2'] = future_data.loc[i, 'temp_lag_1']
            future_data.loc[i + 1, 'temp_lag_3'] = future_data.loc[i, 'temp_lag_2']

    # Create JSON structure
    predictions_json = {
        "predictions": [
            {"timestamp": ts.strftime('%Y-%m-%d %H:%M:%S'), "temperature": temp}
            for ts, temp in zip(future_timestamps, predictions)
        ]
    }

    output_file = "predictions.json"

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(predictions_json, json_file, indent=4)

    print(f"Future predictions saved to {output_file}")

    # Visualization
    plt.figure(figsize=(14, 7))

    # Historical data
    plt.plot(data['created_at'], data['temp'], label="Historical Data", color="blue")

    # Future predictions
    plt.plot(future_timestamps, predictions, label=f"Future Predictions ({"minute".capitalize()} Interval)", color="red", linestyle=":")

    plt.title("Temperature Predictions: Historical and Future Data")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure.png")

    return predictions_json