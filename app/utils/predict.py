# import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def predict(csv_string: str):
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
    
    # 🚨 Remove Outliers using IQR
    Q1 = data['temp'].quantile(0.25)
    Q3 = data['temp'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Get outliers
    outliers = data[(data['temp'] < lower_bound) | (data['temp'] > upper_bound)]
    
    # Group anomalies by proximity (5-minute window)
    outliers['time_diff'] = outliers['created_at'].diff().fillna(pd.Timedelta(seconds=0))
    outliers['group'] = (outliers['time_diff'] > pd.Timedelta(hours=1)).cumsum()
    
    # Group Anomalies by Cluster
    anomaly_groups = []
    for group_id, group in outliers.groupby('group'):
        anomaly_groups.append({
            "start_time": group['created_at'].min().isoformat(),
            "end_time": group['created_at'].max().isoformat(),
            "data": [
                {
                    "id": row['id'],
                    "temp": row['temp'],
                    "date_time": row['created_at'].isoformat()
                }
                for _, row in group.iterrows()
            ]
        })
    
    # Filter outliers
    data = data[(data['temp'] >= lower_bound) & (data['temp'] <= upper_bound)]
    
    print("Lower Bound: ", lower_bound)
    print("Upper Bound: ", upper_bound)
    print("Length: ", len(outliers))
    print(outliers.head())
    
    # Create Lagged Features
    data['temp_lag_1'] = data['temp'].shift(1)
    data['temp_lag_2'] = data['temp'].shift(2)
    data['temp_lag_3'] = data['temp'].shift(3)
    data = data.dropna()

    # Define Features and Target
    data['minute'] = data['created_at'].dt.minute
    data['hour'] = data['created_at'].dt.hour
    data['day'] = data['created_at'].dt.day
    
    features = ['temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'minute', 'hour', 'day']
    X = data[features]
    y = data['temp']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    # Train the Model
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Step 6: Prepare Future Data
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

    # Initialize Lagged Features for Future Data
    future_data['temp_lag_1'] = data['temp'].iloc[-1]
    future_data['temp_lag_2'] = data['temp_lag_1'].iloc[-1]
    future_data['temp_lag_3'] = data['temp_lag_2'].iloc[-1]

    # Predict Future Temperatures
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

    # Calculate Model Accuracy Metrics
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    min_temp = data['temp'].min()
    max_temp = data['temp'].max()

    # Calculate Average Temperature per Hour
    average = []
    for x in range(24):
        avg = round(df[df.timestamp.dt.hour == x]['temperature'].mean(), 2)
        average.append({
            "time": x,
            "normalize": round(((avg - min_temp) / (max_temp - min_temp)), 2),
            "avg_temp": avg
        })

    last_hour = future_timestamps[0].hour
    average = [*average[last_hour:], *average[:last_hour]]

    # Create JSON Structure
    predictions_json = {
        "predictions": [
            {"timestamp": ts, "temperature": temp}
            for ts, temp in zip(future_timestamps, predictions)
        ],
        'averages': average,
        'accuracy': {
            'mse': mse,
            'r2': r2
        },
        'min': min_temp,
        'max': max_temp,
        'anomalies': anomaly_groups
    }

    # Visualization
    # plt.figure(figsize=(14, 7))
    # plt.plot(data['created_at'], data['temp'], label="Historical Data", color="blue")
    # plt.plot(future_timestamps, predictions, label="Future Predictions (Minute Interval)", color="red", linestyle=":")
    # plt.title("Temperature Predictions: Historical and Future Data")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Temperature")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("figure.png")
    # plt.close()

    return predictions_json