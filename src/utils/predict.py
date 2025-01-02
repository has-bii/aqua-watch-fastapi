import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from io import StringIO

def init_predict(csv_string: str):
    """
    Predict temperatures for the next 24 hours using a CSV string.
    
    Parameters:
        csv_string (str): A CSV string with columns 'id', 'temp', 'created_at', and 'env_id'.
        threshold (float): Temperature threshold for binary classification. Defaults to the mean temperature.
    
    Returns:
        dict: A dictionary containing predictions, model metrics, and retrained predictions.
    """
    # Load the CSV string into a DataFrame
    data = pd.read_csv(StringIO(csv_string))
    
    # Validate input data
    if data.empty:
        raise ValueError("Input CSV string contains no data.")
    if not {"temp", "created_at"}.issubset(data.columns):
        raise ValueError("Missing required columns: 'temp', 'created_at'.")
    
    # Drop unused columns
    data = data.drop(["id", "env_id"], axis=1, errors='ignore')
    
    # Parse 'created_at' into datetime
    data['created_at'] = pd.to_datetime(data['created_at'], utc=True)
    data['year'] = data['created_at'].dt.year
    data['month'] = data['created_at'].dt.month
    data['day'] = data['created_at'].dt.day
    data['hour'] = data['created_at'].dt.hour
    data['minute'] = data['created_at'].dt.minute
    data['second'] = data['created_at'].dt.second
    
    # Convert 'created_at' to numeric values (seconds since the first timestamp)
    data['timestamp'] = (data['created_at'] - data['created_at'].min()).dt.total_seconds()
    
    # Independent variable (timestamp) and dependent variable (temperature)
    X = pd.DataFrame(data.drop(['created_at', 'temp'], axis=1))
    Y = data['temp'].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    metrics = {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
    }
    
    # Retrain the model on the full dataset
    model.fit(X_scaled, Y)
    
    # Get the current timestamp (use the current date and time)
    current_timestamp = pd.Timestamp.now(tz='UTC')
    
    # Predict for the next 24 hours (hourly intervals from the current timestamp)
    future_timestamps = np.arange(current_timestamp.timestamp() + 3600, current_timestamp.timestamp() + 3600 * 25, 3600).reshape(-1, 1)
    
    # Generate future times, starting from the current date
    future_times = [
        current_timestamp + pd.Timedelta(hours=offset)
        for offset in range(1, 25)  # next 24 hours
    ]
    
    # Create future data with the same columns as used in training
    future_data = pd.DataFrame({
        'timestamp': future_timestamps.flatten(),
        'year': [time.year for time in future_times],
        'month': [time.month for time in future_times],
        'day': [time.day for time in future_times],
        'hour': [time.hour for time in future_times],
        'minute': [time.minute for time in future_times],
        'second': [time.second for time in future_times],
    })
    
    future_data = future_data[X.columns]
    
    # Scale the future data
    future_data_scaled = scaler.transform(future_data)
    
    # Predict the temperature for the next 24 hours
    retrained_predictions = model.predict(future_data_scaled)
    
    # Generate the result list as dictionaries with time (UTC) and retrained predicted temperature
    future_results = [
        {
            "time": time.strftime('%Y-%m-%d %H:%M:%S') + " +00:00",  # Explicitly mark UTC timezone
            "temp": temp
        }
        for time, temp in zip(future_times, retrained_predictions)
    ]
    
    return {
        "metrics": metrics,
        "predictions": future_results
    }
