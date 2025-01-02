import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

<<<<<<< HEAD
def init_predict(csv_string: str):
    """
    Predict temperatures for the next 24 hours using a CSV string.
=======
def predict(csv_data):
    # Convert the CSV string into a DataFrame
    from io import StringIO
    data = pd.read_csv(StringIO(csv_data))
>>>>>>> 4543e2a2de93c874840eaca705c8321b44dfaaa8
    
    # Convert 'created_at' to datetime and then to ordinal form for regression
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['time_ordinal'] = data['created_at'].map(pd.Timestamp.toordinal)
    
    # Prepare features and target variable
    X = data['time_ordinal'].values.reshape(-1, 1)
    y = data['temp'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on testing set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Generate predictions for the next 24 hours
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    if datetime.now().minute != 0:
        now += timedelta(hours=1)
    future_times = [now + timedelta(hours=i) for i in range(24)]
    future_ordinals = np.array([t.toordinal() for t in future_times]).reshape(-1, 1)
    future_predictions = model.predict(future_ordinals)
    
    # Compile evaluation metrics
    evaluation_metrics = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2
    }
    
    # Format future predictions for output
    future_output = {
        'predictions': [{'date': str(time), 'temp': temp} for time, temp in zip(future_times, future_predictions)],
        'evaluation': evaluation_metrics
    }
    
    return future_output