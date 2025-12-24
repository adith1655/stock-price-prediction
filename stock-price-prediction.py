import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    label.dropna(inplace=True)
    y = np.array(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response

# 1. Load the Data
filename = 'nifty50.csv' 
try:
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip() # Clean column names
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=True)
    else:
        # Fallback: Just reverse it blindly assuming it's Newest->Oldest
        df = df.iloc[::-1]
        
    print("Data sorted. Latest date in data:", df['Date'].iloc[-1])

except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    exit()

# 2. Select Target
if 'Close' in df.columns:
    forecast_col = 'Close'
elif 'Closing Price' in df.columns:
    forecast_col = 'Closing Price'
else:
    print("Error: Could not find 'Close' column.")
    exit()

forecast_out = 5
test_size = 0.2

# 3. Apply Machine Learning
try:
    X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)
    learner = LinearRegression()
    learner.fit(X_train, Y_train)

    score = learner.score(X_test, Y_test)
    forecast = learner.predict(X_lately)
    
    response = {}
    response['test_score'] = score
    response['forecast_set'] = forecast
    
    print("\n--- Corrected Results ---")
    print(response)

except Exception as e:
    print(f"An error occurred: {e}")