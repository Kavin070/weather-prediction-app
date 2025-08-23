import pandas as pd
import numpy as np

def generate_weather_data(n_samples=1000):
    """Generate synthetic weather data for training"""
    np.random.seed(42)
    
    data = {
        'temperature': np.random.normal(20, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'pressure': np.random.normal(1013, 20, n_samples),
        'wind_speed': np.random.exponential(5, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
    }
    
    # Create weather conditions based on features
    conditions = []
    for i in range(n_samples):
        temp = data['temperature'][i]
        humidity = data['humidity'][i]
        pressure = data['pressure'][i]
        clouds = data['cloud_cover'][i]
        
        if humidity > 80 and clouds > 70 and pressure < 1005:
            condition = 'Rainy'
        elif clouds < 20 and temp > 25:
            condition = 'Sunny'
        elif clouds > 80 and temp < 10:
            condition = 'Snowy'
        elif clouds > 50:
            condition = 'Cloudy'
        else:
            condition = 'Clear'
        
        conditions.append(condition)
    
    data['weather_condition'] = conditions
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the data generator
    df = generate_weather_data(100)
    print("Sample data generated:")
    print(df.head())
    print(f"\nWeather conditions: {df['weather_condition'].unique()}")