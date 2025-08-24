from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import requests
import os

app = Flask(__name__)
CORS(app)

# Global model variable
weather_model = None

# OpenWeatherMap API configuration
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '8f38a492cf893447c3181c9289354561')
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

def train_model_at_runtime():
    """Always train fresh model to avoid pickle compatibility issues"""
    try:
        print("🤖 Training fresh ML model at runtime...")
        
        # Import ML dependencies
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        print("✅ All ML dependencies loaded successfully")
        
        # Generate training data directly (avoid import issues)
        print("📊 Generating training data...")
        np.random.seed(42)
        n_samples = 2000
        
        # Create synthetic weather data
        data = {
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'pressure': np.random.normal(1013, 20, n_samples),
            'wind_speed': np.random.exponential(5, n_samples),
            'cloud_cover': np.random.uniform(0, 100, n_samples),
        }
        
        # Generate weather conditions based on features
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
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} training samples")
        
        # Train model
        X = df[['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']]
        y = df['weather_condition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use smaller forest for faster training
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with {accuracy:.1%} accuracy")
        print(f"🎯 Available classes: {list(model.classes_)}")
        
        # Create wrapper class
        class RuntimeWeatherModel:
            def __init__(self, sklearn_model):
                self.model = sklearn_model
                self.is_trained = True
            
            def predict(self, temperature, humidity, pressure, wind_speed, cloud_cover):
                input_data = [[temperature, humidity, pressure, wind_speed, cloud_cover]]
                prediction = self.model.predict(input_data)[0]
                probabilities = self.model.predict_proba(input_data)[0]
                prob_dict = dict(zip(self.model.classes_, probabilities))
                return prediction, prob_dict
        
        return RuntimeWeatherModel(model)
        
    except ImportError as e:
        print(f"❌ Missing ML dependencies: {e}")
        return None
    except Exception as e:
        print(f"❌ Runtime training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

class SimpleFallbackModel:
    """Rule-based fallback model"""
    def __init__(self):
        self.is_trained = True
    
    def predict(self, temperature, humidity, pressure, wind_speed, cloud_cover):
        if cloud_cover > 80 and humidity > 85:
            prediction = "Rainy"
            probabilities = {"Rainy": 0.8, "Cloudy": 0.15, "Sunny": 0.05}
        elif cloud_cover > 60:
            prediction = "Cloudy"
            probabilities = {"Cloudy": 0.6, "Rainy": 0.25, "Sunny": 0.15}
        elif temperature > 25 and cloud_cover < 30:
            prediction = "Sunny"
            probabilities = {"Sunny": 0.7, "Cloudy": 0.25, "Rainy": 0.05}
        else:
            prediction = "Clear"
            probabilities = {"Clear": 0.5, "Cloudy": 0.3, "Sunny": 0.2}
        
        return prediction, probabilities

def initialize_model():
    """Initialize weather prediction model"""
    global weather_model
    print("="*60)
    print("🔄 INITIALIZING WEATHER PREDICTION MODEL")
    print("="*60)
    
    # First try: Load pre-trained model (if exists and compatible)
    model_path = 'models/weather_model.pkl'
    if os.path.exists(model_path):
        try:
            print(f"📁 Attempting to load: {model_path}")
            with open(model_path, 'rb') as f:
                weather_model = pickle.load(f)
            
            # Test the model
            test_pred, test_probs = weather_model.predict(20, 60, 1010, 10, 50)
            print(f"✅ Pre-trained model loaded successfully!")
            print(f"🎯 Test prediction: {test_pred}")
            
            print("="*60)
            print("🎉 PRE-TRAINED ML MODEL ACTIVE!")
            print("="*60)
            return
            
        except Exception as e:
            print(f"❌ Pre-trained model failed: {e}")
    
    # Second try: Train fresh model at runtime
    print("🤖 Training fresh model at runtime...")
    weather_model = train_model_at_runtime()
    
    if weather_model is not None:
        print("="*60)
        print("🎉 RUNTIME-TRAINED ML MODEL ACTIVE!")
        print("="*60)
        return
    
    # Final fallback: Simple rule-based model
    print("🔄 Using simple rule-based fallback...")
    weather_model = SimpleFallbackModel()
    print("="*60)
    print("⚠️  FALLBACK MODEL ACTIVE!")
    print("="*60)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if weather_model is None:
            return jsonify({
                'success': False,
                'error': 'Weather prediction model is not available'
            }), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind_speed = float(data['wind_speed'])
        cloud_cover = float(data['cloud_cover'])
        
        # Validate ranges
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be between 0 and 100'}), 400
        if not (0 <= cloud_cover <= 100):
            return jsonify({'success': False, 'error': 'Cloud cover must be between 0 and 100'}), 400
        if wind_speed < 0:
            return jsonify({'success': False, 'error': 'Wind speed cannot be negative'}), 400
        
        prediction, probabilities = weather_model.predict(temperature, humidity, pressure, wind_speed, cloud_cover)
        prob_percentages = {k: round(v * 100, 1) for k, v in probabilities.items()}
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': prob_percentages
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

@app.route('/get-live-weather')
def get_live_weather():
    city = request.args.get('city', 'London')
    if not city or not city.strip():
        return jsonify({'success': False, 'error': 'City name is required'})
    
    city = city.strip()
    
    try:
        params = {'q': city, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'success': True,
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': round(data['main']['temp'], 1),
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': round(data['wind'].get('speed', 0) * 3.6, 1),
                'cloud_cover': data['clouds']['all'],
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'feels_like': round(data['main']['feels_like'], 1),
                'visibility': round(data.get('visibility', 0) / 1000, 1) if data.get('visibility') else 0
            }
            return jsonify(weather_data)
        elif response.status_code == 404:
            return jsonify({'success': False, 'error': f'City "{city}" not found'})
        else:
            return jsonify({'success': False, 'error': f'Weather service error: {response.status_code}'})
            
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Request timeout. Please try again.'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'Unable to connect to weather service'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/model-info')
def model_info():
    if weather_model and hasattr(weather_model, 'is_trained') and weather_model.is_trained:
        model_type = "Machine Learning" if hasattr(weather_model, 'model') else "Rule-based"
        return jsonify({
            'trained': True,
            'type': model_type,
            'features': ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        })
    return jsonify({'trained': False})

# Initialize model when module loads
initialize_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    print(f"🌐 Server starting on port: {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)