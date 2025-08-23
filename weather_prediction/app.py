from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from data_generator import generate_weather_data
from model import WeatherPredictor
import requests
import os

app = Flask(__name__)
CORS(app)

# Global model variable
weather_model = None

# OpenWeatherMap API configuration - now uses environment variable
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '8f38a492cf893447c3181c9289354561')
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"


class SimpleFallbackModel:
    """Simple rule-based weather prediction as fallback"""
    def __init__(self):
        self.is_trained = True
    
    def predict(self, temperature, humidity, pressure, wind_speed, cloud_cover):
        """Simple rule-based prediction"""
        # Basic weather prediction logic
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
            prediction = "Partly Cloudy"
            probabilities = {"Partly Cloudy": 0.5, "Cloudy": 0.3, "Sunny": 0.2}
        
        return prediction, probabilities
    
def initialize_model():
    """Initialize and train the model with fallback"""
    global weather_model
    print("="*50)
    print("🔄 Starting model initialization...")
    print("="*50)
    
    try:
        print("Step 1: Testing imports...")
        from data_generator import generate_weather_data
        print("✅ data_generator imported successfully")
        
        from model import WeatherPredictor
        print("✅ WeatherPredictor imported successfully")
        
        print("Step 2: Generating weather data (200 records)...")
        df = generate_weather_data(200)  # Reduced from 1000
        print(f"✅ Generated {len(df)} weather records")
        
        print("Step 3: Creating WeatherPredictor instance...")
        weather_model = WeatherPredictor()
        print("✅ WeatherPredictor instance created")
        
        print("Step 4: Training model...")
        results = weather_model.train(df)
        print(f"✅ Model trained successfully!")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        
        print("="*50)
        print("🎉 ML MODEL INITIALIZATION COMPLETE!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ ML Model failed: {e}")
        print("🔄 Falling back to simple rule-based model...")
        
        try:
            weather_model = SimpleFallbackModel()
            print("✅ Fallback model initialized successfully!")
            print("="*50)
            print("🎉 FALLBACK MODEL ACTIVE!")
            print("="*50)
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {fallback_error}")
            weather_model = None
    
    # Final status check
    if weather_model is None:
        print("="*50)
        print("❌ ALL MODELS FAILED!")
        print("="*50)
    else:
        print(f"✅ Final status - Model ready: {hasattr(weather_model, 'is_trained') and weather_model.is_trained}")

@app.route('/')
def home():
    """Serve the main webpage"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for weather prediction"""
    try:
        if weather_model is None:
            return jsonify({
                'success': False,
                'error': 'Weather prediction model is not available'
            }), 500
            
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind_speed = float(data['wind_speed'])
        cloud_cover = float(data['cloud_cover'])
        
        # Validate ranges
        if not (0 <= humidity <= 100):
            return jsonify({
                'success': False,
                'error': 'Humidity must be between 0 and 100'
            }), 400
            
        if not (0 <= cloud_cover <= 100):
            return jsonify({
                'success': False,
                'error': 'Cloud cover must be between 0 and 100'
            }), 400
            
        if wind_speed < 0:
            return jsonify({
                'success': False,
                'error': 'Wind speed cannot be negative'
            }), 400
        
        # Make prediction
        prediction, probabilities = weather_model.predict(
            temperature, humidity, pressure, wind_speed, cloud_cover
        )
        
        # Convert probabilities to percentages
        prob_percentages = {k: round(v * 100, 1) for k, v in probabilities.items()}
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': prob_percentages
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input values: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/get-live-weather')
def get_live_weather():
    """Get live weather data for a city"""
    city = request.args.get('city', 'London')  # Default to London
    
    # Validate city input
    if not city or not city.strip():
        return jsonify({
            'success': False,
            'error': 'City name is required'
        })
    
    city = city.strip()
    
    try:
        # Make API request to OpenWeatherMap
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        
        print(f"Fetching weather for: {city}")  # Debug log
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        
        print(f"API Response Status: {response.status_code}")  # Debug log
        
        if response.status_code == 200:
            data = response.json()
            print(f"API Response Data: {data}")  # Debug log
            
            # Extract weather parameters with error handling
            try:
                weather_data = {
                    'success': True,
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'temperature': round(data['main']['temp'], 1),
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': round(data['wind'].get('speed', 0) * 3.6, 1),  # Convert m/s to km/h
                    'cloud_cover': data['clouds']['all'],
                    'description': data['weather'][0]['description'],
                    'icon': data['weather'][0]['icon'],
                    'feels_like': round(data['main']['feels_like'], 1),
                    'visibility': round(data.get('visibility', 0) / 1000, 1) if data.get('visibility') else 0  # Convert to km
                }
                
                return jsonify(weather_data)
                
            except KeyError as e:
                return jsonify({
                    'success': False,
                    'error': f'Missing data in weather response: {str(e)}'
                })
                
        elif response.status_code == 401:
            return jsonify({
                'success': False,
                'error': 'Invalid API key. Please check your OpenWeatherMap API key.'
            })
        elif response.status_code == 404:
            return jsonify({
                'success': False,
                'error': f'City "{city}" not found. Please check the spelling and try again.'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Weather service error: {response.status_code}'
            })
            
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Request timeout. Please try again.'
        })
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': 'Unable to connect to weather service. Please check your internet connection.'
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Weather API request error: {str(e)}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if weather_model and hasattr(weather_model, 'is_trained') and weather_model.is_trained:
        return jsonify({
            'trained': True,
            'features': ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        })
    return jsonify({'trained': False})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    initialize_model()
    
    if weather_model is None:
        print("⚠️  WARNING: Model failed to initialize!")
    
    print("\n" + "="*50)
    print("🌤️  WEATHER PREDICTION WEB APP")
    print("="*50)
    print("📝 API Key Status:")
    if WEATHER_API_KEY == "8f38a492cf893447c3181c9289354561":
        print("   ⚠️  Using default/fallback API key")
    else:
        print("   ✅ Using environment variable API key")
    print("="*50)
    
    # Use environment variable for port (required for deployment)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Server starting on port: {port}")
    print("="*50)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)