#predictor.py
from model import WeatherPredictor

class InteractivePredictor:
    def __init__(self, model):
        self.model = model
        
    def get_user_input(self):
        """Get weather parameters from user"""
        try:
            print("\n" + "="*40)
            print("WEATHER PREDICTION INPUT")
            print("="*40)
            
            temp = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            pressure = float(input("Atmospheric pressure (hPa): "))
            wind_speed = float(input("Wind speed (km/h): "))
            cloud_cover = float(input("Cloud cover (%): "))
            
            return temp, humidity, pressure, wind_speed, cloud_cover
            
        except ValueError:
            print("Please enter valid numbers!")
            return None
            
    def show_prediction(self, params):
        """Show prediction results"""
        if params is None:
            return
            
        temp, humidity, pressure, wind_speed, cloud_cover = params
        
        prediction, probabilities = self.model.predict(
            temp, humidity, pressure, wind_speed, cloud_cover
        )
        
        print(f"\n{'='*40}")
        print("PREDICTION RESULTS")
        print(f"{'='*40}")
        print(f"Predicted Weather: {prediction}")
        print(f"\nConfidence Levels:")
        
        for condition, prob in sorted(probabilities.items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {condition}: {prob:.1%}")
            
    def run_interactive_session(self):
        """Run interactive prediction session"""
        while True:
            params = self.get_user_input()
            self.show_prediction(params)
            
            continue_pred = input("\nMake another prediction? (y/n): ").lower()
            if continue_pred != 'y':
                break

if __name__ == "__main__":
    from data_generator import generate_weather_data
    
    # Create and train model
    df = generate_weather_data(1000)
    model = WeatherPredictor()
    model.train(df)
    
    # Run interactive predictor
    predictor = InteractivePredictor(model)
    predictor.run_interactive_session()