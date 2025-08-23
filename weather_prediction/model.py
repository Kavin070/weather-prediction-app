#model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, df):
        """Train the weather prediction model"""
        # Prepare features and target
        X = df[['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']]
        y = df['weather_condition']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        # Get predictions for evaluation
        y_pred = self.model.predict(self.X_test)
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'report': classification_report(self.y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
    
    def predict(self, temperature, humidity, pressure, wind_speed, cloud_cover):
        """Make a prediction for given weather conditions"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
            
        input_data = [[temperature, humidity, pressure, wind_speed, cloud_cover]]
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Create probability dictionary
        prob_dict = dict(zip(self.model.classes_, probabilities))
        
        return prediction, prob_dict

if __name__ == "__main__":
    # Test the model
    from data_generator import generate_weather_data
    
    df = generate_weather_data(500)
    model = WeatherPredictor()
    results = model.train(df)
    
    print(f"Model Accuracy: {results['accuracy']:.3f}")
    print(f"Feature Importance:\n{results['feature_importance']}")