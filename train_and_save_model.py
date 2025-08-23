# train_and_save_model.py
import pickle
import os
from data_generator import generate_weather_data
from model import WeatherPredictor

def train_and_save_model():
    """Train the model and save it to disk"""
    print("="*60)
    print("🚀 TRAINING AND SAVING WEATHER PREDICTION MODEL")
    print("="*60)
    
    try:
        # Generate training data
        print("Step 1: Generating training data...")
        df = generate_weather_data(2000)  # More data for better accuracy
        print(f"✅ Generated {len(df)} weather records")
        
        # Create and train model
        print("Step 2: Creating and training model...")
        model = WeatherPredictor()
        results = model.train(df)
        print(f"✅ Model trained with {results['accuracy']:.1%} accuracy")
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save the trained model
        model_path = 'models/weather_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Model saved to: {model_path}")
        
        # Print model performance
        print("\n📊 MODEL PERFORMANCE:")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        print(f"   Classes: {list(model.model.classes_)}")
        
        print("\n🎉 MODEL TRAINING COMPLETE!")
        print("   You can now run your Flask app with the pre-trained model.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    success = train_and_save_model()
    if not success:
        print("Training failed!")
        exit(1)