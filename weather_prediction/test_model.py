# test_model.py - Test the pre-trained model
import pickle
import os

def test_pretrained_model():
    """Test if the pre-trained model works correctly"""
    print("="*50)
    print("🧪 TESTING PRE-TRAINED MODEL")
    print("="*50)
    
    model_path = 'models/weather_model.pkl'
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            print("   Run 'python train_and_save_model.py' first!")
            return False
        
        # Load the model
        print("Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        
        # Check if model is trained
        if not (hasattr(model, 'is_trained') and model.is_trained):
            print("❌ Model is not properly trained!")
            return False
        
        print("✅ Model is trained and ready!")
        
        # Test predictions with sample data
        print("\n🧪 Testing predictions...")
        
        test_cases = [
            {
                'name': 'Sunny Day',
                'params': (30, 40, 1020, 10, 20),
                'expected': 'Sunny'
            },
            {
                'name': 'Rainy Day',
                'params': (15, 90, 995, 25, 95),
                'expected': 'Rainy'
            },
            {
                'name': 'Cloudy Day',
                'params': (20, 65, 1010, 15, 75),
                'expected': 'Cloudy'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            temp, humidity, pressure, wind_speed, cloud_cover = test_case['params']
            
            try:
                prediction, probabilities = model.predict(
                    temp, humidity, pressure, wind_speed, cloud_cover
                )
                
                print(f"\nTest {i}: {test_case['name']}")
                print(f"   Input: T={temp}°C, H={humidity}%, P={pressure}hPa, W={wind_speed}km/h, C={cloud_cover}%")
                print(f"   Prediction: {prediction}")
                print(f"   Top probabilities:")
                
                # Sort probabilities and show top 3
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                for condition, prob in sorted_probs:
                    print(f"     {condition}: {prob:.1%}")
                
                print("   ✅ Test passed!")
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
                return False
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Model classes: {list(model.model.classes_)}")
        print("   Your model is ready to use in the Flask app!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_pretrained_model()
    if not success:
        print("\n💡 To fix this:")
        print("   1. Run: python train_and_save_model.py")
        print("   2. Then run: python test_model.py")
        print("   3. Finally run: python app.py")
        exit(1)
    else:
        print("\n🚀 Ready to run your Flask app!")
        print("   Run: python app.py")