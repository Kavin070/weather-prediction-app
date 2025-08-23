#main.py
from data_generator import generate_weather_data
from model import WeatherPredictor
from visualizer import WeatherVisualizer
from predictor import InteractivePredictor

def main():
    print("🌤️  Weather Prediction ML Project")
    print("="*50)
    
    # Generate data
    print("📊 Generating weather data...")
    df = generate_weather_data(1000)
    print(f"Generated {len(df)} weather records")
    
    # Show basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Weather conditions: {list(df['weather_condition'].unique())}")
    
    # Train model
    print("\n🤖 Training machine learning model...")
    model = WeatherPredictor()
    results = model.train(df)
    
    print(f"✅ Model trained with {results['accuracy']:.1%} accuracy")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    viz = WeatherVisualizer(df)
    
    # Show options
    while True:
        print(f"\n{'='*50}")
        print("What would you like to do?")
        print("1. View data distributions")
        print("2. View feature importance")
        print("3. View weather condition distribution")
        print("4. Make weather predictions")
        print("5. Show model performance")
        print("6. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            viz.plot_distributions()
        elif choice == '2':
            viz.plot_feature_importance(results['feature_importance'])
        elif choice == '3':
            viz.plot_weather_distribution()
        elif choice == '4':
            predictor = InteractivePredictor(model)
            predictor.run_interactive_session()
        elif choice == '5':
            print(f"\n📊 MODEL PERFORMANCE")
            print(f"Accuracy: {results['accuracy']:.1%}")
            print(f"\nFeature Importance:")
            print(results['feature_importance'].to_string(index=False))
        elif choice == '6':
            print("👋 Thanks for using the Weather Predictor!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()