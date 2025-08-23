#visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

class WeatherVisualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_distributions(self):
        """Plot weather condition distributions by features"""
        plt.figure(figsize=(15, 10))
        
        features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.df, x='weather_condition', y=feature)
            plt.title(f'{feature.title()} by Weather Condition')
            plt.xticks(rotation=45)
        
        # Correlation heatmap
        plt.subplot(2, 3, 6)
        df_numeric = self.df.copy()
        le = LabelEncoder()
        df_numeric['weather_condition_encoded'] = le.fit_transform(self.df['weather_condition'])
        correlation_matrix = df_numeric.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_importance_df):
        """Plot feature importance from trained model"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title('Feature Importance in Weather Prediction')
        plt.xlabel('Importance Score')
        plt.show()
        
    def plot_weather_distribution(self):
        """Plot the distribution of weather conditions"""
        plt.figure(figsize=(8, 6))
        weather_counts = self.df['weather_condition'].value_counts()
        plt.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Weather Conditions')
        plt.show()

if __name__ == "__main__":
    from data_generator import generate_weather_data
    
    df = generate_weather_data(1000)
    viz = WeatherVisualizer(df)
    viz.plot_weather_distribution()
    viz.plot_distributions()