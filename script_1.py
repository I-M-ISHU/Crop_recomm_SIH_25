import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


np.random.seed(42)
random.seed(42)


def generate_synthetic_data(crop_df, samples_per_crop=500):
    data = []
    labels = []
    
    for idx, crop in crop_df.iterrows():
        for _ in range(samples_per_crop):
            soil_ph = np.random.normal((crop['soil_ph_min'] + crop['soil_ph_max'])/2, 0.3)
            soil_ph = np.clip(soil_ph, crop['soil_ph_min']-0.5, crop['soil_ph_max']+0.5)
            
            temperature = np.random.normal((crop['temp_min'] + crop['temp_max'])/2, 3)
            temperature = np.clip(temperature, crop['temp_min']-5, crop['temp_max']+5)
            
            rainfall = np.random.normal((crop['rainfall_min'] + crop['rainfall_max'])/2, 100)
            rainfall = np.clip(rainfall, crop['rainfall_min']-200, crop['rainfall_max']+200)
            
            nitrogen = np.random.normal((crop['nitrogen_min'] + crop['nitrogen_max'])/2, 10)
            nitrogen = np.clip(nitrogen, crop['nitrogen_min']-20, crop['nitrogen_max']+20)
            
            phosphorus = np.random.normal((crop['phosphorus_min'] + crop['phosphorus_max'])/2, 5)
            phosphorus = np.clip(phosphorus, crop['phosphorus_min']-10, crop['phosphorus_max']+10)
            
            potassium = np.random.normal((crop['potassium_min'] + crop['potassium_max'])/2, 5)
            potassium = np.clip(potassium, crop['potassium_min']-10, crop['potassium_max']+10)
            
            humidity = np.random.normal((crop['humidity_min'] + crop['humidity_max'])/2, 5)
            humidity = np.clip(humidity, crop['humidity_min']-10, crop['humidity_max']+10)
            
            month = np.random.randint(crop['plant_month_start'], crop['plant_month_end']+1)
            
            data.append([
                soil_ph, temperature, rainfall, nitrogen, phosphorus, potassium, 
                humidity, month, crop['season'], crop['soil_type']
            ])
            labels.append(crop['crop_name'])
    
    return np.array(data), np.array(labels)


print("Generating synthetic training data...")
X, y = generate_synthetic_data(crop_df, samples_per_crop=300)

feature_names = ['soil_ph', 'temperature', 'rainfall', 'nitrogen', 'phosphorus', 
                'potassium', 'humidity', 'month', 'season', 'soil_type']


train_df = pd.DataFrame(X, columns=feature_names)
train_df['crop'] = y

print(f"Training data shape: {train_df.shape}")
print("\nSample of training data:")
print(train_df.head())
print(f"\nCrop distribution:")
print(train_df['crop'].value_counts().head())