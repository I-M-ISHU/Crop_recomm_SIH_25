# Complete Model Training Script - Crop Recommendation System
# Ye complete script hai jo model train karta hai from scratch!

import pandas as pd
import numpy as np
import pickle
import random
import warnings
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import joblib

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Warning se pareshani mat lena bhai
warnings.filterwarnings('ignore')

print("🌾 Crop Recommendation Model Training Script 🌾")
print("="*60)

# STEP 1: CROP DATABASE CREATION
# =============================

print("\nSTEP 1: Creating Crop Database...")

# Crop database - Indian agriculture ke liye specially designed
crop_data = {
    'crop_name': ['Wheat', 'Rice_Basmati', 'Rice_Non_Basmati', 'Rice_Short_Grain', 'Sugarcane',
                  'Maize', 'Soybean', 'Cotton', 'Groundnut', 'Mustard', 'Gram', 'Pigeon_Pea', 
                  'Sunflower', 'Sesame', 'Barley', 'Jowar', 'Bajra', 'Ragi', 'Potato', 'Onion'],
    
    # Soil pH requirements
    'soil_ph_min': [6.0, 6.0, 6.0, 6.0, 6.0, 5.5, 6.0, 5.8, 6.0, 6.5, 6.0, 6.5, 6.0, 5.5, 6.0, 6.0, 6.5, 5.0, 4.8, 6.0],
    'soil_ph_max': [7.5, 7.5, 7.5, 7.5, 7.5, 7.0, 7.0, 8.0, 7.5, 7.5, 7.5, 7.5, 7.5, 8.0, 7.8, 8.5, 8.0, 6.5, 6.5, 7.5],
    
    # Temperature requirements (°C)
    'temp_min': [15, 20, 20, 20, 20, 18, 20, 18, 22, 10, 10, 20, 20, 25, 12, 26, 25, 20, 15, 13],
    'temp_max': [25, 35, 35, 35, 35, 30, 30, 32, 30, 25, 25, 30, 25, 30, 22, 30, 35, 27, 20, 25],
    
    # Rainfall requirements (mm/season)
    'rainfall_min': [500, 1000, 1000, 1000, 1000, 600, 600, 600, 500, 250, 350, 600, 450, 300, 450, 450, 350, 800, 600, 600],
    'rainfall_max': [1000, 2500, 2500, 2500, 1800, 1200, 1200, 1200, 1250, 600, 650, 1500, 900, 650, 650, 900, 650, 1200, 1100, 1000],
    
    # Nitrogen requirements (kg/ha)
    'nitrogen_min': [100, 80, 80, 70, 120, 120, 30, 100, 20, 60, 15, 25, 60, 40, 80, 80, 80, 50, 150, 100],
    'nitrogen_max': [180, 150, 150, 120, 200, 200, 60, 150, 40, 100, 25, 40, 100, 60, 120, 120, 120, 80, 220, 150],
    
    # Phosphorus requirements (kg/ha)
    'phosphorus_min': [40, 40, 40, 30, 40, 60, 60, 50, 40, 40, 60, 60, 30, 25, 40, 40, 40, 30, 80, 50],
    'phosphorus_max': [80, 80, 80, 60, 80, 120, 120, 100, 80, 80, 120, 120, 60, 50, 80, 80, 80, 60, 120, 100],
    
    # Potassium requirements (kg/ha)
    'potassium_min': [40, 40, 40, 30, 80, 40, 40, 50, 75, 40, 20, 25, 50, 35, 40, 40, 40, 30, 150, 80],
    'potassium_max': [80, 80, 80, 60, 150, 80, 80, 100, 125, 80, 40, 50, 100, 70, 80, 80, 80, 60, 200, 120],
    
    # Humidity requirements (%)
    'humidity_min': [50, 70, 70, 70, 70, 60, 60, 50, 60, 60, 60, 60, 60, 65, 65, 60, 60, 70, 80, 65],
    'humidity_max': [70, 90, 90, 90, 85, 80, 80, 70, 80, 80, 80, 80, 70, 85, 75, 80, 80, 85, 90, 85],
    
    # Growing season (1=Kharif, 2=Rabi, 3=Zaid, 4=Perennial)
    'season': [2, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2],
    
    # Best planting months
    'plant_month_start': [11, 6, 6, 6, 2, 6, 6, 5, 6, 10, 10, 6, 6, 6, 11, 6, 6, 6, 10, 10],
    'plant_month_end': [12, 8, 8, 8, 3, 7, 7, 6, 7, 12, 12, 7, 7, 7, 12, 7, 7, 7, 12, 12],
    
    # Soil type preference
    'soil_type': [2, 3, 4, 2, 2, 2, 2, 5, 2, 2, 2, 5, 2, 2, 2, 5, 1, 2, 2, 2],
    
    # Expected yield (quintals/hectare)
    'expected_yield': [45, 50, 45, 40, 800, 50, 25, 15, 25, 15, 12, 15, 18, 8, 35, 25, 20, 15, 250, 300],
    
    # Crop duration (days)
    'crop_duration': [120, 120, 120, 115, 365, 110, 100, 180, 110, 120, 120, 180, 110, 90, 120, 110, 85, 120, 90, 120]
}

# DataFrame banate hain
crop_df = pd.DataFrame(crop_data)

print(f"✅ Crop database created with {len(crop_df)} crops")

# STEP 2: SYNTHETIC DATA GENERATION
# =================================

print("\nSTEP 2: Generating Synthetic Training Data...")

# Random seed set kar diya - same results ke liye
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(crop_df, samples_per_crop=300):
    """
    Har crop ke liye realistic synthetic data banata hai
    Thoda noise add kar dete hain taaki real world conditions mimic ho
    """
    data = []
    labels = []
    
    for idx, crop in crop_df.iterrows():
        for _ in range(samples_per_crop):
            # Parameters generate kar rahe hain suitable ranges ke andar with some noise
            
            # Soil pH - crop ke around normal distribution
            soil_ph = np.random.normal((crop['soil_ph_min'] + crop['soil_ph_max'])/2, 0.3)
            soil_ph = np.clip(soil_ph, crop['soil_ph_min']-0.5, crop['soil_ph_max']+0.5)
            
            # Temperature - seasonal variation ke saath
            temperature = np.random.normal((crop['temp_min'] + crop['temp_max'])/2, 3)
            temperature = np.clip(temperature, crop['temp_min']-5, crop['temp_max']+5)
            
            # Rainfall - monsoon pattern consider karke
            rainfall = np.random.normal((crop['rainfall_min'] + crop['rainfall_max'])/2, 100)
            rainfall = np.clip(rainfall, crop['rainfall_min']-200, crop['rainfall_max']+200)
            
            # Nitrogen - soil fertility ke hisaab se
            nitrogen = np.random.normal((crop['nitrogen_min'] + crop['nitrogen_max'])/2, 10)
            nitrogen = np.clip(nitrogen, crop['nitrogen_min']-20, crop['nitrogen_max']+20)
            
            # Phosphorus - balanced nutrition ke liye
            phosphorus = np.random.normal((crop['phosphorus_min'] + crop['phosphorus_max'])/2, 5)
            phosphorus = np.clip(phosphorus, crop['phosphorus_min']-10, crop['phosphorus_max']+10)
            
            # Potassium - crop quality ke liye important
            potassium = np.random.normal((crop['potassium_min'] + crop['potassium_max'])/2, 5)
            potassium = np.clip(potassium, crop['potassium_min']-10, crop['potassium_max']+10)
            
            # Humidity - climate conditions
            humidity = np.random.normal((crop['humidity_min'] + crop['humidity_max'])/2, 5)
            humidity = np.clip(humidity, crop['humidity_min']-10, crop['humidity_max']+10)
            
            # Month - planting season ke andar randomly select
            month = np.random.randint(crop['plant_month_start'], crop['plant_month_end']+1)
            
            # Sab data array mein daal diya
            data.append([
                soil_ph, temperature, rainfall, nitrogen, phosphorus, potassium,
                humidity, month, crop['season'], crop['soil_type']
            ])
            labels.append(crop['crop_name'])
    
    return np.array(data), np.array(labels)

# Training data generate kar lete hain
X, y = generate_synthetic_data(crop_df, samples_per_crop=300)

# Feature names define kar diye hain
feature_names = ['soil_ph', 'temperature', 'rainfall', 'nitrogen', 'phosphorus',
                'potassium', 'humidity', 'month', 'season', 'soil_type']

# DataFrame mein convert kar diya
train_df = pd.DataFrame(X, columns=feature_names)
train_df['crop'] = y

print(f"✅ Generated {train_df.shape[0]} training samples")
print(f"✅ Features: {len(feature_names)}")

# STEP 3: DATA PREPARATION
# ========================

print("\nSTEP 3: Preparing Data for Machine Learning...")

# Features aur target separate kar diya
X = train_df.drop('crop', axis=1)
y = train_df['crop']

# Data split kar diya - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Features ko scale kar diya
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training set size: {X_train.shape}")
print(f"✅ Test set size: {X_test.shape}")


print("\nSTEP 4: Training Multiple Models...")

# Multiple models test kar rahe hain - best wala choose karenge
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),  # probability=True for predict_proba
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
    
    print(f"  ✅ {name} Accuracy: {accuracy:.4f}")

# Best model find kar lete hain
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# STEP 5: FINAL MODEL SELECTION (Random Forest)
# ==============================================

print("\nSTEP 5: Training Final Random Forest Model...")

# Random Forest usually best perform karta hai agricultural data ke liye
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Final Random Forest Accuracy: {final_accuracy:.4f}")

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# STEP 6: MODEL EVALUATION
# ========================

print("\nSTEP 6: Model Evaluation...")

# Classification report generate kar lete hain
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 7: MODEL SAVING
# ====================

print("\nSTEP 7: Saving the Trained Model...")

# Model aur sab components save kar dete hain
model_components = {
    'model': rf_model,
    'scaler': scaler,
    'crop_database': crop_df,
    'feature_names': feature_names,
    'accuracy': final_accuracy,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_samples': len(train_df),
    'num_crops': len(crop_df),
    'feature_importance': feature_importance
}

# File mein save kar diya
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model_components, f)

print(f"✅ Model saved successfully to 'crop_recommendation_model.pkl'")

# STEP 8: MODEL TESTING
# =====================

print("\nSTEP 8: Testing the Model with Sample Data...")

# Test cases banate hain
test_cases = [
    {
        'name': 'Wheat-friendly conditions (November)',
        'params': [6.5, 18, 750, 140, 60, 60, 65, 11, 2, 2]  # season=2 (Rabi), soil_type=2 (Loamy)
    },
    {
        'name': 'Rice-friendly conditions (July)', 
        'params': [6.2, 28, 1200, 100, 50, 50, 80, 7, 1, 3]  # season=1 (Kharif), soil_type=3 (Clay)
    },
    {
        'name': 'High rainfall monsoon (August)',
        'params': [7.0, 26, 1500, 120, 70, 80, 85, 8, 1, 4]  # season=1 (Kharif), soil_type=4 (Alluvial)
    }
]

for test_case in test_cases:
    print(f"\n🧪 Test: {test_case['name']}")
    
    # Prediction banate hain
    input_data = np.array([test_case['params']])
    probabilities = rf_model.predict_proba(input_data)[0]
    crop_names = rf_model.classes_
    
    # Top 3 recommendations nikaal rhe hain
    recommendations = []
    for i, crop in enumerate(crop_names):
        recommendations.append({
            'crop': crop,
            'confidence': probabilities[i],
            'score': probabilities[i] * 100
        })
    
    # Sort by confidence
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    print("Top 3 Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['crop']} - {rec['score']:.1f}% confidence")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("="*60)

print(f"\n📊 Final Statistics:")
print(f"Model Accuracy: {final_accuracy:.4f}")
print(f"Total Training Samples: {len(train_df)}")
print(f"Number of Crops: {len(crop_df)}")
print(f"Number of Features: {len(feature_names)}")
print(f"Model Type: Random Forest Classifier")
print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n📁 Files Created:")
print(f"✅ crop_recommendation_model.pkl - Trained model")
print(f"✅ crop_database.csv - Crop database")