# Crop database banate hain - ye wahi data hai jo model train karne ke liye use hota hai
import pandas as pd
import numpy as np

# Comprehensive crop database - Indian agriculture ke liye optimized
crop_data = {
    # Crop ke naam yahan daal diye hain
    'crop_name': ['Wheat', 'Rice_Basmati', 'Rice_Non_Basmati', 'Rice_Short_Grain', 'Sugarcane',
                  'Maize', 'Soybean', 'Cotton', 'Groundnut', 'Mustard', 'Gram', 'Pigeon_Pea', 
                  'Sunflower', 'Sesame', 'Barley', 'Jowar', 'Bajra', 'Ragi', 'Potato', 'Onion'],
    
    # Soil pH ki requirements (minimum se maximum)
    'soil_ph_min': [6.0, 6.0, 6.0, 6.0, 6.0, 5.5, 6.0, 5.8, 6.0, 6.5, 6.0, 6.5, 6.0, 5.5, 6.0, 6.0, 6.5, 5.0, 4.8, 6.0],
    'soil_ph_max': [7.5, 7.5, 7.5, 7.5, 7.5, 7.0, 7.0, 8.0, 7.5, 7.5, 7.5, 7.5, 7.5, 8.0, 7.8, 8.5, 8.0, 6.5, 6.5, 7.5],
    
    # Temperature ki requirements (°C mein) - optimal range
    'temp_min': [15, 20, 20, 20, 20, 18, 20, 18, 22, 10, 10, 20, 20, 25, 12, 26, 25, 20, 15, 13],
    'temp_max': [25, 35, 35, 35, 35, 30, 30, 32, 30, 25, 25, 30, 25, 30, 22, 30, 35, 27, 20, 25],
    
    # Rainfall ki requirements (mm/season mein)
    'rainfall_min': [500, 1000, 1000, 1000, 1000, 600, 600, 600, 500, 250, 350, 600, 450, 300, 450, 450, 350, 800, 600, 600],
    'rainfall_max': [1000, 2500, 2500, 2500, 1800, 1200, 1200, 1200, 1250, 600, 650, 1500, 900, 650, 650, 900, 650, 1200, 1100, 1000],
    
    # Nitrogen ki requirements (kg/ha mein)
    'nitrogen_min': [100, 80, 80, 70, 120, 120, 30, 100, 20, 60, 15, 25, 60, 40, 80, 80, 80, 50, 150, 100],
    'nitrogen_max': [180, 150, 150, 120, 200, 200, 60, 150, 40, 100, 25, 40, 100, 60, 120, 120, 120, 80, 220, 150],
    
    # Phosphorus ki requirements (kg/ha mein)
    'phosphorus_min': [40, 40, 40, 30, 40, 60, 60, 50, 40, 40, 60, 60, 30, 25, 40, 40, 40, 30, 80, 50],
    'phosphorus_max': [80, 80, 80, 60, 80, 120, 120, 100, 80, 80, 120, 120, 60, 50, 80, 80, 80, 60, 120, 100],
    
    # Potassium ki requirements (kg/ha mein)
    'potassium_min': [40, 40, 40, 30, 80, 40, 40, 50, 75, 40, 20, 25, 50, 35, 40, 40, 40, 30, 150, 80],
    'potassium_max': [80, 80, 80, 60, 150, 80, 80, 100, 125, 80, 40, 50, 100, 70, 80, 80, 80, 60, 200, 120],
    
    # Humidity ki requirements (% mein)
    'humidity_min': [50, 70, 70, 70, 70, 60, 60, 50, 60, 60, 60, 60, 60, 65, 65, 60, 60, 70, 80, 65],
    'humidity_max': [70, 90, 90, 90, 85, 80, 80, 70, 80, 80, 80, 80, 70, 85, 75, 80, 80, 85, 90, 85],
    
    # Growing season (1=Kharif, 2=Rabi, 3=Zaid, 4=Perennial)
    'season': [2, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2],
    
    # Best planting months (1-12)
    'plant_month_start': [11, 6, 6, 6, 2, 6, 6, 5, 6, 10, 10, 6, 6, 6, 11, 6, 6, 6, 10, 10],
    'plant_month_end': [12, 8, 8, 8, 3, 7, 7, 6, 7, 12, 12, 7, 7, 7, 12, 7, 7, 7, 12, 12],
    
    # Soil type preference (1=Sandy, 2=Loamy, 3=Clay, 4=Alluvial, 5=Black)
    'soil_type': [2, 3, 4, 2, 2, 2, 2, 5, 2, 2, 2, 5, 2, 2, 2, 5, 1, 2, 2, 2],
    
    # Expected yield (quintals/hectare)
    'expected_yield': [45, 50, 45, 40, 800, 50, 25, 15, 25, 15, 12, 15, 18, 8, 35, 25, 20, 15, 250, 300],
    
    # Crop duration (days mein)
    'crop_duration': [120, 120, 120, 115, 365, 110, 100, 180, 110, 120, 120, 180, 110, 90, 120, 110, 85, 120, 90, 120]
}

# DataFrame bana diya taaki use karna easy ho jaye
crop_df = pd.DataFrame(crop_data)

print("Crop Database successfully create ho gaya!")
print(f"Total crops: {len(crop_df)}")
print("\nFirst 5 rows:")
print(crop_df.head())

print("\nCrop names:")
print(crop_df['crop_name'].tolist())

# CSV file mein save kar dete hain
crop_df.to_csv('crop_database.csv', index=False)
print("\nCrop database 'crop_database.csv' mein save ho gaya!")