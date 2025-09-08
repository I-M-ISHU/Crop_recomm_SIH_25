import pandas as pd
import numpy as np

def load_and_display_crop_database():
    """
    Crop database ko load kar ke details dikhata hai
    """
    print("🌾 Crop Database Loader 🌾")
    print("="*50)
    
    try:
        crop_df = pd.read_csv('crop_database.csv')
        
        print(f"\n✅ Crop database successfully loaded!")
        print(f"📊 Total crops: {len(crop_df)}")
        print(f"📋 Total features: {crop_df.shape[1]}")
        
        print(f"\n📈 Database Info:")
        print(crop_df.info())
        
        print(f"\n🌱 Available Crops:")
        for i, crop in enumerate(crop_df['crop_name'], 1):
            print(f"{i:2d}. {crop}")
        
        print(f"\n🔍 Sample Data (First 3 crops):")
        print(crop_df.head(3))
        
        print(f"\n📊 Statistical Summary:")
        numeric_cols = crop_df.select_dtypes(include=[np.number]).columns
        print(crop_df[numeric_cols].describe())
        
        season_names = {1: "Kharif (Monsoon)", 2: "Rabi (Winter)", 3: "Zaid (Summer)", 4: "Perennial"}
        print(f"\n🌞 Season Distribution:")
        season_counts = crop_df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"  {season_names.get(season, f'Season {season}')}: {count} crops")
        
        soil_names = {1: "Sandy", 2: "Loamy", 3: "Clay", 4: "Alluvial", 5: "Black"}
        print(f"\n🏔️ Soil Type Distribution:")
        soil_counts = crop_df['soil_type'].value_counts().sort_index()
        for soil_type, count in soil_counts.items():
            print(f"  {soil_names.get(soil_type, f'Type {soil_type}')}: {count} crops")
        
        print(f"\n📏 Parameter Ranges:")
        print(f"  Soil pH: {crop_df['soil_ph_min'].min():.1f} - {crop_df['soil_ph_max'].max():.1f}")
        print(f"  Temperature: {crop_df['temp_min'].min()}°C - {crop_df['temp_max'].max()}°C")
        print(f"  Rainfall: {crop_df['rainfall_min'].min()}mm - {crop_df['rainfall_max'].max()}mm")
        print(f"  Nitrogen: {crop_df['nitrogen_min'].min()} - {crop_df['nitrogen_max'].max()} kg/ha")
        print(f"  Phosphorus: {crop_df['phosphorus_min'].min()} - {crop_df['phosphorus_max'].max()} kg/ha")
        print(f"  Potassium: {crop_df['potassium_min'].min()} - {crop_df['potassium_max'].max()} kg/ha")
        print(f"  Humidity: {crop_df['humidity_min'].min()}% - {crop_df['humidity_max'].max()}%")
        
        print(f"\n🚜 Yield & Duration Info:")
        print(f"  Expected Yield: {crop_df['expected_yield'].min()} - {crop_df['expected_yield'].max()} quintals/ha")
        print(f"  Crop Duration: {crop_df['crop_duration'].min()} - {crop_df['crop_duration'].max()} days")
        print(f"  Average Yield: {crop_df['expected_yield'].mean():.1f} quintals/ha")
        print(f"  Average Duration: {crop_df['crop_duration'].mean():.1f} days")
        
        return crop_df
        
    except FileNotFoundError:
        print("❌ Error: crop_database.csv file nahi mila!")
        print("Pehle model training script run karo ya crop database create karo.")
        return None
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return None

def show_crop_details(crop_df, crop_name):
    """
    Specific crop ki details dikhata hai
    """
    if crop_df is None:
        return
        
    crop_info = crop_df[crop_df['crop_name'] == crop_name]
    if crop_info.empty:
        print(f"❌ Crop '{crop_name}' not found in database")
        return
    
    crop = crop_info.iloc[0]
    season_names = {1: "Kharif", 2: "Rabi", 3: "Zaid", 4: "Perennial"}
    soil_names = {1: "Sandy", 2: "Loamy", 3: "Clay", 4: "Alluvial", 5: "Black"}
    
    print(f"\n🌱 {crop_name} - Detailed Information")
    print("="*40)
    
    print(f"🌞 Growing Season: {season_names.get(crop['season'], 'Unknown')}")
    print(f"🏔️ Preferred Soil: {soil_names.get(crop['soil_type'], 'Unknown')}")
    print(f"📅 Planting Months: {crop['plant_month_start']} to {crop['plant_month_end']}")
    print(f"⏱️ Crop Duration: {crop['crop_duration']} days")
    print(f"🚜 Expected Yield: {crop['expected_yield']} quintals/hectare")
    
    print(f"\n📊 Optimal Conditions:")
    print(f"  Soil pH: {crop['soil_ph_min']} - {crop['soil_ph_max']}")
    print(f"  Temperature: {crop['temp_min']}°C - {crop['temp_max']}°C")
    print(f"  Rainfall: {crop['rainfall_min']} - {crop['rainfall_max']} mm")
    print(f"  Humidity: {crop['humidity_min']}% - {crop['humidity_max']}%")
    
    print(f"\n🧪 Nutrient Requirements (kg/ha):")
    print(f"  Nitrogen: {crop['nitrogen_min']} - {crop['nitrogen_max']}")
    print(f"  Phosphorus: {crop['phosphorus_min']} - {crop['phosphorus_max']}")
    print(f"  Potassium: {crop['potassium_min']} - {crop['potassium_max']}")

if __name__ == "__main__":
    crop_df = load_and_display_crop_database()
    
    if crop_df is not None:
        print(f"\n" + "="*50)
        print("🔍 Sample Crop Details:")
        
        show_crop_details(crop_df, 'Wheat')
        
        print(f"\n💡 Usage:")
        print(f"crop_df = pd.read_csv('crop_database.csv')")
        print(f"print(crop_df.head())")
        
        print(f"\n📋 Database ready for model training!")