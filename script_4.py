class CropRecommendationSystem:
    def __init__(self, model, scaler, crop_database, feature_names):
        self.model = model
        self.scaler = scaler
        self.crop_db = crop_database
        self.feature_names = feature_names
        
    def get_season_from_month(self, month):
        """Convert month to season"""
        if month in [6, 7, 8, 9]:
            return 1  # Kharif
        elif month in [10, 11, 12, 1]:
            return 2  # Rabi
        elif month in [3, 4, 5]:
            return 3  # Zaid
        else:
            return 2  # Default to Rabi
    
    def determine_soil_type(self, soil_ph, nitrogen, phosphorus, potassium):
        """Determine soil type based on nutrient content"""
        # Simple heuristic based on nutrient levels
        if nitrogen > 150 and phosphorus > 80 and potassium > 100:
            return 4  # Alluvial (very fertile)
        elif nitrogen > 100 and phosphorus > 60:
            return 5  # Black soil (cotton belt)
        elif nitrogen < 80 and phosphorus < 40:
            return 1  # Sandy (low nutrients)
        elif soil_ph > 7.0:
            return 3  # Clay (alkaline)
        else:
            return 2  # Loamy (balanced)
    
    def recommend_crop(self, soil_ph, temperature, rainfall, nitrogen, 
                      phosphorus, potassium, humidity, month, latitude=None, longitude=None):
        """
        Recommend the best crop based on input parameters
        """
        season = self.get_season_from_month(month)
        soil_type = self.determine_soil_type(soil_ph, nitrogen, phosphorus, potassium)
        
        input_data = np.array([[
            soil_ph, temperature, rainfall, nitrogen, phosphorus, 
            potassium, humidity, month, season, soil_type
        ]])
        
        probabilities = self.model.predict_proba(input_data)[0]
        crop_names = self.model.classes_
        
        recommendations = []
        for i, crop in enumerate(crop_names):
            recommendations.append({
                'crop': crop,
                'confidence': probabilities[i],
                'suitability_score': probabilities[i] * 100
            })
        
        recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
        
        top_recommendations = []
        for rec in recommendations[:3]:
            crop_info = self.crop_db[self.crop_db['crop_name'] == rec['crop']].iloc[0]
            
            suitability_factors = self._analyze_suitability(
                soil_ph, temperature, rainfall, nitrogen, phosphorus, 
                potassium, humidity, month, crop_info
            )
            
            top_recommendations.append({
                'crop': rec['crop'],
                'confidence': rec['confidence'],
                'suitability_score': rec['suitability_score'],
                'expected_yield': crop_info['expected_yield'],
                'crop_duration': crop_info['crop_duration'],
                'suitability_factors': suitability_factors
            })
        
        return {
            'recommendations': top_recommendations,
            'input_analysis': {
                'soil_ph': soil_ph,
                'temperature': temperature,
                'rainfall': rainfall,
                'season': season,
                'soil_type': soil_type,
                'month': month
            }
        }
    
    def _analyze_suitability(self, soil_ph, temperature, rainfall, nitrogen, 
                           phosphorus, potassium, humidity, month, crop_info):
        """Analyze why a crop is suitable"""
        factors = []
        
        if crop_info['soil_ph_min'] <= soil_ph <= crop_info['soil_ph_max']:
            factors.append(f"Soil pH ({soil_ph:.1f}) is optimal")
        else:
            factors.append(f"Soil pH ({soil_ph:.1f}) needs adjustment")
        
        if crop_info['temp_min'] <= temperature <= crop_info['temp_max']:
            factors.append(f"Temperature ({temperature}°C) is suitable")
        else:
            factors.append(f"Temperature ({temperature}°C) may be challenging")
        
        if crop_info['rainfall_min'] <= rainfall <= crop_info['rainfall_max']:
            factors.append(f"Rainfall ({rainfall}mm) is adequate")
        else:
            factors.append(f"Rainfall ({rainfall}mm) may need irrigation/drainage")
        
        if crop_info['nitrogen_min'] <= nitrogen <= crop_info['nitrogen_max']:
            factors.append("Nitrogen levels are good")
        else:
            factors.append("Nitrogen levels need adjustment")
            
        if crop_info['plant_month_start'] <= month <= crop_info['plant_month_end']:
            factors.append("Good planting time")
        else:
            factors.append("Consider different planting time")
        
        return factors

crop_system = CropRecommendationSystem(rf_model, scaler, crop_df, feature_names)

print("Crop Recommendation System created successfully!")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Trained on {len(crop_df)} crop varieties")
print(f"Features used: {', '.join(feature_names)}")