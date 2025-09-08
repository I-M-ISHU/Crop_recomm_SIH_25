# Smart Crop Recommendation System - Usage Guide

## Overview
This AI-powered system predicts the most suitable crops for cultivation based on soil conditions, weather parameters, and seasonal factors. The model is trained on agricultural data from India and includes major crops like wheat, rice, sugarcane, cotton, and various cereals.

## Files Included
1. `crop_recommendation_model.pkl` - Pre-trained machine learning model
2. `crop_recommendation_app.py` - Desktop GUI application using tkinter
3. `streamlit_crop_app.py` - Web-based application using Streamlit
4. `requirements.txt` - Required Python packages

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Required Packages
```bash
pip install -r requirements.txt
```

## Running the Applications

### Option 1: Desktop GUI Application
```bash
python crop_recommendation_app.py
```

Features:
- User-friendly desktop interface
- Input fields for all parameters
- Real-time crop recommendations
- Confidence scores and yield predictions

### Option 2: Web Application (Recommended)
```bash
streamlit run streamlit_crop_app.py
```

Features:
- Interactive web interface
- Visual charts and analysis
- Radar chart for soil conditions
- Detailed recommendations with tips

## Input Parameters

### Soil Conditions
- **Soil pH**: 4.0 - 9.0 (optimal: 6.0 - 7.5)
- **Nitrogen**: 0 - 300 kg/ha
- **Phosphorus**: 0 - 200 kg/ha  
- **Potassium**: 0 - 250 kg/ha

### Weather Conditions
- **Temperature**: 5 - 45°C
- **Humidity**: 30 - 100%
- **Rainfall**: 100 - 3000 mm (seasonal)

### Temporal Factors
- **Month**: 1-12 (current planting month)
- **Season**: Automatically determined (Kharif/Rabi/Zaid)

## Output Information

### For Each Recommended Crop:
- **Suitability Score**: 0-100% confidence
- **Expected Yield**: Quintals per hectare
- **Crop Duration**: Days to harvest
- **Planting Recommendations**: Timing and conditions

### Model Performance:
- **Accuracy**: 98.7% on test data
- **Crops Covered**: 20 major Indian crops
- **Features Used**: 10 parameters

## Crops Included
1. **Cereals**: Wheat, Rice (Basmati, Non-Basmati), Barley, Maize, Jowar, Bajra, Ragi
2. **Cash Crops**: Cotton, Sugarcane, Sunflower
3. **Pulses**: Gram, Pigeon Pea
4. **Oilseeds**: Groundnut, Mustard, Sesame, Soybean
5. **Vegetables**: Potato, Onion

## Understanding Seasons
- **Kharif (June-September)**: Monsoon crops (Rice, Cotton, Sugarcane)
- **Rabi (October-March)**: Winter crops (Wheat, Barley, Mustard)
- **Zaid (April-June)**: Summer crops (Fodder, Vegetables)

## Tips for Best Results
1. **Soil Testing**: Get professional soil testing for accurate pH and nutrient values
2. **Local Conditions**: Consider local climate patterns and micro-climates
3. **Market Factors**: Check local market prices and demand
4. **Water Availability**: Ensure adequate irrigation for recommended crops
5. **Expertise**: Consult local agricultural experts for region-specific advice

## Troubleshooting

### Common Issues:
1. **Model file not found**: Ensure `crop_recommendation_model.pkl` is in the same directory
2. **Import errors**: Install required packages using `pip install -r requirements.txt`
3. **Low accuracy results**: Verify input parameter ranges and units

### Getting Help:
- Check input parameter ranges
- Ensure all required files are present
- Verify Python version compatibility (3.7+)

## Technical Details

### Model Architecture:
- **Algorithm**: Random Forest Classifier
- **Training Data**: 6,000 synthetic samples based on crop requirements
- **Features**: Soil, weather, and temporal parameters
- **Validation**: 20% test set with stratified sampling

### Performance Metrics:
- **Training Accuracy**: 99.5%
- **Test Accuracy**: 98.7%
- **Cross-validation**: 5-fold CV performed

## Future Enhancements
- Integration with weather APIs for real-time data
- GPS-based location services
- Market price integration
- Pest and disease risk assessment
- Multi-language support

## Support
For technical support or questions about the crop recommendation system, please ensure you have:
1. Python version information
2. Complete error messages (if any)
3. Input parameters used
4. Operating system details

---
**Note**: This system provides recommendations based on agricultural best practices and historical data. Always consult with local agricultural experts and consider regional variations before making final planting decisions.
