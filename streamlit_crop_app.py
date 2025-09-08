
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .recommendation-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the pre-trained model and components"""
    try:
        with open('crop_recommendation_model.pkl', 'rb') as f:
            components = pickle.load(f)
        return components
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'crop_recommendation_model.pkl' is available.")
        return None

def get_season_from_month(month):
    """Convert month to season"""
    if month in [6, 7, 8, 9]:
        return 1, "Kharif (Monsoon)"
    elif month in [10, 11, 12, 1]:
        return 2, "Rabi (Winter)"
    elif month in [3, 4, 5]:
        return 3, "Zaid (Summer)"
    else:
        return 2, "Rabi (Winter)"

def determine_soil_type(soil_ph, nitrogen, phosphorus, potassium):
    """Determine soil type based on nutrient content"""
    soil_types = {1: "Sandy", 2: "Loamy", 3: "Clay", 4: "Alluvial", 5: "Black"}

    if nitrogen > 150 and phosphorus > 80 and potassium > 100:
        return 4, soil_types[4]
    elif nitrogen > 100 and phosphorus > 60:
        return 5, soil_types[5]
    elif nitrogen < 80 and phosphorus < 40:
        return 1, soil_types[1]
    elif soil_ph > 7.0:
        return 3, soil_types[3]
    else:
        return 2, soil_types[2]

def create_radar_chart(soil_params):
    """Create radar chart for soil analysis"""
    categories = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity']

    # Normalize values for radar chart (0-100 scale)
    values = [
        (soil_params['soil_ph'] / 14) * 100,
        min((soil_params['nitrogen'] / 200) * 100, 100),
        min((soil_params['phosphorus'] / 150) * 100, 100),
        min((soil_params['potassium'] / 200) * 100, 100),
        min((soil_params['temperature'] / 40) * 100, 100),
        (soil_params['humidity'] / 100) * 100
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Conditions',
        line_color='#4caf50'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Soil & Weather Analysis",
        height=400
    )

    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Smart Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered crop selection based on soil and weather conditions</p>', unsafe_allow_html=True)

    # Load model
    components = load_model()
    if components is None:
        st.stop()

    model = components['model']
    crop_db = components['crop_database']
    accuracy = components['accuracy']

    # Sidebar for input parameters
    st.sidebar.header("📊 Input Parameters")

    # Soil parameters
    st.sidebar.subheader("🌱 Soil Conditions")
    soil_ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
    nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 0, 300, 120, 5)
    phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 0, 200, 60, 5)
    potassium = st.sidebar.slider("Potassium (kg/ha)", 0, 250, 60, 5)

    # Weather parameters
    st.sidebar.subheader("🌤️ Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", 5, 45, 25, 1)
    humidity = st.sidebar.slider("Humidity (%)", 30, 100, 70, 1)
    rainfall = st.sidebar.slider("Rainfall (mm)", 100, 3000, 800, 50)

    # Temporal parameters
    st.sidebar.subheader("📅 Timing")
    month = st.sidebar.selectbox("Current Month", 
                                list(range(1, 13)), 
                                index=datetime.now().month-1,
                                format_func=lambda x: datetime(2023, x, 1).strftime('%B'))

    # Additional location input (optional)
    st.sidebar.subheader("📍 Location (Optional)")
    latitude = st.sidebar.number_input("Latitude", value=26.8467, help="Optional: For future enhancements")
    longitude = st.sidebar.number_input("Longitude", value=80.9462, help="Optional: For future enhancements")

    # Analysis button
    if st.sidebar.button("🔍 Analyze & Recommend", type="primary"):
        # Determine season and soil type
        season_num, season_name = get_season_from_month(month)
        soil_type_num, soil_type_name = determine_soil_type(soil_ph, nitrogen, phosphorus, potassium)

        # Prepare input data
        input_data = np.array([[
            soil_ph, temperature, rainfall, nitrogen, phosphorus, 
            potassium, humidity, month, season_num, soil_type_num
        ]])

        # Get predictions
        probabilities = model.predict_proba(input_data)[0]
        crop_names = model.classes_

        # Create recommendations
        recommendations = []
        for i, crop in enumerate(crop_names):
            crop_info = crop_db[crop_db['crop_name'] == crop]
            if not crop_info.empty:
                crop_info = crop_info.iloc[0]
                recommendations.append({
                    'crop': crop,
                    'confidence': probabilities[i],
                    'suitability_score': probabilities[i] * 100,
                    'expected_yield': crop_info['expected_yield'],
                    'crop_duration': crop_info['crop_duration']
                })

        # Sort by confidence
        recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)

        # Main content area
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Input Analysis")

            # Display analysis metrics
            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.metric("Season", season_name)
                st.metric("Soil Type", soil_type_name)
                st.metric("Soil pH", f"{soil_ph:.1f}")
                st.metric("Temperature", f"{temperature}°C")

            with metrics_col2:
                st.metric("Humidity", f"{humidity}%")
                st.metric("Rainfall", f"{rainfall}mm")
                st.metric("Nitrogen", f"{nitrogen} kg/ha")
                st.metric("Phosphorus", f"{phosphorus} kg/ha")

            # Radar chart
            soil_params = {
                'soil_ph': soil_ph,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity
            }

            radar_fig = create_radar_chart(soil_params)
            st.plotly_chart(radar_fig, use_container_width=True)

        with col2:
            st.subheader("🏆 Top Crop Recommendations")

            # Display top 5 recommendations
            for i, rec in enumerate(recommendations[:5], 1):
                confidence_color = "#4caf50" if rec['suitability_score'] >= 70 else "#ff9800" if rec['suitability_score'] >= 40 else "#f44336"

                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="color: {confidence_color}; margin: 0;">
                        {i}. {rec['crop'].replace('_', ' ').title()}
                    </h4>
                    <p style="margin: 0.5rem 0;">
                        <strong>Suitability Score:</strong> {rec['suitability_score']:.1f}%<br>
                        <strong>Expected Yield:</strong> {rec['expected_yield']} quintals/ha<br>
                        <strong>Crop Duration:</strong> {rec['crop_duration']} days
                    </p>
                    <div style="background-color: {confidence_color}; height: 4px; width: {rec['suitability_score']}%; border-radius: 2px;"></div>
                </div>
                """, unsafe_allow_html=True)

        # Detailed analysis
        st.subheader("📈 Detailed Analysis")

        # Create comparison chart
        top_crops = recommendations[:8]
        crop_names_clean = [rec['crop'].replace('_', ' ').title() for rec in top_crops]
        suitability_scores = [rec['suitability_score'] for rec in top_crops]

        fig_bar = px.bar(
            x=crop_names_clean,
            y=suitability_scores,
            title="Crop Suitability Comparison",
            labels={'x': 'Crops', 'y': 'Suitability Score (%)'},
            color=suitability_scores,
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Model information
        st.subheader("ℹ️ Model Information")
        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        with info_col2:
            st.metric("Crops in Database", len(crop_db))
        with info_col3:
            st.metric("Features Used", "10")

        # Recommendations and tips
        st.subheader("💡 Agricultural Tips")

        best_crop = recommendations[0]
        if best_crop['suitability_score'] >= 70:
            st.success(f"✅ **{best_crop['crop'].replace('_', ' ').title()}** is highly recommended for your conditions!")
        elif best_crop['suitability_score'] >= 40:
            st.warning(f"⚠️ **{best_crop['crop'].replace('_', ' ').title()}** is moderately suitable. Consider soil improvements.")
        else:
            st.error("❌ Current conditions may not be optimal for high-yield cultivation. Consider soil amendment.")

        # Additional recommendations
        with st.expander("🔍 Detailed Recommendations"):
            st.write("**Soil Management:**")
            if soil_ph < 6.0:
                st.write("- Consider applying lime to increase soil pH")
            elif soil_ph > 7.5:
                st.write("- Consider applying sulfur or organic matter to decrease soil pH")

            if nitrogen < 80:
                st.write("- Apply nitrogen-rich fertilizers or organic manure")
            if phosphorus < 40:
                st.write("- Apply phosphate fertilizers")
            if potassium < 40:
                st.write("- Apply potash or wood ash")

            st.write("**Climate Considerations:**")
            st.write(f"- Current season: {season_name}")
            st.write(f"- Planting timing is important for crop success")
            st.write(f"- Monitor weather forecasts for optimal planting conditions")

# Information sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("""
    This AI-powered system analyzes soil and weather conditions to recommend the most suitable crops for cultivation.

    **Features:**
    - Real-time crop recommendations
    - Suitability scoring
    - Expected yield predictions
    - Agricultural tips
    """)

    st.subheader("📚 How to Use")
    st.write("""
    1. Adjust the parameters in the sidebar
    2. Click 'Analyze & Recommend'
    3. Review the recommendations
    4. Consider the agricultural tips
    """)

if __name__ == "__main__":
    main()
