print("Testing Crop Recommendation System\n")
print("="*60)

print("Test Case 1: Wheat-friendly conditions (November)")
recommendations = crop_system.recommend_crop(
    soil_ph=6.5,
    temperature=18,
    rainfall=750,
    nitrogen=140,
    phosphorus=60,
    potassium=60,
    humidity=65,
    month=11
)

print("\nTop 3 Recommendations:")
for i, rec in enumerate(recommendations['recommendations'], 1):
    print(f"\n{i}. {rec['crop']}")
    print(f"   Confidence: {rec['confidence']:.3f}")
    print(f"   Suitability Score: {rec['suitability_score']:.1f}%")
    print(f"   Expected Yield: {rec['expected_yield']} quintals/hectare")
    print(f"   Crop Duration: {rec['crop_duration']} days")
    print(f"   Key Factors: {'; '.join(rec['suitability_factors'][:3])}")

print("\n" + "="*60)

print("Test Case 2: Rice-friendly conditions (July)")
recommendations = crop_system.recommend_crop(
    soil_ph=6.2,
    temperature=28,
    rainfall=1200,
    nitrogen=100,
    phosphorus=50,
    potassium=50,
    humidity=80,
    month=7
)

print("\nTop 3 Recommendations:")
for i, rec in enumerate(recommendations['recommendations'], 1):
    print(f"\n{i}. {rec['crop']}")
    print(f"   Confidence: {rec['confidence']:.3f}")
    print(f"   Suitability Score: {rec['suitability_score']:.1f}%")
    print(f"   Expected Yield: {rec['expected_yield']} quintals/hectare")
    print(f"   Crop Duration: {rec['crop_duration']} days")

print("\n" + "="*60)

print("Test Case 3: High rainfall monsoon conditions (August)")
recommendations = crop_system.recommend_crop(
    soil_ph=7.0,
    temperature=26,
    rainfall=1500,
    nitrogen=120,
    phosphorus=70,
    potassium=80,
    humidity=85,
    month=8
)

print("\nTop 3 Recommendations:")
for i, rec in enumerate(recommendations['recommendations'], 1):
    print(f"\n{i}. {rec['crop']}")
    print(f"   Confidence: {rec['confidence']:.3f}")
    print(f"   Suitability Score: {rec['suitability_score']:.1f}%")
    print(f"   Expected Yield: {rec['expected_yield']} quintals/hectare")