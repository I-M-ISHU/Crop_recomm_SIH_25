import pickle

model_components = {
    'model': rf_model,
    'scaler': scaler,
    'crop_database': crop_df,
    'feature_names': feature_names,
    'accuracy': accuracy
}

with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model_components, f)

print("Model saved successfully to 'crop_recommendation_model.pkl'")
print(f"Model accuracy: {accuracy:.4f}")
print(f"File contains: model, scaler, crop database, and feature names")