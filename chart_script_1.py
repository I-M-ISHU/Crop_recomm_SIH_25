import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Data from the JSON
crops = ["Wheat", "Rice Basmati", "Rice Non-Bas", "Sugarcane", "Cotton", "Maize", "Soybean", "Groundnut", "Mustard", "Barley"]
parameters = ["Soil pH", "Temp (°C)", "Rainfall", "Nitrogen"]
suitability_matrix = [
    [1, 1, 1, 1], 
    [1, 2, 2, 1], 
    [1, 2, 2, 1], 
    [1, 2, 2, 2], 
    [2, 2, 1, 1], 
    [1, 1, 1, 2], 
    [1, 1, 1, 0], 
    [1, 2, 1, 0], 
    [1, 0, 0, 1], 
    [1, 0, 1, 1]
]

# Convert to DataFrame for easier handling
df = pd.DataFrame(suitability_matrix, index=crops, columns=parameters)

# Create text labels for hover
hover_labels = []
requirement_map = {0: "Low", 1: "Moderate", 2: "High"}
for i, crop in enumerate(crops):
    row_labels = []
    for j, param in enumerate(parameters):
        value = suitability_matrix[i][j]
        row_labels.append(requirement_map[value])
    hover_labels.append(row_labels)

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=df.values,
    x=parameters,
    y=crops,
    colorscale=[
        [0, '#2E8B57'],    # Green for low requirement (0)
        [0.5, '#D2BA4C'],  # Yellow for moderate requirement (1)
        [1, '#DB4545']     # Red for high requirement (2)
    ],
    showscale=True,
    colorbar=dict(
        title="Requirement",
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Low", "Moderate", "High"]
    ),
    hoverongaps=False,
    hovertemplate='<b>%{y}</b><br>%{x}: %{customdata}<extra></extra>',
    customdata=hover_labels
))

# Update layout
fig.update_layout(
    title="Crop Suitability by Parameters",
    xaxis_title="Parameters",
    yaxis_title="Crops"
)

# Save the chart
fig.write_image("crop_suitability_heatmap.png")