import plotly.graph_objects as go
import plotly.express as px


data = {"components": [{"name": "Soil Parameters", "type": "input", "items": ["pH Level", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]}, {"name": "Weather Data", "type": "input", "items": ["Temperature", "Humidity", "Rainfall"]}, {"name": "Temporal Factors", "type": "input", "items": ["Current Month", "Season Detection"]}, {"name": "Feature Engineering", "type": "processing", "items": ["Season Classification", "Soil Type Determination", "Parameter Scaling"]}, {"name": "ML Model", "type": "processing", "items": ["Random Forest", "98.7% Accuracy", "20 Crop Classes"]}, {"name": "Output Generation", "type": "output", "items": ["Crop Rankings", "Confidence Scores", "Yield Predictions", "Duration Estimates"]}, {"name": "Applications", "type": "output", "items": ["Desktop GUI", "Web Interface", "Mobile Ready"]}]}


colors = {
    'input': '#1FB8CD',
    'processing': '#2E8B57', 
    'output': '#DB4545'
}


fig = go.Figure()


box_width = 2
box_height = 1.5
x_positions = {'input': 1, 'processing': 5, 'output': 9}

input_comps = [c for c in data['components'] if c['type'] == 'input']
processing_comps = [c for c in data['components'] if c['type'] == 'processing']  
output_comps = [c for c in data['components'] if c['type'] == 'output']


def add_box(fig, x, y, width, height, color, title, items, comp_type):
    
    fig.add_shape(
        type="rect",
        x0=x-width/2, y0=y-height/2,
        x1=x+width/2, y1=y+height/2,
        line=dict(color=color, width=2),
        fillcolor=color,
        opacity=0.3
    )
    
    
    fig.add_annotation(
        x=x, y=y+height/3,
        text=f"<b>{title[:15]}</b>",
        showarrow=False,
        font=dict(size=12, color=color),
        align="center"
    )
    
    
    items_text = "<br>".join([item[:15] for item in items[:4]])  # Limit to 4 items to fit
    fig.add_annotation(
        x=x, y=y-height/6,
        text=items_text,
        showarrow=False,
        font=dict(size=9, color="black"),
        align="center"
    )


y_start = 6
for i, comp in enumerate(input_comps):
    y_pos = y_start - i * 2.5
    add_box(fig, x_positions['input'], y_pos, box_width, box_height, 
            colors['input'], comp['name'], comp['items'], 'input')

  
y_start = 4
for i, comp in enumerate(processing_comps):
    y_pos = y_start - i * 2.5
    add_box(fig, x_positions['processing'], y_pos, box_width, box_height,
            colors['processing'], comp['name'], comp['items'], 'processing')


y_start = 4
for i, comp in enumerate(output_comps):
    y_pos = y_start - i * 2.5
    add_box(fig, x_positions['output'], y_pos, box_width, box_height,
            colors['output'], comp['name'], comp['items'], 'output')


input_y_positions = [6, 3.5, 1]
processing_y_positions = [4, 1.5]

for input_y in input_y_positions:
    for proc_y in processing_y_positions:
        
        fig.add_annotation(
            x=3.2, y=proc_y,
            ax=2.2, ay=input_y,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray",
            showarrow=True,
            text=""
        )


output_y_positions = [4, 1.5]

for proc_y in processing_y_positions:
    for output_y in output_y_positions:
        
        fig.add_annotation(
            x=7.2, y=output_y,
            ax=6.2, ay=proc_y,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray",
            showarrow=True,
            text=""
        )


fig.add_annotation(x=1, y=7.5, text="<b>INPUT STAGE</b>", showarrow=False, 
                  font=dict(size=14, color=colors['input']))
fig.add_annotation(x=5, y=7.5, text="<b>PROCESSING</b>", showarrow=False,
                  font=dict(size=14, color=colors['processing']))
fig.add_annotation(x=9, y=7.5, text="<b>OUTPUT STAGE</b>", showarrow=False,
                  font=dict(size=14, color=colors['output']))


fig.update_layout(
    title='Crop Rec System Architecture',
    xaxis=dict(
        range=[0, 10.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[-1, 8],
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    plot_bgcolor='white',
    showlegend=False
)

fig.update_traces(cliponaxis=False)


fig.write_image('crop_system_arch.png')