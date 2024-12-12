import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
import datetime as dt
import numpy as np

# Register Dash page -------------------------------------------- -------------------------------------------------- -------
dash.register_page(__name__, name='2. Post Lap')

# Load data -----------------------------------------------------------------------------------------------------
corners_data = pd.read_csv('data/corners.csv')
telemetry = pd.read_csv('data/all_telemetry.csv')

# Get unique drivers for dropdown
drivers_drop = telemetry['DriverCode'].unique().tolist()

# Display Page Layout
layout = dbc.Container([

        # Store component to load saved data
        dcc.Store(id='lap-store'),
        dcc.Store(id='current-lap-store'), 

# ========================================================== Track image ==================================================================================
        dbc.Row([dbc.Col([html.Img(src='/assets/image/track.png',className="track-image-post")])]),

# ==========================================================  Telemetry lap fast and slow ==================================================================================

    # Box telemetry
    html.Div(children=[html.Div(" ", className="weather-section")],style={"position": "absolute", "top": "68px", "left": "585px", "width": "630px", "height": "25px"}),

    dbc.Row([dbc.Col([dcc.Graph(id='telemetry-lap',config={'displayModeBar': False},className="telemetry-lap")], width=12)]),

# ==========================================================  Cut Graph  ==================================================================================
   
    html.Div(children=[html.Div("LapTime", className="weather-section")],style={"position": "absolute", "top": "195px", "left": "30px", "width": "540px", "height": "25px"}),

    dcc.Dropdown(id='driver-dropdown', className="choice-driver",options=[{'label': driver, 'value': driver} for driver in drivers_drop], value='LEC', clearable=False, style={'background-color': 'rgba(0,0,0,0)', 'color': '#f5f5dc'}),
    dbc.Row([dbc.Col([dcc.Graph(id='cut-graph', config={'displayModeBar': False}, className="cut-style")], width=12)]),

# ========================================================== Straight Line Speed  ==================================================================================
   
    # Let's add the Dropdown for the round selection
    dbc.Row([dbc.Col([dcc.Dropdown(id='lap-dropdown', className="choice-lap", placeholder="Seleziona il giro", clearable=True, style={'background-color': 'rgba(0,0,0,0)', 'color': '#f5f5dc'})], width=12)]),

    html.Div(children=[html.Div("Straight Line Speed", className="weather-section")],style={"position": "absolute", "top": "600px", "left": "45px", "width": "510px", "height": "25px"}),

    dcc.Graph(id='scatter-plot', className="speed-line"),

# ==========================================================  Compound  ==================================================================================

    html.Div(children=[html.Div("Best Compound", className="weather-section")],style={"position": "absolute", "top": "600px", "left": "600px", "width": "510px", "height": "25px"}),

    dcc.Graph(id='compound-plot', className="compound-vs"),

# ==========================================================  Telemetry gap ==================================================================================

    html.Div(children=[html.Div(" ", className="weather-section")],style={"position": "absolute", "top": "937px", "left": "355px", "width": "530px", "height": "25px"}),

    dbc.Row([dbc.Col([dcc.Graph(id='graph-tele-vs', config={'displayModeBar': False}, className="tel-vs-style")], width=12)]),


], fluid=True)



# ==========================================================  Callback  Telemetry ==================================================================================


# Define colors for compounds -----------------------------------------------------------------------------------------------------
compound_colors = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFF200',
    'HARD': '#FBFBFB',
    'INTERMEDIATE': '#39B54A',
    'WET': '#00AEEF',
}

# Define markers for each team -----------------------------------------------------------------------------------------------------
team_markers = {
    'Red Bull Racing': 'circle',
    'Mercedes': 'triangle-up',
    'Ferrari': 'square',
    'McLaren': 'diamond',
    'Alpine': 'star',
    'AlphaTauri': 'x',
    'Aston Martin': 'triangle-down',
    'Haas': 'triangle-left',
    'Williams': 'triangle-right',
    'Alfa Romeo': 'hexagon',
}

# Create a dictionary for curve labels -----------------------------------------------------------------------------------------------------
labels = {}
for _, row in corners_data.iterrows():
    corner_number = row['Number']
    text = row['Text']
    distance = row['Distance']

    # Add the distance to the corresponding curve
    if text not in labels:
        labels[text] = []
    labels[text].append(distance)

# Calculate the center position for each curve
for text, distances in labels.items():
    labels[text] = sum(distances) / len(distances)  # Place the label in the center of the distance


# Creating a Fast and Slow Lap Telemetry Graph -----------------------------------------------------------------------------------------------------
def plot_telemetry_comparison(fastest_lap, slowest_lap, driver_code):
    # Filter data for fast and slow lap
    fastest_lap_data = telemetry[(telemetry['LapNumber'] == fastest_lap) & (telemetry['DriverCode'] == driver_code)]
    slowest_lap_data = telemetry[(telemetry['LapNumber'] == slowest_lap) & (telemetry['DriverCode'] == driver_code)]

    # Colors for every metric
    colors = {
        'Fast': '#FAF33E',
        'Slow': '#218CFE',
    }

    # Create 6 subplots
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.04)

    # Add data for speed
    fig.add_trace(go.Scatter(x=fastest_lap_data['Distance'], y=fastest_lap_data['Speed'], mode='lines',
                             name=f'Speed (Fast Lap {fastest_lap})', line=dict(color=colors['Fast'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=slowest_lap_data['Distance'], y=slowest_lap_data['Speed'], mode='lines',
                             name=f'Speed (Slow Lap {slowest_lap})', line=dict(color=colors['Slow'])), row=1, col=1)

    # Add data for Throttle
    fig.add_trace(go.Scatter(x=fastest_lap_data['Distance'], y=fastest_lap_data['Throttle'], mode='lines',
                             name='Throttle (Fast Lap)', line=dict(color=colors['Fast'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=slowest_lap_data['Distance'], y=slowest_lap_data['Throttle'], mode='lines',
                             name='Throttle (Slow Lap)', line=dict(color=colors['Slow'])), row=2, col=1)

    # TAdd data for Brake
    fig.add_trace(go.Scatter(x=fastest_lap_data['Distance'], y=fastest_lap_data['Brake'], mode='lines',
                             name='Brake (Fast Lap)', line=dict(color=colors['Fast'])), row=3, col=1)
    fig.add_trace(go.Scatter(x=slowest_lap_data['Distance'], y=slowest_lap_data['Brake'], mode='lines',
                             name='Brake (Slow Lap)', line=dict(color=colors['Slow'])), row=3, col=1)

    # Add data for Gear
    fig.add_trace(go.Scatter(x=fastest_lap_data['Distance'], y=fastest_lap_data['nGear'], mode='lines',
                             name='Gear (Fast Lap)', line=dict(color=colors['Fast'])), row=4, col=1)
    fig.add_trace(go.Scatter(x=slowest_lap_data['Distance'], y=slowest_lap_data['nGear'], mode='lines',
                             name='Gear (Slow Lap)', line=dict(color=colors['Slow'])), row=4, col=1)

    # Add data for RPM
    fig.add_trace(go.Scatter(x=fastest_lap_data['Distance'], y=fastest_lap_data['RPM'], mode='lines',
                             name='RPM (Fast Lap)', line=dict(color=colors['Fast'])), row=5, col=1)
    fig.add_trace(go.Scatter(x=slowest_lap_data['Distance'], y=slowest_lap_data['RPM'], mode='lines',
                             name='RPM (Slow Lap)', line=dict(color=colors['Slow'])), row=5, col=1)

    # Add curve labels on the X-axis
    for text, center_distance in labels.items():
        # Add an annotation for each curve at the center of the distance
        fig.add_annotation(
            x=center_distance,
            y=0,  # Position the annotation on the Y-axis
            text=text,
            showarrow=False,
            font=dict(family='serif', size=9, color='#f5f5dc'),
            xanchor='center',
            yanchor='bottom'
        )

    #  layout
    fig.update_layout(
        title_text=f'Telemetry Comparison - {driver_code} | Fast Lap: {fastest_lap} | Slow Lap: {slowest_lap}',
        title_font=dict(family='serif', size=14, color='#fbfcf7'),
        title_x=0.5,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
    )

    for i in range(1, 6): 
        fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                         tickfont=dict(family='serif', size=10, color='#f5f5dc'),linecolor='#8A8A8A',zerolinecolor='#8A8A8A', gridcolor='#8A8A8A', row=i, col=1)
        
        # Aggiorna le etichette degli assi Y
    fig.update_yaxes(title_text=" Speed (km/h)", row=1, col=1)  
    fig.update_yaxes(title_text="Throttle (%)", row=2, col=1)  
    fig.update_yaxes(title_text="Brake", row=3, col=1)      
    fig.update_yaxes(title_text="Gear", row=4, col=1)           
    fig.update_yaxes(title_text="RPM", row=5, col=1)  

    # Set the title for the Y-axis of the last subplot
    fig.update_xaxes(title_text="Distance (meters)", row=5, col=1)

    # Sets the font for the X-axis (common to all subplots)
    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                     tickfont=dict(family='serif', size=10, color='#f5f5dc'),linecolor='#8A8A8A',zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    # Synchronize zoom for all subplots
    fig.update_xaxes(matches='x')

    return fig

@callback(
    Output('telemetry-lap', 'figure'),
    [Input('lap-store', 'data'),   # Completed laps
     Input('current-lap-store', 'data')]  # Current lap
)
def update_telemetry_graph(lap_store_data, current_lap_data):
    driver_code = 'LEC'  # Directly set the driver to LEC
    print(lap_store_data)
    print(current_lap_data)

    # Retrieve completed laps for the driver
    completed_laps = lap_store_data[driver_code]['laps'].copy()  # Use copy to avoid accidental modifications

    # Exclude the current lap from completed laps
    if current_lap_data and driver_code in current_lap_data:
        current_lap = current_lap_data[driver_code]
        if current_lap in completed_laps:
            completed_laps.remove(current_lap)  # Remove the current lap


    # Filter telemetry data for completed laps excluding lap 1
    telemetry_data_filtered = telemetry[
        (telemetry['DriverCode'] == driver_code) & 
        (telemetry['LapNumber'].isin(completed_laps)) &
        (telemetry['LapNumber'] != 1)  # Exclude lap 1
    ]

    # Select the maximum time for each lap
    lap_times = telemetry_data_filtered.groupby('LapNumber')['Time'].max()

    # Debugging: print lap times
    print(f"Lap Times:\n{lap_times.to_string()}")

    # Identify fastest and slowest laps
    if lap_times.empty:
        print("No lap times available to analyze.")
        return go.Figure()  # Return an empty figure if no lap times

    fastest_lap = lap_times.idxmin()  # Lap with the lowest time
    slowest_lap = lap_times.idxmax()  # Lap with the highest time

    # Debugging: print fastest and slowest laps
    print(f"Fastest Lap: {fastest_lap}, Slowest Lap: {slowest_lap}")

    # Generate the telemetry comparison graph
    return plot_telemetry_comparison(fastest_lap, slowest_lap, driver_code)


# ==========================================================  Callback  Cut Graph ==================================================================================
# Convert sector times to total lap time
telemetry['LapTime'] = pd.to_timedelta(telemetry['Sector1Time']) + pd.to_timedelta(telemetry['Sector2Time']) + pd.to_timedelta(telemetry['Sector3Time'])

@callback(
    Output('cut-graph', 'figure'),
    [Input('lap-store', 'data'),
     Input('driver-dropdown', 'value'),
     Input('current-lap-store', 'data')]  
)
def update_cut_graph(lap_store_data, selected_driver, current_lap_data):
    # Initializes an empty dataframe for the combined data
    combined_data = pd.DataFrame()

    # Verify that lap_store_data and selected_driver are not None
    if lap_store_data is None or selected_driver is None or selected_driver not in lap_store_data:
        return go.Figure()  # Return an empty chart if the data is invalid

    # Retrieve completed laps for the driver
    completed_laps = lap_store_data[selected_driver].get('laps', [])

    # Exclude current lap from completed laps
    if current_lap_data and selected_driver in current_lap_data:
        current_lap = current_lap_data[selected_driver]
        if current_lap in completed_laps:
            completed_laps.remove(current_lap)  # Remove current round

    # Handle LEC (Charles Leclerc) data only with laps completed before the current_lap
    if 'LEC' in lap_store_data:
        lec_completed_laps = lap_store_data['LEC'].get('laps', [])

        # If Leclerc's current lap is available, exclude laps after or equal to the current lap
        if current_lap_data and 'LEC' in current_lap_data:
            lec_current_lap = current_lap_data['LEC']
            lec_completed_laps = [lap for lap in lec_completed_laps if lap < lec_current_lap]

        # Filter Leclerc data with completed and valid laps
        lec_data = telemetry[
            (telemetry['DriverCode'] == 'LEC') & 
            (telemetry['LapNumber'].isin(lec_completed_laps))
        ].copy()

        # Add LEC data to the combined dataframe
        if not lec_data.empty:
            combined_data = pd.concat([combined_data, lec_data])

    # Manage selected pilot data (if different from 'LEC')
    if selected_driver and selected_driver != 'LEC':
        driver_completed_laps = lap_store_data[selected_driver].get('laps', [])
        driver_data = telemetry[
            (telemetry['DriverCode'] == selected_driver) & 
            (telemetry['LapNumber'].isin(driver_completed_laps))
        ].copy()

        # Add the selected pilot data to the combined dataframe
        if not driver_data.empty:
            combined_data = pd.concat([combined_data, driver_data])

    # Check if there is valid data
    if combined_data.empty:
        return go.Figure()  # Return an empty chart if there is no data

    # Fix conversion to seconds for timedelta values
    def lap_time_to_seconds(lap_time):
        if isinstance(lap_time, dt.timedelta):  # Check if the value is timedelta
            return lap_time.total_seconds()  # Convert to seconds
        return lap_time 

    # Apply the function to lap times
    combined_data['LapTime'] = combined_data['LapTime'].apply(lap_time_to_seconds)

    # Add TyreLife data if available (make sure TyreLife is present in your data)
    if 'TyreLife' not in combined_data.columns:
        combined_data['TyreLife'] = lap_store_data[selected_driver].get('TyreLife', [None] * len(combined_data))

        # Add FreshTyre data
    if 'FreshTyre' not in combined_data.columns:
        combined_data['FreshTyre'] = lap_store_data[selected_driver].get('FreshTyre', [None] * len(combined_data))

    # Creating the Plotly chart
    fig = px.scatter(
        combined_data,
        x='LapNumber',
        y='LapTime',
        color='Compound',
        symbol='DriverCode',
        color_discrete_map=compound_colors,
        labels={'LapTime': 'Lap Time (seconds)', 'LapNumber': 'Lap Number'},
        hover_data={'TyreLife': True, 'FreshTyre': True},  # Add TyreLife value to hover
    )

    # Set markers mode
    for trace in fig['data']:
        trace['mode'] = 'markers'

    # Invert the Y-axis to correctly represent lap times
    fig.update_yaxes(autorange="reversed")

    # Set the same style as the position chart
    fig.update_layout(
        yaxis=dict(title='Lap Time (seconds)', tickfont=dict(family='serif', size=10, color='#f5f5dc'), range=[0, None]),
        xaxis=dict(title='Lap', tickfont=dict(family='serif', size=10, color='#f5f5dc'), range=[1, 60]),   
        showlegend=True,
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
        margin=dict(l=0, r=0, t=40, b=0),  # Reduce margins to fit chart
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent chart background
        plot_bgcolor='rgba(0,0,0,0)'  
    )
    
    # Set the style of the axis labels
    fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')
    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),linecolor='#8A8A8A',zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    return fig



# ==========================================================  Callback  For Lap Dropdown ==================================================================================

@callback(
    Output('lap-dropdown', 'options'),  # Populate the dropdown options
    [Input('lap-store', 'data'),   # Completed laps
     Input('current-lap-store', 'data')]  # Current lap
)
def update_lap_dropdown(lap_store_data, current_lap_data):
    driver_code = 'PER' 

    # Debugging: View lap_store_data data
    print(f"lap_store_data: {lap_store_data}")

    if lap_store_data is None:
        print("lap_store_data è None")
        return []  # No option available if there is no data

    if driver_code not in lap_store_data:
        print(f"Driver {driver_code} not found in lap_store_data")
        return []  # No option if driver is not present

    # Retrieve completed laps for the driver
    completed_laps = lap_store_data[driver_code].get('laps', [])

    # Debugging: View completed laps
    print(f"Completed laps for {driver_code}: {completed_laps}")

    # If there are no completed laps
    if len(completed_laps) == 0:
        print(f"No lap completed for the driver {driver_code}.")
        return []

    # Exclude current lap from completed laps
    if current_lap_data and driver_code in current_lap_data:
        current_lap = current_lap_data[driver_code]
        if current_lap in completed_laps:
            completed_laps.remove(current_lap)  # Remove current round

    # Filter out lap 1 from the completed laps
    completed_laps = [lap for lap in completed_laps if lap != 1]

    # Generate dropdown options, after removing lap 1 and the current lap
    options = [{'label': f'Lap {lap}', 'value': lap} for lap in completed_laps]

    # Debugging: View generated options
    print(f"Options Dropdown: {options}")

    return options


# ==========================================================  Callback Compound ==================================================================================

# Generate 25 mini-sectors
def assign_minisectors(telemetry, num_minisectors=25):
    total_distance = max(telemetry['Distance'])  # Total circuit distance
    minisector_length = total_distance / num_minisectors  # Length of each mini-sector
    minisectors = [minisector_length * i for i in range(num_minisectors)]
    
    telemetry['Minisector'] = telemetry['Distance'].apply(
        lambda z: minisectors.index(min(minisectors, key=lambda x: abs(x - z))) + 1
    )
    return telemetry

def calculate_fastest_driver_per_point(telemetry):
    """
    Determine the driver with the fastest speed at each point on the track.
    """
    # For each combination of mini-sector and distance, find the record with maximum speed
    fastest_driver = telemetry.loc[telemetry.groupby(['Minisector', 'Distance'])['Speed'].idxmax()]
    
    # Select only relevant columns
    fastest_driver = fastest_driver[['Minisector', 'Distance', 'Driver']].rename(columns={'Driver': 'Fastest_driver'})
    
    # Merges the fastest driver's data into the original dataset
    telemetry = telemetry.merge(fastest_driver, on=['Minisector', 'Distance'], how='left')
    return telemetry

# Faster compound
def calculate_fastest_compounds(telemetry):
    """
    Calculate the fastest compound for each mini-sector based on average speed.
    """
    # Calculate the average speed for compound and mini-sector
    average_speed = telemetry.groupby(['LapNumber', 'Minisector', 'Compound'])['Speed'].mean().reset_index()
    
    # Determine the compound with maximum average speed
    fastest_compounds = average_speed.loc[average_speed.groupby(['LapNumber', 'Minisector'])['Speed'].idxmax()]
    
    # Merge with original telemetry
    telemetry = telemetry.merge(
        fastest_compounds[['LapNumber', 'Minisector', 'Compound']].rename(columns={'Compound': 'Fastest_compound'}),
        on=['LapNumber', 'Minisector'])

    # Mapping compounds into numeric values
    compound_mapping = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    telemetry['Fastest_compound_int'] = telemetry['Fastest_compound'].map(compound_mapping)
    return telemetry


# Function: Plotly graph generation
def generate_minisector_plotly(telemetry, lap):
    """
    Generates an interactive graph for a specific lap, highlighting the fastest compound and the driver with the fastest speed at each point.
    """
    # Filter data for the selected lap
    single_lap = telemetry[telemetry['LapNumber'] == lap]
    
    # Creating the interactive graph
    fig = px.scatter(
        single_lap,
        x='X',
        y='Y',
        color='Fastest_compound',
        color_discrete_map={
            'SOFT': 'red',
            'MEDIUM': 'yellow',
            'HARD': 'white'
        },
        labels={'Fastest_compound': 'Compound'},
        hover_data=['Fastest_driver', 'Speed', 'Distance']  # Show the fastest driver at that point
    )
    
    # Layout
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            title=dict(text="Tyre Compound"),
            yanchor="top", y=0.30,
            xanchor="auto", x=0.60,
            font=dict(family='serif', size=12, color='#f5f5dc')
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(marker=dict(size=5, line=dict(width=0.1, color='white')))
    return fig

@callback(
    Output('compound-plot', 'figure'),
    Input('lap-dropdown', 'value')
)
def update_compound_plot(selected_lap):

    telemetry_compound = assign_minisectors(telemetry)
    telemetry_compound = calculate_fastest_compounds(telemetry_compound)
    telemetry_compound = calculate_fastest_driver_per_point(telemetry_compound)

    # Sort data by distance
    telemetry_compound = telemetry_compound.sort_values(by=['Distance'])

    if selected_lap is None:
        return {}  # Empty chart if no lap is selected
    
    return generate_minisector_plotly(telemetry_compound, selected_lap)

# ==========================================================  Callback  GAP ==================================================================================


# Define the delta_time function
def delta_time(reference_lap, compare_lap):
    """Calculates the delta time of a given lap, along the 'Distance' axis
    of the reference lap.

    Args:
        reference_lap (pd.DataFrame): The lap taken as reference
        compare_lap (pd.DataFrame): The lap to compare

    Returns:
        tuple: (delta, reference, comparison)
          - pd.Series of type `float64` with the delta in seconds.
          - DataFrame for the reference lap
          - DataFrame for the comparison lap
    """
    
    # Ensure 'Time' is in datetime format
    reference_lap['Time'] = pd.to_datetime(reference_lap['Time'])
    compare_lap['Time'] = pd.to_datetime(compare_lap['Time'])

    # Get relevant columns and add distance if not already present
    ref = reference_lap[['Distance', 'Time']].copy()
    comp = compare_lap[['Distance', 'Time']].copy()

    # Interpolate time for the comparison lap to match reference lap distances
    d_reference = np.interp(ref['Distance'], comp['Distance'], 
                            (comp['Time'] - comp['Time'].min()).dt.total_seconds())
    
    # Calculate the delta time
    delta = d_reference - (ref['Time'] - ref['Time'].min()).dt.total_seconds()
    
    return delta, ref, comp

# Driver comparison
def plot_telemetry_comparison_vs(lap_number, compare_driver_code):
    # Set LEC as reference driver
    reference_driver_code = 'LEC'
    
    # Generate colors for pilots using a colorblind palette
    cud_colors = px.colors.qualitative.Safe 
    unique_drivers = telemetry['DriverCode'].unique()
    driver_colors = {driver: cud_colors[i % len(cud_colors)] for i, driver in enumerate(unique_drivers)}

    # Colors for each metric
    colors = {
        f'({reference_driver_code})': '#FAF33E',
        f'({compare_driver_code})': '#218CFE',
    }


    # Filter data for the selected lap and for the two drivers
    reference_driver_data = telemetry[(telemetry['LapNumber'] == lap_number) & 
                                           (telemetry['DriverCode'] == reference_driver_code)]
    compare_driver_data = telemetry[(telemetry['LapNumber'] == lap_number) & 
                                         (telemetry['DriverCode'] == compare_driver_code)]

    # Create the graph with all the telemetry information
    fig = make_subplots(rows=7, cols=1, vertical_spacing=0.04)

    # Add Speed data for both drivers
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'], y=reference_driver_data['Speed'], mode='lines', 
                             name=f'Speed ({reference_driver_code})', line=dict(color=colors[f'({reference_driver_code})'])), row=1, col=1)

    fig.add_trace(go.Scatter(x=compare_driver_data['Distance'], y=compare_driver_data['Speed'], mode='lines', 
                             name=f'Speed ({compare_driver_code})', line=dict(color=colors[f'({compare_driver_code})'])), row=1, col=1)

    # Add Throttle data
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'], y=reference_driver_data['Throttle'], mode='lines', 
                             name=f'Throttle ({reference_driver_code})', line=dict(color=colors[f'({reference_driver_code})'])), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=compare_driver_data['Distance'], y=compare_driver_data['Throttle'], mode='lines', 
                             name=f'Throttle ({compare_driver_code})', line=dict(color=colors[f'({compare_driver_code})'])), row=2, col=1)

    # Add Brake data
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'], y=reference_driver_data['Brake'], mode='lines', 
                             name=f'Brake ({reference_driver_code})', line=dict(color=colors[f'({reference_driver_code})'])), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=compare_driver_data['Distance'], y=compare_driver_data['Brake'], mode='lines', 
                             name=f'Brake ({compare_driver_code})', line=dict(color=colors[f'({compare_driver_code})'])), row=3, col=1)

    # Add Gear data
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'], y=reference_driver_data['nGear'], mode='lines', 
                             name=f'Gear ({reference_driver_code})', line=dict(color=colors[f'({reference_driver_code})'])), row=4, col=1)
    
    fig.add_trace(go.Scatter(x=compare_driver_data['Distance'], y=compare_driver_data['nGear'], mode='lines', 
                             name=f'Gear ({compare_driver_code})', line=dict(color=colors[f'({compare_driver_code})'])), row=4, col=1)

    # Add RPM data
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'], y=reference_driver_data['RPM'], mode='lines', 
                             name=f'RPM ({reference_driver_code})', line=dict(color=colors[f'({reference_driver_code})'])), row=5, col=1)

    fig.add_trace(go.Scatter(x=compare_driver_data['Distance'], y=compare_driver_data['RPM'], mode='lines', 
                             name=f'RPM ({compare_driver_code})', line=dict(color=colors[f'({compare_driver_code})'])), row=5, col=1)

    # Calculate the delta time (gap) with reference driver as LEC
    delta, _, _ = delta_time(reference_driver_data, compare_driver_data)

    # Add Delta Time data to the subplot
    fig.add_trace(go.Scatter(x=reference_driver_data['Distance'],y=delta, mode='lines', name='Gap to ' + reference_driver_code,line=dict(color='black', dash='dash')),row=6,col=1)
    
    # Add curve labels on the X-axis
    for text, center_distance in labels.items():
        # Add an annotation for each curve at the center of the distance
        fig.add_annotation(
            x=center_distance,
            y=0,  # Position the annotation on the Y-axis
            text=text,
            showarrow=False,
            font=dict(family='serif', size=9, color='#f5f5dc'),
            xanchor='center',
            yanchor='bottom'
        )
        
    # Layout
    fig.update_layout(
        title_text=f'Telemetry Comparison - Lap {lap_number}',
        title_font=dict(family='serif', size=14, color='#fbfcf7'),
        title_x=0.5,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
    )

    # Sets the font for the Y-axes of each subplot
    for i in range(1, 7):
        fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                         tickfont=dict(family='serif', size=10, color='#f5f5dc'), linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A', row=i, col=1)

    # Set the title for the Y-axis of the last subplot
    fig.update_xaxes(title_text="Distance (meters)", row=6, col=1)

    # Sets the font for the X-axis (common to all subplots)
    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                     tickfont=dict(family='serif', size=10, color='#f5f5dc'),linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    # Set Y-axis title for the gap subplot
            # Aggiorna le etichette degli assi Y
    fig.update_yaxes(title_text=" Speed (km/h)", row=1, col=1)  
    fig.update_yaxes(title_text="Throttle (%)", row=2, col=1)  
    fig.update_yaxes(title_text="Brake", row=3, col=1)      
    fig.update_yaxes(title_text="Gear", row=4, col=1)           
    fig.update_yaxes(title_text="RPM", row=5, col=1)  
    fig.update_yaxes(title_text="Gap to " + reference_driver_code, row=6, col=1)

    # Synchronize zoom for all subplots
    fig.update_xaxes(matches='x')
    
    return fig

@callback(
    Output('graph-tele-vs', 'figure'),
    [Input('lap-dropdown', 'value'),
     Input('driver-dropdown', 'value')]  
)
def update_cut_graph(selected_lap, selected_driver):
    # Verify that a lap and driver have been selected
    if selected_lap is None or selected_driver is None:
        return go.Figure()  # Return an empty chart if none are selected

    # Call the function to draw telemetry comparison between LEC and selected pilot
    fig = plot_telemetry_comparison_vs(selected_lap, selected_driver)

    return fig


# ==========================================================  Callback Straight Line Speed ==================================================================================

# Definition of the main straight interval in meters
start_straight = -350  # 350 meters before the finish line
end_straight = 450     # 450 meters after the finish line

# Function to prepare data for straight line speed
def prepare_data(telemetry, lap_number, start_straight, end_straight):
    telemetry = telemetry.copy()
    telemetry['Time'] = pd.to_datetime(telemetry['Time'])

    # Filter data for the specific lap
    lap_df = telemetry[telemetry['LapNumber'] == lap_number]

    max_speed = []
    delta_time = []  # To store delta_time instead of end_time
    compound = []
    teams = []
    drivers = lap_df['DriverCode'].unique()

    # List to store all max_time values
    all_max_times = []

    # Iterate over each driver and gather their data
    for driver in drivers:
        driver_df = lap_df[lap_df['DriverCode'] == driver]

        # Filter telemetry data to keep only that on the main straight
        straight_df = driver_df[(driver_df['Distance'] >= start_straight) & 
                                 (driver_df['Distance'] <= end_straight)]
        
        if not straight_df.empty:
            max_speed_value = straight_df['Speed'].max()  # Maximum speed on the main straight
            max_time = straight_df['Time'].max()  # Final time relative to the lap
            tyre_compound = driver_df['Compound'].iloc[-1]  # Tyre compound
            team = driver_df['Team'].iloc[-1]  # Team
            
            max_speed.append(max_speed_value)
            all_max_times.append(max_time)  # Collect all max_time values
            
            compound.append(tyre_compound)
            teams.append(team)

    # Now calculate the best_time (minimum max_time among all drivers)
    best_time = min(all_max_times) if all_max_times else pd.NaT  # Handle case if no data is found

    # Calculate delta_time for each driver in seconds
    for max_time in all_max_times:
        delta_time.append((max_time - best_time).total_seconds())  # Convert timedelta to seconds
    
    # Return the required data
    return max_speed, delta_time, compound, teams, drivers


# Function to create the velocity graph on a straight line
def create_scatter_plot(max_speed, delta_time, compound, teams, drivers, team_markers, compound_colors):
    fig = go.Figure()

    for i, driver in enumerate(drivers):
        marker = team_markers.get(teams[i], 'circle')  # Gets the team symbol

        # Color based on compound
        compound_color = compound_colors.get(compound[i], '#000000')  # Default color black if not found

        fig.add_trace(go.Scatter(
            x=[max_speed[i]],
            y=[delta_time[i]],
            hoverinfo='skip',
            mode='markers+text',
            marker=dict(
                color=compound_color,
                symbol=marker,
                size=15
            ),
            text=driver,
            textfont=dict(color='#FBFBFB'),
            textposition="top right",
            name=driver,
            hovertemplate=f"Driver: {driver}<br>Compound: {compound[i]}<br>Speed: {max_speed[i]} km/h<br>LapTime Delta: {delta_time[i]} s"
        ))

    # Chart aesthetic settings
    fig.update_layout(
        showlegend=False,
        yaxis=dict(title='Time (seconds)', tickfont=dict(family='serif', size=10, color='#f5f5dc')),
        xaxis=dict(title='Maximum speed (km/h)', tickfont=dict(family='serif', size=10, color='#f5f5dc')),   
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Set the style of the axis labels
    fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')
    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    return fig

# Callback to update the chart based on the selected lap
@callback(
    Output('scatter-plot', 'figure'),
    [Input('lap-dropdown', 'value')]
)
def speed_graph(selected_lap):
    max_speed, delta_time, compound, teams, drivers = prepare_data(telemetry, selected_lap, start_straight, end_straight)
    return create_scatter_plot(max_speed, delta_time, compound, teams, drivers, team_markers, compound_colors)