# Import libraries
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, callback, Input, Output, State
import datetime as dt
import plotly.graph_objs as go
import plotly.express as px
import dash
from plotly.subplots import make_subplots
import dash_ag_grid as dag

# Registering Your Page in Dash:
dash.register_page(__name__, path='/', name='1. Now')

# Load data
df_weather = pd.read_csv('data/weather_data.csv')
data_all = pd.read_csv('data/all_telemetry.csv')
predictions = pd.read_csv('data/predictions.csv')

# Get all unique lap numbers
lap_numbers = data_all['LapNumber'].unique()

# Converting 'Time' Column to Datetime Format
data_all['Time'] = pd.to_datetime(data_all['Time']).dt.time

#Filtering Data by lap:
lap_data_dict = {lap_number: data_all[data_all['LapNumber'] == lap_number] for lap_number in lap_numbers}

# Tracking the Status of Each driver
driver_state = {
    driver_code: {
        'lap': 1, 
        'current_time': dt.time(0, 0, 0), 
        'timer': 0,
        'start_time': dt.datetime.now()
    } 
    for driver_code in data_all['DriverCode'].unique()
}

# Generation of Color Blind Safe Colors for Each Driver:
cud_colors = px.colors.qualitative.Safe
unique_drivers = data_all['DriverCode'].unique()
driver_colors = {driver: cud_colors[i % len(cud_colors)] for i, driver in enumerate(unique_drivers)}


# App layout with telemetry and location graph
layout = dbc.Container([

        # Driver image
        dbc.Row([dbc.Col([html.Img(src='/assets/image/16.png',className="driver-image")])]),

# ============================================================  Weather information  ==================================================================================

        # Interval to update data every minute (60,000 milliseconds)
        dcc.Interval(id='interval-component-weather', interval=60000, n_intervals=0),

        # Weather information section
        html.Div(children=[html.Div("Weather information", className="weather-section")],style={"position": "absolute","top": "65px","left": "230px","width": "1000px","height": "25px" }),

        dbc.Col([
            # Air temperature
            html.Div(children="Air temperature [°C]", className="weather-label", style={ "top": "145px","position": "absolute","left": "238px"}),

            # Dynamic element of air temperature
            html.Div(id="air-temp", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "293px"}),

            # Relative humidity
            html.Div(children="Relative humidity [%]", className="weather-label", style={"position": "absolute","top": "145px","left": "388px"}),

            # Dynamic element of Relative humidity        
            html.Div(id="humidity", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "415px"}),

            # Air pressure
            html.Div(children="Air pressure [mbar]", className="weather-label", style={"position": "absolute","top": "145px","left": "543px"}),

            # Dynamic element of Air pressure 
            html.Div(id="pressure", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "578px"}),

            # Rainfall
            html.Div(children="Rainfall?", className="weather-label", style={"position": "absolute","top": "145px","left": "699px"}),

            # Dynamic element of
            html.Div(id="rainfall", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "714px"}),

            # Track temperature
            html.Div(children="Track temperature [°C]", className="weather-label", style={"position": "absolute","top": "145px","left": "775px"}),

            # Dynamic element of Track temperature
            html.Div(id="track-temp", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "820px"}),

            # Wind direction 
            html.Div(children="Wind direction [°]", className="weather-label", style={"position": "absolute","top": "145px","left": "945px" }),

            # Dynamic element of Wind direction 
            html.Div(id="wind-dir", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "970px"}),

            # Wind speed
            html.Div(children="Wind speed [m/s]", className="weather-label", style={ "position": "absolute","top": "145px","left": "1075px",}),

            # Dynamic element of Wind speed
            html.Div(id="wind-speed", className="weather-label-dina", style={"position": "absolute","top": "100px","left": "1110px"})], width=10),

# ==========================================================  Track Position  ==================================================================================

        # Store to store data on completed laps
        dcc.Store(id='lap-store', storage_type='memory'),
        dcc.Store(id='current-lap-store', storage_type='memory'), 

        # Update interval
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),

        # Track Position Graph
        dbc.Row([dbc.Col([dcc.Graph(id='lap-graph', config={'displayModeBar': False}, className="track-position-style")], width=12)]),

# ==========================================================  Telemetry  ==================================================================================

    dcc.Interval(id='interval-component-tel', interval=1000, n_intervals=0),

    # Box telemetry
    html.Div(children=[html.Div(" ", className="weather-section")],style={"position": "absolute","top": "179px","left": "400px","width": "830px","height": "25px" }),

    # Leclerc telemetry
    dbc.Row([dbc.Col([dcc.Graph(id='telemetry-graph', config={'displayModeBar': False}, className="telemetry-style")], width=12)]),

    # ==========================================================  Race  ==================================================================================

    # Box title
    html.Div(children=[html.Div("Driver Standings", className="weather-section")],style={"position": "absolute","top": "705px","left": "50px","width": "515px","height": "25px" }),

    # position-graph
    dbc.Row([dbc.Col([dcc.Graph(id='position-graph', config={'displayModeBar': False}, className="position-style")], width=12)]),

    # ==========================================================  Prediction  ==================================================================================

    # Box title
    html.Div(children=[html.Div("Prediction Next Pit-Stop", className="weather-section")],style={"position": "absolute","top": "705px","left": "615px","width": "515px","height": "25px" }),

    # Container for the table
    html.Div(id='table-container', style={'marginTop': '90px'}) 

], fluid=True)


# ==========================================================  Callback  Weather  ==================================================================================

# Callback to update weather information every minute
@callback(
    [Output('air-temp', 'children'),
     Output('humidity', 'children'),
     Output('pressure', 'children'),
     Output('rainfall', 'children'),
     Output('track-temp', 'children'),
     Output('wind-dir', 'children'),
     Output('wind-speed', 'children')],
    Input('interval-component-weather', 'n_intervals')
)

def update_weather_info(n_intervals):
    # Using index to show data cyclically
    row_index = n_intervals % len(df_weather)
    row = df_weather.iloc[row_index]

    # Returning values
    return (f"{row['AirTemp']:.1f}",
            f"{row['Humidity']:.1f}",
            f"{row['Pressure']:.1f}",
            'Yes' if row['Rainfall'] else 'No',
            f"{row['TrackTemp']:.1f}",
            f"{row['WindDirection']:.1f}",
            f"{row['WindSpeed']:.1f}")

# ==========================================================  Callback  Track  ==================================================================================
# Driver on track
def plot_lap_image(driver_state):
    fig = go.Figure()

   # Add the circuit track for each driver
    for driver_code, state in driver_state.items():
        current_lap = state['lap']
        lap_data = lap_data_dict.get(current_lap)

        if lap_data is None:
            continue

        circuit_data = lap_data[lap_data['DriverCode'] == driver_code]
        if len(circuit_data) == 0:
            continue

        circuit_x = circuit_data['X'].values
        circuit_y = circuit_data['Y'].values

        # Trace the circuit
        fig.add_trace(go.Scatter(
            x=circuit_x,
            y=circuit_y,
            mode='lines',
            name='',  # Do not show the path in the legend
            line=dict(color= '#FBFBFB'),  # Track color
            hoverinfo='skip',  # Disable info when hovering over circuit
            showlegend=False  # Do not show the path in the legend
        ))

    # Add markers for each driver's current position
    for driver_code, state in driver_state.items():
        current_lap = state['lap']
        lap_data = lap_data_dict.get(current_lap)

        if lap_data is None:
            continue

        circuit_data = lap_data[lap_data['DriverCode'] == driver_code]
        if len(circuit_data) == 0:
            continue

        # Filter data for the current time
        current_data = circuit_data[circuit_data['Time'] <= state['current_time']]

        if len(current_data) > 0:
            x_position = current_data['X'].values[-1]  # Last available position up to the current time
            y_position = current_data['Y'].values[-1]
            color = driver_colors.get(driver_code, 'black')

            # Add marker for current pilot position
            fig.add_trace(go.Scatter(
                x=[x_position],
                y=[y_position],
                mode='markers',
                name=f'{driver_code} - Lap {current_lap}',
                marker=dict(size=10, color=color),
                hovertemplate=f'{driver_code}<extra></extra>'
            ))

    fig.update_layout(
        showlegend=True,
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
        margin=dict(l=0, r=0, t=40, b=0),  # Reduce margins to fit chart
        xaxis=dict(visible=False),  # Removes the X-axis
        yaxis=dict(visible=False),  # Removes the Y-axis
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent chart background
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent Track background
    )

    return fig

@callback(
    [Output('lap-graph', 'figure'),
     Output('lap-store', 'data'),
     Output('current-lap-store', 'data')],
    Input('interval-component', 'n_intervals'),
    [State('lap-store', 'data'),
     State('current-lap-store', 'data')]
)
def update_track_position(n_intervals, lap_store_data, current_lap_store):
    # Initialize data if None
    if lap_store_data is None:
        lap_store_data = {}

    if current_lap_store is None:
        current_lap_store = {}

    # Update the status of the drivers and the laps completed
    for driver_code, state in driver_state.items():
        current_lap = state['lap']
        lap_data_for_driver = lap_data_dict.get(current_lap)

        if lap_data_for_driver is None:
            print(f"No lap data for lap {current_lap} for driver {driver_code}.")
            continue

        driver_lap_data = lap_data_for_driver[lap_data_for_driver['DriverCode'] == driver_code]

        if driver_lap_data.empty:
            print(f"No lap data available for driver {driver_code}.")
            continue

        max_time_lap = max(driver_lap_data['Time'])
        elapsed_time = (dt.datetime.now() - state['start_time']).total_seconds()
        state['current_time'] = (dt.datetime.combine(dt.date.today(), dt.time(0)) + dt.timedelta(seconds=elapsed_time)).time()

        # If the driver has completed the current lap
        if state['current_time'] > max_time_lap:
            state['timer'] = 0
            state['lap'] += 1
            state['start_time'] = dt.datetime.now()

            if state['lap'] > max(lap_numbers):
                state['lap'] = max(lap_numbers)

        # Updates the lap_store with all completed laps up to the current lap
        if driver_code not in lap_store_data:
            lap_store_data[driver_code] = {'laps': []}

        if state['lap'] not in lap_store_data[driver_code]['laps']:
            # Add all laps less than or equal to current_lap
            lap_store_data[driver_code]['laps'] = list(range(1, state['lap'] + 1))

        # Update the current_lap_store
        current_lap_store[driver_code] = state['lap']

    print(lap_store_data)

    return plot_lap_image(driver_state), lap_store_data, current_lap_store


# ==========================================================  Callback  Telemetry (with Debug) ==================================================================================


# Telemetry Tracking Function for Charles Leclerc
def plot_telemetry(lap_number, current_time):
    # Filter data by Charles Leclerc
    lap_data = data_all[(data_all['LapNumber'] == lap_number) & (data_all['DriverCode'] == 'LEC')]

    # Filter telemetry data by current distance
    lap_data = lap_data[lap_data['Time'] <= current_time]


    # Seleziona colori per i grafici
    telemetry_colors = {
        'Speed': "#FFAFD7",
        'Throttle': "#F5F3BB",
        'Brake': "#FF6C6E",
        'Gear': "#23C9FF",
        'RPM': "#F9C22E",
        'DRS': "#7ED482"
    }

    # Create 6 subplots
    fig = make_subplots(rows=6, cols=1, vertical_spacing=0.04)

    # Add data for each subplot
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['Speed'], mode='lines', 
                             name='Speed (km/h)', line=dict(color=telemetry_colors['Speed'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['Throttle'], mode='lines', 
                             name='Throttle', line=dict(color=telemetry_colors['Throttle'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['Brake'], mode='lines', 
                             name='Brake', line=dict(color=telemetry_colors['Brake'])), row=3, col=1)
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['nGear'], mode='lines', 
                             name='Gear', line=dict(color=telemetry_colors['Gear'])), row=4, col=1)
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['RPM'], mode='lines', 
                             name='RPM', line=dict(color=telemetry_colors['RPM'])), row=5, col=1)
    fig.add_trace(go.Scatter(x=lap_data['Distance'], y=lap_data['DRS'], mode='lines', 
                             name='DRS', line=dict(color=telemetry_colors['DRS'])), row=6, col=1)

    # Update layout for axes and legend
    fig.update_layout(
        height=500,
        title_text=f"Leclerc Telemetry - Lap {lap_number} | Lap Time: {current_time.strftime('%H:%M:%S')}.{int(current_time.microsecond / 1000):03}",  # Added current time with milliseconds
        title_font=dict(family='serif', size=14, color='#fbfcf7'),
        title_x=0.5,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)',   
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
    )

    # Sets the font for the Y-axes of each subplot
    for i in range(0, 7):  # subplots from 1 to 6
        fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                         tickfont=dict(family='serif', size=10, color='#f5f5dc'), linecolor='#8A8A8A', gridcolor='#8A8A8A', zerolinecolor='#8A8A8A', row=i, col=1)


    # Set the title for the Y-axis of the last subplot
    fig.update_xaxes(title_text="Distance (meters)", row=6, col=1)

    # Sets the font for the X-axis (common to all subplots)
    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                     tickfont=dict(family='serif', size=10, color='#f5f5dc'),linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')
    
    return fig


@callback(
    Output('telemetry-graph', 'figure'),
    Input('interval-component-tel', 'n_intervals')
)
def update_telemetry_graph(n_intervals):
    lap_number = driver_state['LEC']['lap']  # Use Leclerc's current lap
    current_time = driver_state['LEC']['current_time']

    # Filter only telemetry data up to the current time
    lap_data = data_all[(data_all['LapNumber'] == lap_number) & (data_all['DriverCode'] == 'LEC')]
    lap_data = lap_data[lap_data['Time'] <= current_time]

    return plot_telemetry(lap_number, current_time)


# ==========================================================  Callback  Position Race  ==================================================================================

# Function to track positions during the race
def plot_position_race(lap_store, current_lap_store):
    # Get the unique drivers
    drivers = list(lap_store.keys())
    
    # Layout
    fig = go.Figure()

    # Create the graph for each driver
    for drv in drivers:
        # Retrieve the current lap from the current_lap_store
        current_lap = current_lap_store.get(drv, 0)

        # Filter data up to the current completed lap, excluding the current lap itself
        drv_data = data_all[(data_all['DriverCode'] == drv) & 
                            (data_all['LapNumber'] < current_lap)].sort_values(by='LapNumber')
        
        # If there is data, add a trace to the graph
        if not drv_data.empty:
            fig.add_trace(go.Scatter(
                x=drv_data['LapNumber'],
                y=drv_data['Position'],
                mode='lines + markers',
                name=drv,
                line=dict(color=driver_colors[drv])
            ))

    # Set axis limits and labels
    fig.update_layout(
        yaxis=dict(title='Position', range=[20.5, 0.5], tickvals=[1, 5, 10, 15, 20]),
        xaxis=dict(title='Lap', range=[1, 60]),        
        showlegend=True,
        legend=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
        margin=dict(l=0, r=0, t=40, b=0), 
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_yaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                     tickfont=dict(family='serif', size=10, color='#f5f5dc'), linecolor='#8A8A8A',zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    fig.update_xaxes(title=dict(font=dict(family='serif', size=12, color='#f5f5dc')),
                     tickfont=dict(family='serif', size=10, color='#f5f5dc'), linecolor='#8A8A8A', zerolinecolor='#8A8A8A', gridcolor='#8A8A8A')

    return fig

@callback(
    Output('position-graph', 'figure'),
    [Input('lap-store', 'data'),
     Input('current-lap-store', 'data')]
)
def update_position_graph(lap_store_data, current_lap_store):
    # There are no more driver states, so we don't handle driver_state
    # Update drivers' positions based on lap_store and current_lap_store data
    
    for driver_code in lap_store_data:
        if driver_code not in lap_store_data:
            continue

        # Update the number of laps completed for the driver
        completed_laps = lap_store_data[driver_code].get('laps', [])
        current_lap = current_lap_store.get(driver_code, 0)
        
        # Make sure the current round is up to date
        if current_lap > len(completed_laps):
            current_lap = len(completed_laps)

        # If the driver has completed all the laps, stop him on the last lap
        if current_lap > max(lap_numbers):
            current_lap = max(lap_numbers)

        # Exclude current lap from completed laps
        if current_lap_store and driver_code in current_lap_store:
            if current_lap in completed_laps:
                completed_laps.remove(current_lap)

        # Update current_lap_store with current lap
        current_lap_store[driver_code] = current_lap

    return plot_position_race(lap_store_data, current_lap_store)


# ==========================================================  Callback  Prediction  ==================================================================================
# Callback  Prediction
@callback(
    Output('table-container', 'children'),
    Input('current-lap-store', 'data')
)
def update_table(current_lap):
    if not current_lap or not isinstance(current_lap, dict):
        return html.P(
            "Nessun dato disponibile",
            style={
                'textAlign': 'center',
                'color': '#f5f5dc',
                'fontFamily': 'serif',
                'fontSize': '14px'
            }
        )

    rows = []
    for driver, lap in current_lap.items():
        filtered_predictions = predictions[(predictions['driver'] == driver) & (predictions['lap'] == lap)]
        if not filtered_predictions.empty:
            rows.append(filtered_predictions.to_dict('records')[0])

    if not rows:
        return html.P(
            "Nessun dato corrispondente trovato",
            style={
                'textAlign': 'center',
                'color': '#f5f5dc',
                'fontFamily': 'serif',
                'fontSize': '14px'
            }
        )

    # Ordina le righe per `update_time` o un criterio alternativo
    rows = sorted(rows, key=lambda x: x.get('update_time', x['lap']), reverse=True)

    # Column definitions for AgGrid
    column_defs = [{'headerName': col, 'field': col, 'headerClass': 'custom-header'} for col in predictions.columns]

    # AgGrid table with the same style as the previous DataTable
    table = dag.AgGrid(
        id="predictions-table",
        columnDefs=column_defs,  # Column definitions for AgGrid
        rowData=rows,  # Data for the grid
        columnSize="sizeToFit",  # Make columns adjust to fit content
        dashGridOptions={
            "suppressRowHoverHighlight": True,  # Disable row hover highlight
            "animateRows": False,  # Disable animation
            "suppressScrollOnNewData": True  # Avoid scroll reset when data is updated
        },
        style={
            'overflowX': 'auto',
            'height': '350px',
            'overflowY': 'auto',
            'position': 'absolute',
            'top': '735px',
            'left': '615px',
            'width': '515px'
        },
        # Default column styles for all columns
        defaultColDef={
            'cellStyle': {
                'backgroundColor': 'rgba(66, 62, 62, 0.9)',
                'color': '#f5f5dc',
                'fontFamily': 'serif',
                'textAlign': 'center',
                'fontSize': '14px'
            }
        },
    )

    return html.Div(
        table
    )
