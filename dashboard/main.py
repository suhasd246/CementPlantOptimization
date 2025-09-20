# ~/Documents/cement-operations-optimization/src/dashboard/main.py
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import asyncio
from typing import Dict, List, Any
import dash_bootstrap_components as dbc
import time
import os

# Disable dotenv loading to avoid permission issues
os.environ['FLASK_SKIP_DOTENV'] = '1'

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Cement Plant Optimization Dashboard"

# API Client
class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_plant_status(self):
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_optimization_history(self):
        try:
            response = requests.get(f"{self.base_url}/optimization-history", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def optimize_raw_materials(self, data):
        try:
            response = requests.post(f"{self.base_url}/optimize/raw-materials", 
                                   json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def optimize_grinding(self, data):
        try:
            response = requests.post(f"{self.base_url}/optimize/grinding", 
                                   json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def optimize_clinkerization(self, data):
        try:
            response = requests.post(f"{self.base_url}/optimize/clinkerization", 
                                   json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def optimize_quality(self, data):
        try:
            response = requests.post(f"{self.base_url}/optimize/quality", 
                                   json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

# Initialize API client
api_client = APIClient()

# Sample data for demonstration
def generate_sample_data():
    """Generate sample data for the dashboard"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    
    data = {
        'timestamp': dates,
        'energy_consumption': np.random.normal(2000, 200, len(dates)),
        'quality_score': np.random.normal(0.85, 0.05, len(dates)),
        'sustainability_score': np.random.normal(0.75, 0.08, len(dates)),
        'efficiency': np.random.normal(0.82, 0.06, len(dates)),
        'alternative_fuel_ratio': np.random.normal(0.25, 0.05, len(dates)),
        'temperature': np.random.normal(1450, 50, len(dates)),
        'vibration': np.random.normal(2.5, 0.8, len(dates))
    }
    
    return pd.DataFrame(data)

# Generate sample data
df = generate_sample_data()

# Dashboard layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🏭 Cement Plant Optimization Platform", 
                       className="text-center text-white mb-3"),
                html.P("AI-driven autonomous operations for energy efficiency, quality control, and sustainability",
                       className="text-center text-white-50 mb-0")
            ], className="bg-primary p-4 rounded mb-4")
        ])
    ]),
    
    # Connection Status
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.Span("🔗", className="me-2"),
                html.Span("API Status: ", className="me-2"),
                html.Span(id="api-status", children="Connecting...", className="fw-bold")
            ], id="connection-alert", color="info", className="mb-3")
        ])
    ]),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("⚡ Energy Efficiency", className="card-title text-success"),
                    html.H2(id="efficiency-metric", children="83.3%", className="text-success"),
                    html.P("+2.3% from last week", className="text-muted small")
                ])
            ], className="h-100 shadow-sm")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎯 Quality Score", className="card-title text-info"),
                    html.H2(id="quality-metric", children="85.7%", className="text-info"),
                    html.P("+1.8% from last week", className="text-muted small")
                ])
            ], className="h-100 shadow-sm")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🌱 Sustainability", className="card-title text-warning"),
                    html.H2(id="sustainability-metric", children="75.2%", className="text-warning"),
                    html.P("+3.2% from last week", className="text-muted small")
                ])
            ], className="h-100 shadow-sm")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔄 Alternative Fuel", className="card-title text-danger"),
                    html.H2(id="fuel-metric", children="25.8%", className="text-danger"),
                    html.P("+5.1% from last week", className="text-muted small")
                ])
            ], className="h-100 shadow-sm")
        ], width=3)
    ], className="mb-4"),
    
    # Main Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📊 Energy Consumption Trend", className="mb-0"),
                    html.Small("Last 24 hours", className="text-muted")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='energy-chart')
                ])
            ], className="shadow-sm")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🌡️ Process Temperature", className="mb-0"),
                    html.Small("Kiln temperature monitoring", className="text-muted")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='temperature-chart')
                ])
            ], className="shadow-sm")
        ], width=6)
    ], className="mb-4"),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Process Control Panel"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Raw Material Moisture Content"),
                            dcc.Slider(
                                id='moisture-slider',
                                min=0.05, max=0.25, step=0.01, value=0.15,
                                marks={0.05: '5%', 0.15: '15%', 0.25: '25%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Kiln Temperature"),
                            dcc.Slider(
                                id='temperature-slider',
                                min=1400, max=1500, step=10, value=1450,
                                marks={1400: '1400°C', 1450: '1450°C', 1500: '1500°C'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Alternative Fuel Ratio"),
                            dcc.Slider(
                                id='fuel-slider',
                                min=0.1, max=0.5, step=0.05, value=0.25,
                                marks={0.1: '10%', 0.25: '25%', 0.5: '50%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Grinding Fineness"),
                            dcc.Slider(
                                id='fineness-slider',
                                min=0.8, max=0.95, step=0.01, value=0.9,
                                marks={0.8: '80%', 0.9: '90%', 0.95: '95%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🚀 Run Raw Materials Optimization", 
                                     id="optimize-raw-btn", color="primary", 
                                     className="w-100 mb-2")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("⚙️ Run Grinding Optimization", 
                                     id="optimize-grinding-btn", color="success", 
                                     className="w-100 mb-2")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🔥 Run Clinkerization Optimization", 
                                     id="optimize-clinker-btn", color="warning", 
                                     className="w-100 mb-2")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("🎯 Run Quality Optimization", 
                                     id="optimize-quality-btn", color="info", 
                                     className="w-100 mb-2")
                        ], width=6)
                    ])
                ])
            ], className="shadow-sm")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Real-time Alerts"),
                dbc.CardBody([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Div([
                                html.Span("��", className="me-2"),
                                html.Strong("High Temperature Alert"),
                                html.Br(),
                                html.Small("Kiln temperature exceeds optimal range", className="text-muted")
                            ])
                        ], color="danger"),
                        dbc.ListGroupItem([
                            html.Div([
                                html.Span("��", className="me-2"),
                                html.Strong("Energy Efficiency Warning"),
                                html.Br(),
                                html.Small("Grinding energy consumption above threshold", className="text-muted")
                            ])
                        ], color="warning"),
                        dbc.ListGroupItem([
                            html.Div([
                                html.Span("��", className="me-2"),
                                html.Strong("Quality Target Met"),
                                html.Br(),
                                html.Small("Product quality within optimal range", className="text-muted")
                            ])
                        ], color="success")
                    ])
                ])
            ], className="shadow-sm")
        ], width=6)
    ], className="mb-4"),
    
    # Optimization Results
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Recent Optimization Results"),
                dbc.CardBody([
                    html.Div(id="optimization-results")
                ])
            ], className="shadow-sm")
        ])
    ], className="mb-4"),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Process Data Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='process-table',
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Energy (kWh)", "id": "energy_consumption", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Quality Score", "id": "quality_score", "type": "numeric", "format": {"specifier": ".2%"}},
                            {"name": "Temperature (°C)", "id": "temperature", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Efficiency", "id": "efficiency", "type": "numeric", "format": {"specifier": ".2%"}}
                        ],
                        data=df.tail(10).to_dict('records'),
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ]
                    )
                ])
            ], className="shadow-sm")
        ])
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  # Update every 10 seconds
        n_intervals=0
    ),
    
    # Store for optimization results
    dcc.Store(id='optimization-store', data=[])
], fluid=True)

# Callbacks for interactivity
@app.callback(
    [Output('energy-chart', 'figure'),
     Output('temperature-chart', 'figure'),
     Output('api-status', 'children'),
     Output('connection-alert', 'color')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(n):
    """Update charts with new data"""
    # Check API connection
    api_status = api_client.get_plant_status()
    if api_status:
        status_text = "Connected"
        alert_color = "success"
    else:
        status_text = "Disconnected"
        alert_color = "danger"
    
    # Generate new sample data
    new_data = generate_sample_data()
    
    # Update energy chart
    energy_fig = px.line(new_data.tail(24), x='timestamp', y='energy_consumption',
                        title="Last 24 Hours Energy Consumption",
                        color_discrete_sequence=['#28a745'])
    energy_fig.update_layout(showlegend=False, height=300)
    
    # Update temperature chart
    temp_fig = px.line(new_data.tail(24), x='timestamp', y='temperature',
                      title="Kiln Temperature Trend",
                      color_discrete_sequence=['#dc3545'])
    temp_fig.update_layout(showlegend=False, height=300)
    
    return energy_fig, temp_fig, status_text, alert_color

@app.callback(
    [Output('optimize-raw-btn', 'children'),
     Output('optimize-grinding-btn', 'children'),
     Output('optimize-clinker-btn', 'children'),
     Output('optimize-quality-btn', 'children')],
    [Input('optimize-raw-btn', 'n_clicks'),
     Input('optimize-grinding-btn', 'n_clicks'),
     Input('optimize-clinker-btn', 'n_clicks'),
     Input('optimize-quality-btn', 'n_clicks')],
    [State('moisture-slider', 'value'),
     State('temperature-slider', 'value'),
     State('fuel-slider', 'value'),
     State('fineness-slider', 'value')]
)
def run_optimizations(raw_clicks, grinding_clicks, clinker_clicks, quality_clicks,
                      moisture, temperature, fuel, fineness):
    """Handle optimization button clicks"""
    
    # Raw materials optimization
    if raw_clicks:
        data = {
            "timestamp": datetime.now().isoformat(),
            "limestone_quality": 0.85,
            "clay_content": 0.15,
            "iron_ore_grade": 0.75,
            "moisture_content": moisture,
            "particle_size_distribution": [0.2, 0.3, 0.25, 0.15, 0.1],
            "temperature": 25.0,
            "flow_rate": 150.0
        }
        result = api_client.optimize_raw_materials(data)
        if result:
            return "✅ Raw Materials Optimized!", "⚙️ Run Grinding Optimization", "🔥 Run Clinkerization Optimization", "🎯 Run Quality Optimization"
    
    # Grinding optimization
    if grinding_clicks:
        data = {
            "timestamp": datetime.now().isoformat(),
            "mill_power": 2000.0,
            "feed_rate": 100.0,
            "product_fineness": fineness,
            "energy_consumption": 1800.0,
            "temperature": 80.0,
            "vibration_level": 2.5,
            "noise_level": 95.0
        }
        result = api_client.optimize_grinding(data)
        if result:
            return "🚀 Run Raw Materials Optimization", "✅ Grinding Optimized!", "🔥 Run Clinkerization Optimization", "🎯 Run Quality Optimization"
    
    # Clinkerization optimization
    if clinker_clicks:
        data = {
            "timestamp": datetime.now().isoformat(),
            "kiln_temperature": temperature,
            "residence_time": 30.0,
            "fuel_consumption": 350.0,
            "alternative_fuel_ratio": fuel,
            "clinker_quality": 0.88,
            "exhaust_gas_temperature": 300.0,
            "oxygen_content": 5.0
        }
        result = api_client.optimize_clinkerization(data)
        if result:
            return "🚀 Run Raw Materials Optimization", "⚙️ Run Grinding Optimization", "✅ Clinkerization Optimized!", "🎯 Run Quality Optimization"
    
    # Quality optimization
    if quality_clicks:
        data = {
            "quality_variance": 0.06,
            "product_type": "OPC",
            "compressive_strength": 45.0,
            "fineness": fineness,
            "consistency": 0.90,
            "setting_time": 120.0,
            "temperature": 25.0,
            "humidity": 60.0
        }
        result = api_client.optimize_quality(data)
        if result:
            return "🚀 Run Raw Materials Optimization", "⚙️ Run Grinding Optimization", "🔥 Run Clinkerization Optimization", "✅ Quality Optimized!"
    
    return "🚀 Run Raw Materials Optimization", "⚙️ Run Grinding Optimization", "🔥 Run Clinkerization Optimization", "🎯 Run Quality Optimization"

@app.callback(
    Output('optimization-results', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_optimization_results(n):
    """Update optimization results display"""
    try:
        results = api_client.get_optimization_history()
        if not results:
            return dbc.Alert("No optimization results available", color="info")
        
        result_cards = []
        for result in results[-5:]:  # Show last 5 results
            card = dbc.Card([
                dbc.CardBody([
                    html.H6(f"Process: {result.get('process', 'Unknown')}", className="card-title"),
                    html.P(f"Expected Improvement: {result.get('expected_improvement', 0):.1%}"),
                    html.P(f"Confidence: {result.get('confidence_score', 0):.1%}"),
                    html.Small(f"Time: {result.get('timestamp', 'Unknown')}", className="text-muted")
                ])
            ], className="mb-2")
            result_cards.append(card)
        
        return result_cards
    except:
        return dbc.Alert("Error loading optimization results", color="warning")

if __name__ == '__main__':
    print("📊 Starting Cement Plant Optimization Dashboard...")
    print("🔗 Dashboard: http://localhost:8050")
    print("=" * 60)
    app.run_server(debug=True, host='0.0.0.0', port=8050)
