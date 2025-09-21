import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from datetime import datetime
import requests
import json
import os

# --- Configuration & API Client ---
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

class APIClient:
    """A robust client to interact with the FastAPI backend."""
    def get_optimization_history(self):
        try:
            response = requests.get(f"{API_BASE_URL}/optimization-history", timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return []

    def get_health(self):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=3)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def get_plant_status_history(self):
        try:
            response = requests.get(f"{API_BASE_URL}/plant-status-history", timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return []

    def _run_optimization(self, endpoint: str, data: dict):
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": e.response.status_code if e.response else 503}
    
    def generate_supervisor_report(self, data: dict):
        try:
            response = requests.post(f"{API_BASE_URL}/generate-report", json=data, timeout=30)
            response.raise_for_status()
            return {"report": response.text}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def optimize_raw_materials(self, data: dict): return self._run_optimization("/optimize/raw-materials", data)
    def optimize_grinding(self, data: dict): return self._run_optimization("/optimize/grinding", data)
    def optimize_clinkerization(self, data: dict): return self._run_optimization("/optimize/clinkerization", data)
    def optimize_quality(self, data: dict): return self._run_optimization("/optimize/quality", data)

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
app.title = "Cement Plant Optimization Dashboard"
api_client = APIClient()

# --- Helper function for creating parameter rows ---
def make_param_row(label, component):
    return dbc.Row([
        dbc.Col(html.Label(label), width=4, className="text-end"),
        dbc.Col(component, width=8)
    ], className="mb-2 align-items-center")

# --- Dashboard Layout ---
app.layout = dbc.Container([
    # Header and Status
    dbc.Row(dbc.Col(html.Div([
        html.H1("🏭 Cement Plant Optimization Platform", className="text-center text-white mb-3"),
        html.P("AI-driven autonomous operations for energy efficiency, quality control, and sustainability", className="text-center text-white-50 mb-0")
    ], className="bg-primary p-4 rounded mb-4 shadow"))),
    dbc.Row(dbc.Col(dbc.Alert([
        html.Span("🔗 API Status: ", className="me-2"),
        html.Span(id="api-status", children="Connecting...", className="fw-bold")
    ], id="connection-alert", color="info", className="mb-3"))),
    
    # KPIs and Live Chart
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("⚡ Efficiency"), html.H2(id="efficiency-metric", children="- %")]), className="text-center text-success border-0"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("🎯 Quality"), html.H2(id="quality-metric", children="- %")]), className="text-center text-info border-0"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("🌱 Sustainability"), html.H2(id="sustainability-metric", children="- %")]), className="text-center text-warning border-0"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4("⚠️ Alerts"), html.H2(id="alerts-metric", children="-")]), className="text-center text-danger border-0"), width=3),
            ]),
            dcc.Graph(id='status-history-chart', style={'height': '300px'})
        ]), className="shadow-sm"), width=12)
    ], className="mb-4"),
    
    # --- MAJOR CHANGE: ACCORDION CONTROL PANEL ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("🎛️ AI Control Panel"),
            dbc.CardBody([
                dbc.Accordion([
                    # Accordion Item for Raw Materials
                    dbc.AccordionItem(children=[
                        make_param_row("Moisture Content", dcc.Slider(id='rm_moisture', min=0.05, max=0.25, step=0.01, value=0.18, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Limestone Quality", dcc.Slider(id='rm_limestone', min=0.8, max=1.0, step=0.01, value=0.92, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Flow Rate (t/h)", dcc.Slider(id='rm_flow', min=300, max=500, step=10, value=410, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        dbc.Button("🚀 Optimize Raw Materials", id="optimize-raw-btn", color="primary", className="w-100 mt-3")
                    ], title= html.Span("🚀 Raw Materials Optimization", className="fw-bold")),
                    
                    # Accordion Item for Grinding
                    dbc.AccordionItem(children=[
                        make_param_row("Product Fineness", dcc.Slider(id='gr_fineness', min=0.8, max=0.98, step=0.01, value=0.88, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Mill Power (kW)", dcc.Slider(id='gr_power', min=4000, max=5500, step=100, value=4800, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Feed Rate (t/h)", dcc.Slider(id='gr_feed', min=180, max=250, step=5, value=210, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        dbc.Button("⚙️ Optimize Grinding", id="optimize-grinding-btn", color="success", className="w-100 mt-3")
                    ], title= html.Span("⚙️ Grinding Optimization", className="fw-bold")),

                    # Accordion Item for Clinkerization
                    dbc.AccordionItem(children=[
                        make_param_row("Kiln Temperature (°C)", dcc.Slider(id='cl_temp', min=1380, max=1520, step=10, value=1480, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Alternative Fuel Ratio", dcc.Slider(id='cl_fuel', min=0.1, max=0.5, step=0.05, value=0.20, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        dbc.Button("🔥 Optimize Clinkerization", id="optimize-clinker-btn", color="warning", className="w-100 mt-3")
                    ], title= html.Span("🔥 Clinkerization Optimization", className="fw-bold")),

                    # Accordion Item for Quality
                    dbc.AccordionItem(children=[
                        make_param_row("Compressive Strength (MPa)", dcc.Slider(id='qu_strength', min=40, max=60, step=1, value=55, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        make_param_row("Setting Time (min)", dcc.Slider(id='qu_setting_time', min=30, max=120, step=5, value=90, marks=None, tooltip={"placement": "bottom", "always_visible": True})),
                        dbc.Button("🎯 Optimize Quality", id="optimize-quality-btn", color="info", className="w-100 mt-3")
                    ], title= html.Span("🎯 Quality Optimization", className="fw-bold"))
                ], always_open=False)
            ])
        ], className="shadow-sm"), md=6),
       
        # Results Section
        dbc.Col(dbc.Card([
            dbc.CardHeader("🤖 AI Optimization Result"),
            dbc.CardBody(
                dcc.Loading(id="loading-results", type="default", children=[
                    html.Div(id="optimization-results", children="Select a process and click optimize."),
                    html.Div(id="llm-report-output", className="mt-3")
                ])
            )
        ], className="shadow-sm"), md=6),
    ], className="mb-4"),
    ## History table
     dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📜 AI Optimization Log"),
            dbc.CardBody(
                dash_table.DataTable(
                    id='optimization-history-table',
                    columns=[
                        {"name": "Time", "id": "timestamp"},
                        {"name": "Process", "id": "process"},
                        {"name": "Recommendation", "id": "action"},
                        {"name": "Confidence", "id": "confidence_score", "type": "numeric", "format": dash.dash_table.Format.Format(precision=1, scheme=dash.dash_table.Format.Scheme.percentage)},
                    ],
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    page_size=5, # Show 5 rows at a time
                )
                )
            ], className="shadow-sm"))
        ], className="mb-4"),
    
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0),
    dcc.Store(id='optimization-store')
], fluid=True, className="bg-light p-4")


# --- Main Callbacks ---

# This callback for live data is the same
@app.callback(
    [Output('api-status', 'children'), Output('connection-alert', 'color'),
     Output('status-history-chart', 'figure'),
     Output('efficiency-metric', 'children'), Output('quality-metric', 'children'),
     Output('sustainability-metric', 'children'), Output('alerts-metric', 'children'),
     Output('optimization-history-table', 'data')], # New Output
    [Input('interval-component', 'n_intervals')]
)
def update_live_data(n):
    is_connected = api_client.get_health()
    no_data_fig = {"layout": {"title": "API Disconnected - No Data"}}
    no_data_kpis = ("-%", "-%", "-%", "-")
    
    if not is_connected:
        return "Disconnected", "danger", no_data_fig, *no_data_kpis, []

    # Fetch both history types
    status_history = api_client.get_plant_status_history()
    optimization_log = api_client.get_optimization_history()

    # --- Process Optimization Log for the Table ---
    log_data = []
    if optimization_log:
        for log in optimization_log:
            # If there are recommendations, show the first one. Otherwise, show 'None'.
            action = log['recommendations'][0]['action'] if log.get('recommendations') else "None"
            log_data.append({
                "timestamp": pd.to_datetime(log['timestamp']).strftime('%H:%M:%S'),
                "process": log['process'].replace('_', ' ').title(),
                "action": action,
                "confidence_score": log['confidence_score']
            })

    # --- Process Status History for Charts and KPIs ---
    if not status_history:
        no_data_fig["layout"]["title"] = "No Plant History Data From API"
        return "Connected", "success", no_data_fig, *no_data_kpis, log_data
        
    df = pd.DataFrame(status_history)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')

    latest = df.iloc[-1]
    kpis = (
        f"{latest['overall_efficiency']:.1%}", f"{latest['quality_score']:.1%}",
        f"{latest['sustainability_score']:.1%}", str(latest['critical_alerts'])
    )
    
    fig = px.line(df, x='timestamp', y=['overall_efficiency', 'quality_score', 'sustainability_score'],
                  title="Plant Performance Over Time", labels={'value': 'Score', 'variable': 'Metric'})
    fig.update_layout(height=300, legend_title_text='Metrics', margin=dict(t=30, b=10))
    
    return "Connected", "success", fig, *kpis, log_data

# <-- MAJOR CHANGE: New callback for the new accordion buttons -->
@app.callback(
    Output('optimization-store', 'data'),
    [Input('optimize-raw-btn', 'n_clicks'), Input('optimize-grinding-btn', 'n_clicks'),
     Input('optimize-clinker-btn', 'n_clicks'), Input('optimize-quality-btn', 'n_clicks')],
    [State('rm_moisture', 'value'), State('rm_limestone', 'value'), State('rm_flow', 'value'),
     State('gr_fineness', 'value'), State('gr_power', 'value'), State('gr_feed', 'value'),
     State('cl_temp', 'value'), State('cl_fuel', 'value'),
     State('qu_strength', 'value'), State('qu_setting_time', 'value')],
    prevent_initial_call=True
)
def run_optimizations(raw_clicks, grinding_clicks, clinker_clicks, quality_clicks,
                      rm_moisture, rm_limestone, rm_flow,
                      gr_fineness, gr_power, gr_feed,
                      cl_temp, cl_fuel,
                      qu_strength, qu_setting_time):
    button_id = ctx.triggered_id
    if not button_id: return dash.no_update
    
    timestamp = datetime.now().isoformat()
    result = None

    if button_id == 'optimize-raw-btn':
        data = { "timestamp": timestamp, "limestone_quality": rm_limestone, "clay_content": 0.12, "iron_ore_grade": 0.78, "moisture_content": rm_moisture, "particle_size_distribution": [0.2,0.3,0.25,0.15,0.1], "temperature": 28.0, "flow_rate": rm_flow }
        result = api_client.optimize_raw_materials(data)
    elif button_id == 'optimize-grinding-btn':
        data = { "timestamp": timestamp, "mill_power": gr_power, "feed_rate": gr_feed, "product_fineness": gr_fineness, "energy_consumption": gr_power * 0.85, "temperature": 105.0, "vibration_level": 3.1, "noise_level": 98.0 }
        result = api_client.optimize_grinding(data)
    elif button_id == 'optimize-clinker-btn':
        data = { "timestamp": timestamp, "kiln_temperature": cl_temp, "residence_time": 28.0, "fuel_consumption": 3800.0, "alternative_fuel_ratio": cl_fuel, "clinker_quality": 0.95, "exhaust_gas_temperature": 350.0, "oxygen_content": 2.5 }
        result = api_client.optimize_clinkerization(data)
    elif button_id == 'optimize-quality-btn':
        data = { "timestamp": timestamp, "product_type": "OPC-53", "compressive_strength": qu_strength, "fineness": gr_fineness, "consistency": 0.3, "setting_time": qu_setting_time, "temperature": 22.0, "humidity": 65.0 }
        result = api_client.optimize_quality(data)
        
    return result

@app.callback(
    [Output('optimization-results', 'children'),
     Output('llm-report-output', 'children')],
    [Input('optimization-store', 'data')]
)
def display_optimization_results(data):
    # <-- NEW: A beautiful placeholder card for the initial state -->
    placeholder = dbc.Card(
        dbc.CardBody([
            html.Div(className="text-center h-100 d-flex flex-column justify-content-center align-items-center", children=[
                html.I(className="fas fa-cogs fa-3x text-muted mb-3"),
                html.H4("Waiting for Task", className="card-title"),
                html.P(
                    "Select a process from the control panel and click optimize to see the AI's recommendation here.",
                    className="text-muted"
                )
            ])
        ]),
        className="h-100 border-0 bg-transparent" # Style to blend in
    )

    if not data:
        return placeholder, "" # Return the placeholder on initial load
    
    if "error" in data:
        return dbc.Alert(f"API Error: {data.get('error', 'Unknown error')}", color="danger"), ""
    
    if not data.get('recommendations'):
        return dbc.Alert("✅ No specific recommendations needed. Process is optimal.", color="success"), ""
    
    # This part for displaying the actual result remains the same
    rec = data['recommendations'][0]
    result_card = dbc.Card([
        dbc.CardBody([
            html.H5(rec['action'], className="card-title text-primary"),
            html.P(f"Impact: {rec['impact']}", className="card-text"),
            dbc.Row([
                dbc.Col(html.P([html.Strong("Current: "), f"{rec['current_value']:.2f}"])),
                dbc.Col(html.P([html.Strong("Target: "), f"{rec['target_value']:.2f}"])),
            ]),
            html.Hr(),
            html.Strong("Model Confidence"),
            dbc.Progress(label=f"{data['confidence_score']:.1%}", value=data['confidence_score']*100, className="mb-3"),
            dbc.Button("📝 Generate Supervisor's Report", id="generate-report-btn", color="secondary", className="w-100")
        ])
    ], color="light")
    return result_card, ""

# <-- NEW CALLBACK: Triggered by the "Generate Report" button -->
@app.callback(
    Output('llm-report-output', 'children', allow_duplicate=True),
    Input('generate-report-btn', 'n_clicks'),
    State('optimization-store', 'data'),
    prevent_initial_call=True
)
def generate_llm_report(n_clicks, data):
    if not n_clicks or not data:
        return dash.no_update
    
    spinner = dbc.Spinner(color="primary")
    
    # Immediately show spinner
    # In a real app, you might use a clientside callback for a faster spinner
    
    report_data = api_client.generate_supervisor_report(data)
    
    if "error" in report_data:
        return dbc.Alert(f"LLM Error: {report_data['error']}", color="danger")
        
    report_text = report_data.get("report", "No report generated.")
    
    return dcc.Markdown(report_text, className="border p-3 rounded bg-white")


# --- Run the App ---
if __name__ == '__main__':
    print("📊 Starting Cement Plant Optimization Dashboard...")
    print("🔗 Live at: http://localhost:8050")
    app.run_server(debug=True, port=8050)