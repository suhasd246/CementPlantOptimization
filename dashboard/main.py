import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import numpy as np
import plotly.graph_objects as go
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
            response = requests.get(f"{API_BASE_URL}/optimization-history", timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return []

    def get_health(self):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def get_plant_status_history(self):
        try:
            response = requests.get(f"{API_BASE_URL}/plant-status-history", timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return []

    def _run_optimization(self, endpoint: str, data: dict):
        try:
            # Increased timeout to 30s for slow AI calls
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=90)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status_code": e.response.status_code if e.response else 503,
            }

    # Note: The generate_supervisor_report client method is no longer needed
    # as the backend handles this automatically in the optimize endpoints.

    def optimize_raw_materials(self, data: dict):
        return self._run_optimization("/optimize/raw-materials", data)

    def optimize_grinding(self, data: dict):
        return self._run_optimization("/optimize/grinding", data)

    def optimize_clinkerization(self, data: dict):
        return self._run_optimization("/optimize/clinkerization", data)

    def optimize_quality(self, data: dict):
        return self._run_optimization("/optimize/quality", data)


# --- App Initialization ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
)
app.title = "Cement Plant Optimization Dashboard"
api_client = APIClient()

# --- Reusable Components ---


def make_param_row(label, component):
    """Helper function for creating parameter rows."""
    return dbc.Row(
        [
            dbc.Col(html.Label(label), width=4, className="text-end"),
            dbc.Col(component, width=8),
        ],
        className="mb-2 align-items-center",
    )


def prettify_parameter_name(param_name: str) -> str:
    """Converts a machine_name to a Human Readable Name."""
    if not param_name:
        return "Value"
    return param_name.replace("_", " ").title()


def create_gauge(value, title, color="blue", max_val=1.0):
    """Creates a Plotly gauge figure."""
    if isinstance(value, str) and "%" in value:
        try:
            # Convert percentage string like "85.0%" to float 0.85
            value = float(value.strip("%")) / 100.0
        except ValueError:
            value = 0  # Default if conversion fails

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,  # Display as percentage
            number={"suffix": "%", "font": {"size": 20}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {
                    "range": [0, max_val * 100],
                    "tickwidth": 1,
                    "tickcolor": "darkblue",
                },
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {
                        "range": [0, max_val * 100 * 0.5],
                        "color": "lightgray",
                    },  # Low range
                    {
                        "range": [max_val * 100 * 0.5, max_val * 100 * 0.8],
                        "color": "gray",
                    },  # Mid range
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_val * 100 * 0.9,  # Example threshold for "good"
                },
            },
        )
    )
    fig.update_layout(
        height=160,  # Adjust height as needed
        margin=dict(l=5, r=5, t=45, b=5),  # Adjust margins
    )
    return fig


# Define the placeholder card as a global variable
placeholder_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                className="text-center h-100 d-flex flex-column justify-content-center align-items-center",
                children=[
                    html.I(className="fas fa-cogs fa-3x text-muted mb-3"),
                    html.H4("Waiting for Task", className="card-title"),
                    html.P(
                        "Select a process from the control panel and click optimize to see the AI's recommendation here.",
                        className="text-muted",
                    ),
                ],
            )
        ]
    ),
    className="h-100 border-0 bg-transparent",
    style={"min-height": "300px"},  # Give it a minimum height
)

# --- Dashboard Layout ---
app.layout = dbc.Container(
    [
        # Header and Status
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1(
                            "🏭 Cement Plant Optimization Platform",
                            className="text-center text-white mb-3",
                        ),
                        html.P(
                            "AI-driven autonomous operations for energy efficiency, quality control, and sustainability",
                            className="text-center text-white-50 mb-0",
                        ),
                    ],
                    className="bg-primary p-4 rounded mb-4 shadow",
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Alert(
                    [
                        html.Span("🔗 API Status: ", className="me-2"),
                        html.Span(
                            id="api-status",
                            children="Connecting...",
                            className="fw-bold",
                        ),
                    ],
                    id="connection-alert",
                    color="info",
                    className="mb-3",
                )
            )
        ),
        # KPIs and Live Chart
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Graph(
                                                id="efficiency-gauge",
                                                figure=create_gauge(
                                                    0, "⚡ Efficiency", "green"
                                                ),
                                            ),
                                            width=3,
                                        ),
                                        dbc.Col(
                                            dcc.Graph(
                                                id="quality-gauge",
                                                figure=create_gauge(
                                                    0, "🎯 Quality", "blue"
                                                ),
                                            ),
                                            width=3,
                                        ),
                                        dbc.Col(
                                            dcc.Graph(
                                                id="sustainability-gauge",
                                                figure=create_gauge(
                                                    0, "🌱 Sustainability", "orange"
                                                ),
                                            ),
                                            width=3,
                                        ),
                                        # Keep the simple Card for the Alerts count
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H4("⚠️ Alerts"),
                                                        html.H2(
                                                            id="alerts-metric",
                                                            children="-",
                                                            className="display-4",
                                                        ),  # Larger text
                                                    ]
                                                ),
                                                className="text-center text-danger h-100",
                                            ),  # Ensure card fills height
                                            width=3,
                                        ),
                                    ],
                                    className="mb-3 align-items-stretch",
                                ),
                                dcc.Graph(
                                    id="status-history-chart", style={"height": "300px"}
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # Control Panel & Results
        dbc.Row(
            [
                # --- Control Panel (Left) ---
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🎛️ AI Control Panel"),
                            dbc.CardBody(
                                [
                                    dbc.Accordion(
                                        [
                                            # Accordion Item for Raw Materials
                                            dbc.AccordionItem(
                                                children=[
                                                    make_param_row(
                                                        "Moisture Content",
                                                        dcc.Slider(
                                                            id="rm_moisture",
                                                            min=0.05,
                                                            max=0.25,
                                                            step=0.01,
                                                            value=0.18,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Limestone Quality",
                                                        dcc.Slider(
                                                            id="rm_limestone",
                                                            min=0.8,
                                                            max=1.0,
                                                            step=0.01,
                                                            value=0.92,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Flow Rate (t/h)",
                                                        dcc.Slider(
                                                            id="rm_flow",
                                                            min=300,
                                                            max=500,
                                                            step=10,
                                                            value=410,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    dbc.Button(
                                                        "🚀 Optimize Raw Materials",
                                                        id="optimize-raw-btn",
                                                        color="primary",
                                                        className="w-100 mt-3",
                                                    ),
                                                ],
                                                title=html.Span(
                                                    "🚀 Raw Materials Optimization",
                                                    className="fw-bold",
                                                ),
                                            ),
                                            # Accordion Item for Grinding
                                            dbc.AccordionItem(
                                                children=[
                                                    make_param_row(
                                                        "Product Fineness",
                                                        dcc.Slider(
                                                            id="gr_fineness",
                                                            min=0.8,
                                                            max=0.98,
                                                            step=0.01,
                                                            value=0.88,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Mill Power (kW)",
                                                        dcc.Slider(
                                                            id="gr_power",
                                                            min=4000,
                                                            max=5500,
                                                            step=100,
                                                            value=4800,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Feed Rate (t/h)",
                                                        dcc.Slider(
                                                            id="gr_feed",
                                                            min=180,
                                                            max=250,
                                                            step=5,
                                                            value=210,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    dbc.Button(
                                                        "⚙️ Optimize Grinding",
                                                        id="optimize-grinding-btn",
                                                        color="success",
                                                        className="w-100 mt-3",
                                                    ),
                                                ],
                                                title=html.Span(
                                                    "⚙️ Grinding Optimization",
                                                    className="fw-bold",
                                                ),
                                            ),
                                            # Accordion Item for Clinkerization
                                            dbc.AccordionItem(
                                                children=[
                                                    make_param_row(
                                                        "Kiln Temperature (°C)",
                                                        dcc.Slider(
                                                            id="cl_temp",
                                                            min=1380,
                                                            max=1520,
                                                            step=10,
                                                            value=1480,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Alternative Fuel Ratio",
                                                        dcc.Slider(
                                                            id="cl_fuel",
                                                            min=0.1,
                                                            max=0.5,
                                                            step=0.05,
                                                            value=0.20,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    dbc.Button(
                                                        "🔥 Optimize Clinkerization",
                                                        id="optimize-clinker-btn",
                                                        color="warning",
                                                        className="w-100 mt-3",
                                                    ),
                                                ],
                                                title=html.Span(
                                                    "🔥 Clinkerization Optimization",
                                                    className="fw-bold",
                                                ),
                                            ),
                                            # Accordion Item for Quality
                                            dbc.AccordionItem(
                                                children=[
                                                    make_param_row(
                                                        "Compressive Strength (MPa)",
                                                        dcc.Slider(
                                                            id="qu_strength",
                                                            min=40,
                                                            max=60,
                                                            step=1,
                                                            value=55,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Setting Time (min)",
                                                        dcc.Slider(
                                                            id="qu_setting_time",
                                                            min=30,
                                                            max=120,
                                                            step=5,
                                                            value=90,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    make_param_row(
                                                        "Gypsum Added (%)",
                                                        dcc.Slider(
                                                            id="qu_gypsum",
                                                            min=2.5,
                                                            max=5.5,
                                                            step=0.1,
                                                            value=4.5,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ),
                                                    dbc.Button(
                                                        "🎯 Optimize Quality",
                                                        id="optimize-quality-btn",
                                                        color="info",
                                                        className="w-100 mt-3",
                                                    ),
                                                ],
                                                title=html.Span(
                                                    "🎯 Quality Optimization",
                                                    className="fw-bold",
                                                ),
                                            ),
                                        ],
                                        always_open=False,
                                    )
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
                # --- Results Section (Right) ---
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("🤖 AI Optimization Result"),
                            dbc.CardBody(
                                # The Loading spinner now wraps the output Div
                                dcc.Loading(
                                    id="loading-results",
                                    type="default",
                                    children=[
                                        # The Div's children are set to the placeholder by default
                                        html.Div(
                                            id="optimization-results",
                                            children=placeholder_card,
                                        )
                                    ],
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    md=6,
                ),
            ],
            className="mb-4",
        ),
        # --- History Table ---
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("📜 AI Optimization Log"),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="optimization-history-table",
                                    columns=[
                                        {"name": "Time", "id": "timestamp"},
                                        {"name": "Process", "id": "process"},
                                        {"name": "Recommendation", "id": "action"},
                                        {
                                            "name": "Confidence",
                                            "id": "confidence_score",
                                            "type": "numeric",
                                            "format": dash.dash_table.Format.Format(
                                                precision=1,
                                                scheme=dash.dash_table.Format.Scheme.percentage,
                                            ),
                                        },
                                    ],
                                    style_cell={"textAlign": "left", "padding": "5px"},
                                    style_header={
                                        "backgroundColor": "#f8f9fa",
                                        "fontWeight": "bold",
                                    },
                                    page_size=5,
                                )
                            ),
                        ],
                        className="shadow-sm",
                    )
                )
            ],
            className="mb-4",
        ),
        # --- Interval & Storage Components ---
        dcc.Interval(
            id="interval-component", interval=10 * 1000, n_intervals=0
        ),  # 10 seconds
        dcc.Store(id="optimization-store"),  # Stores the *result* of the optimization
    ],
    fluid=True,
    className="bg-light p-4",
)


# --- Callbacks ---


@app.callback(
    [
        Output("api-status", "children"),
        Output("connection-alert", "color"),
        Output("status-history-chart", "figure"),
        Output("efficiency-gauge", "figure"),
        Output("quality-gauge", "figure"),
        Output("sustainability-gauge", "figure"),
        Output("alerts-metric", "children"),
        Output("optimization-history-table", "data"),
    ],
    [Input("interval-component", "n_intervals"), Input("optimization-store", "data")],
    [State("status-history-chart", "figure")],
)
def update_live_data_and_projections(n_intervals, optimization_data, current_fig):
    button_id = ctx.triggered_id

    # --- DEFAULT VALUES ---
    is_connected = api_client.get_health()
    no_data_eff_gauge = create_gauge(0, "⚡ Efficiency", "green")
    no_data_qual_gauge = create_gauge(0, "🎯 Quality", "blue")
    no_data_sus_gauge = create_gauge(0, "🌱 Sustainability", "orange")
    no_data_alerts = "-"
    no_data_fig = go.Figure(
        layout={
            "title": "Plant Performance Over Time",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": "Connecting or No Data...",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
        }
    )
    no_data_fig.update_layout(height=300, margin=dict(t=40, b=20, l=40, r=20))

    if not is_connected:
        kpis = (
            no_data_eff_gauge,
            no_data_qual_gauge,
            no_data_sus_gauge,
            no_data_alerts,
        )
        return "Disconnected", "danger", no_data_fig, *kpis, []

    # --- Fetch Live Data ---
    status_history = api_client.get_plant_status_history()
    optimization_log = api_client.get_optimization_history()

    # --- Process Log Table ---
    log_data = []
    if optimization_log:
        for log in optimization_log:
            action = (
                log["recommendations"][0]["action"]
                if log.get("recommendations")
                else "maintain_parameters"
            )
            log_data.append(
                {
                    "timestamp": pd.to_datetime(log["timestamp"]).strftime("%H:%M:%S"),
                    "process": log["process"].replace("_", " ").title(),
                    "action": action.replace("_", " ").title(),
                    "confidence_score": log["confidence_score"],
                }
            )

    # --- Process Status History ---
    if not status_history:
        no_data_fig.update_layout(
            annotations=[
                {
                    "text": "No Plant History Data From API",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ]
        )
        kpis = (
            no_data_eff_gauge,
            no_data_qual_gauge,
            no_data_sus_gauge,
            no_data_alerts,
        )
        return "Connected", "success", no_data_fig, *kpis, log_data

    # --- Prepare DataFrame ---
    df = pd.DataFrame(status_history)
    # Ensure timestamp is datetime for plotting
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.iloc[-1]

    # --- LOGIC FOR GAUGES (Default: Live Data) ---
    eff_val = latest["overall_efficiency"]
    qual_val = latest["quality_score"]
    sus_val = latest["sustainability_score"]
    alerts_val = str(latest["critical_alerts"])

    eff_gauge = create_gauge(eff_val, "⚡ Efficiency", "green")
    qual_gauge = create_gauge(qual_val, "🎯 Quality", "blue")
    sus_gauge = create_gauge(sus_val, "🌱 Sustainability", "orange")
    gauge_title_suffix = ""

    # --- LOGIC FOR GRAPH (Default: Scatter Plot with Trendlines) ---
    fig_title = "Plant Performance Over Time"
    df_melted = df.melt(id_vars=['timestamp'], value_vars=['overall_efficiency', 'quality_score', 'sustainability_score'],
                        var_name='Metric', value_name='Score')

    # --- ADD CHECK FOR SUFFICIENT DATA ---
    # Check if we have at least 2 unique timestamps
    sufficient_data_for_trend = df['timestamp'].nunique() >= 2

    try:
        if sufficient_data_for_trend:
            # Only add trendline if we have enough data
            fig = px.scatter(df_melted, x='timestamp', y='Score', color='Metric',
                             trendline="ols",
                             title=fig_title + " (Live with Trend)",
                             labels={'Score': 'Score', 'Metric': 'Metric'})
        else:
            # Plot without trendline if data is insufficient
            fig = px.scatter(df_melted, x='timestamp', y='Score', color='Metric',
                             title=fig_title + " (Live - Need More Data for Trend)",
                             labels={'Score': 'Score', 'Metric': 'Metric'})

    except Exception as e: # Catch other potential plotting errors
        print(f"Plotting failed: {e}")
        # Fallback to the empty figure defined earlier
        fig = no_data_fig
        # Ensure the empty figure has the correct layout
        fig.update_layout(height=300, margin=dict(t=40, b=20, l=40, r=20))


    # --- "WHAT-IF" SCENARIO logic remains the same ---
    if button_id == 'optimization-store' and optimization_data:
        # ... (Gauge update logic) ...

        # Update Graph Title appropriately
        if sufficient_data_for_trend:
             fig.update_layout(title="Plant Performance (Live Trend) - Projection Applied")
        else:
             fig.update_layout(title="Plant Performance (Live) - Projection Applied")
        # ... (Optional projected points logic) ...


    # --- Return all values ---
    fig.update_layout(height=300, legend_title_text='Metrics', margin=dict(t=40, b=20, l=40, r=20))
    kpis = (eff_gauge, qual_gauge, sus_gauge, alerts_val)

    return "Connected", "success", fig, *kpis, log_data


# Callback 2: Combined Optimization and Display (Handles the spinner)
@app.callback(
    [Output("optimization-results", "children"), Output("optimization-store", "data")],
    [
        Input("optimize-raw-btn", "n_clicks"),
        Input("optimize-grinding-btn", "n_clicks"),
        Input("optimize-clinker-btn", "n_clicks"),
        Input("optimize-quality-btn", "n_clicks"),
    ],
    [
        State("rm_moisture", "value"),
        State("rm_limestone", "value"),
        State("rm_flow", "value"),
        State("gr_fineness", "value"),
        State("gr_power", "value"),
        State("gr_feed", "value"),
        State("cl_temp", "value"),
        State("cl_fuel", "value"),
        State("qu_strength", "value"),
        State("qu_setting_time", "value"),
        State("qu_gypsum", "value"),
    ],
    prevent_initial_call=True,
)
def run_optimization_and_display(
    raw_clicks,
    grinding_clicks,
    clinker_clicks,
    quality_clicks,
    rm_moisture,
    rm_limestone,
    rm_flow,
    gr_fineness,
    gr_power,
    gr_feed,
    cl_temp,
    cl_fuel,
    qu_strength,
    qu_setting_time,
    qu_gypsum,
):

    button_id = ctx.triggered_id
    if not button_id:
        return (placeholder_card,), dash.no_update

    timestamp = datetime.now().isoformat()
    result_data = None

    if button_id == "optimize-raw-btn":
        data = {
            "timestamp": timestamp,
            "limestone_quality": rm_limestone,
            "clay_content": 0.12,
            "iron_ore_grade": 0.78,
            "moisture_content": rm_moisture,
            "particle_size_distribution": [0.2, 0.3, 0.25, 0.15, 0.1],
            "temperature": 28.0,
            "flow_rate": rm_flow,
        }
        result_data = api_client.optimize_raw_materials(data)
    elif button_id == "optimize-grinding-btn":
        data = {
            "timestamp": timestamp,
            "mill_power": gr_power,
            "feed_rate": gr_feed,
            "product_fineness": gr_fineness,
            "energy_consumption": gr_power * 0.85,
            "temperature": 105.0,
            "vibration_level": 3.1,
            "noise_level": 98.0,
        }
        result_data = api_client.optimize_grinding(data)
    elif button_id == "optimize-clinker-btn":
        data = {
            "timestamp": timestamp,
            "kiln_temperature": cl_temp,
            "residence_time": 28.0,
            "fuel_consumption": 3800.0,
            "alternative_fuel_ratio": cl_fuel,
            "clinker_quality": 0.95,
            "exhaust_gas_temperature": 350.0,
            "oxygen_content": 2.5,
        }
        result_data = api_client.optimize_clinkerization(data)
    elif button_id == "optimize-quality-btn":
        data = {
            "timestamp": timestamp,
            "product_type": "OPC-53",
            "compressive_strength": qu_strength,
            "fineness": gr_fineness,
            "consistency": 0.3,
            "setting_time": qu_setting_time,
            "temperature": 22.0,
            "humidity": 65.0,
            "gypsum_added": qu_gypsum,
        }
        result_data = api_client.optimize_quality(data)

    if not result_data:
        return (placeholder_card,), dash.no_update

    if "error" in result_data:
        alert = dbc.Alert(
            f"API Error: {result_data.get('error', 'Unknown error')}", color="danger"
        )
        return (alert,), result_data

    if not result_data.get("recommendations"):
        alert = dbc.Alert(
            "✅ No specific recommendations needed. Process is optimal.",
            color="success",
        )
        return (alert,), result_data

    # --- Build the unified result card ---
    rec = result_data["recommendations"][0]
    report_text = result_data.get(
        "report", "Report generation failed or is not available."
    )

    result_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    # --- Numerical Optimization Part ---
                    html.H5(rec["action"], className="card-title text-primary"),
                    html.P(f"Impact: {rec['impact']}", className="card-text"),
                    # This is the new, structured block
                    html.Div(
                        [
                            # Use the new function on the 'parameter' field
                            html.H6(
                                f"Parameter: {prettify_parameter_name(rec.get('parameter', ''))}",
                                className="card-subtitle text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Strong("Current: "),
                                                html.Span(
                                                    f"{rec['current_value']:.2f}",
                                                    className="fs-5 fw-bold text-danger ms-2",
                                                ),
                                            ]
                                        ),
                                        className="mt-2",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Strong("Target: "),
                                                html.Span(
                                                    f"{rec['target_value']:.2f}",
                                                    className="fs-5 fw-bold text-success ms-2",
                                                ),
                                            ]
                                        ),
                                        className="mt-2",
                                    ),
                                ]
                            ),
                        ],
                        className="bg-light p-3 rounded mb-3",
                    ),
                    html.Strong("Model Confidence"),
                    dbc.Progress(
                        label=f"{result_data['confidence_score']:.1%}",
                        value=result_data["confidence_score"] * 100,
                        className="mb-3",
                    ),
                    # --- LLM Report Part (Now included) ---
                    html.Hr(),
                    html.H6(
                        "📝 Operational Summary for Supervisor",
                        className="card-subtitle",
                    ),
                    dcc.Markdown(
                        report_text, className="border p-3 rounded bg-white mt-2"
                    ),
                ]
            )
        ],
        color="light",
    )

    # Return both the card for the UI AND the data for the store
    # We return as a tuple (item,) to match the [Output(...)] list
    return (result_card,), result_data


# --- Run the App ---
if __name__ == "__main__":
    print("📊 Starting Cement Plant Optimization Dashboard...")
    print("🔗 Live at: http://localhost:8050")
    app.run_server(debug=True, port=8050)
