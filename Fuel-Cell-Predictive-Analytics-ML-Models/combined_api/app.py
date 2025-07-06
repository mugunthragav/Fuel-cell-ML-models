import dash
from dash import dcc, html, Input, Output, State, ALL, dash_table
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# API base URL
API_URL = "http://localhost:8000"

# Plot directory (relative path)
PLOT_DIR = "./plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Model input fields with full names, descriptions, and instructions
MODEL_INPUTS = {
    'Anomaly Detection Stack Voltage': {
        'description': 'Detects anomalies in stack voltage data using DBSCAN clustering, identifying outliers in cell voltages and operating conditions.',
        'json': None,
        'file_endpoint': '/model1/detect_anomalies/file',
        'instructions': (
            'Upload a CSV or XLSX file with columns: Date_Time, Load, Loaddensity, H2Flow, TstackIn, TstackOut, '
            'CoolantFlow, AnodePressureDiff, CathodePressureDiff, RH_An, RH_Cat, and Cell1 to Cell96. '
            'Click Generate Report to view anomaly count, cell distribution plot, and anomaly table. '
            'Or select a cell number to view its specific anomaly table and time-series plot.'
        )
    },
    'Health State Classification': {
        'description': 'Classifies the health state of the fuel cell (Critical, Degraded, Healthy) based on operating conditions using a Random Forest model.',
        'json': [
            {'name': 'Time_h', 'full_name': 'Time (hours)', 'type': 'number', 'default': 1.0},
            {'name': 'TinH2', 'full_name': 'Hydrogen Inlet Temperature (°C)', 'type': 'number', 'default': 60.0},
            {'name': 'ToutH2', 'full_name': 'Hydrogen Outlet Temperature (°C)', 'type': 'number', 'default': 62.0},
            {'name': 'TinAIR', 'full_name': 'Air Inlet Temperature (°C)', 'type': 'number', 'default': 25.0},
            {'name': 'ToutAIR', 'full_name': 'Air Outlet Temperature (°C)', 'type': 'number', 'default': 30.0},
            {'name': 'TinWAT', 'full_name': 'Water Inlet Temperature (°C)', 'type': 'number', 'default': 55.0},
            {'name': 'ToutWAT', 'full_name': 'Water Outlet Temperature (°C)', 'type': 'number', 'default': 57.0},
            {'name': 'PinAIR', 'full_name': 'Air Inlet Pressure (mbara)', 'type': 'number', 'default': 1500.0},
            {'name': 'PoutAIR', 'full_name': 'Air Outlet Pressure (mbara)', 'type': 'number', 'default': 1400.0},
            {'name': 'PoutH2', 'full_name': 'Hydrogen Outlet Pressure (mbara)', 'type': 'number', 'default': 1450.0},
            {'name': 'PinH2', 'full_name': 'Hydrogen Inlet Pressure (mbara)', 'type': 'number', 'default': 1500.0},
            {'name': 'DinH2', 'full_name': 'Hydrogen Inlet Flow Rate (l/min)', 'type': 'number', 'default': 0.8},
            {'name': 'DoutH2', 'full_name': 'Hydrogen Outlet Flow Rate (l/min)', 'type': 'number', 'default': 0.1},
            {'name': 'DinAIR', 'full_name': 'Air Inlet Flow Rate (l/min)', 'type': 'number', 'default': 300.0},
            {'name': 'DoutAIR', 'full_name': 'Air Outlet Flow Rate (l/min)', 'type': 'number', 'default': 290.0},
            {'name': 'DWAT', 'full_name': 'Water Flow Rate (l/min)', 'type': 'number', 'default': 2.0},
            {'name': 'HrAIRFC', 'full_name': 'Air Relative Humidity at Fuel Cell (%)', 'type': 'number', 'default': 50.0}
        ],
        'file_endpoint': '/model2/predict_health/file',
        'json_endpoint': '/model2/predict_health/json',
        'instructions': (
            'For JSON input, enter values for all fields and click Submit JSON to see the health state (e.g., Healthy). '
            'For file input, upload a CSV or XLSX file with columns matching the model’s required features and click Submit File to see a table.'
        )
    },
    'Time Series Anomaly Detection': {
        'description': 'Detects anomalies in time-series data (current and voltage) using a transformer autoencoder and Isolation Forest.',
        'json': None,
        'file_endpoint': '/model3/detect_anomalies',
        'instructions': (
            'Upload a CSV file with columns: time, current, voltage. Time will be treated as seconds (1, 2, 3, ..., n). '
            'Click Submit File to view anomaly report and plots highlighting anomalies.'
        )
    },
    'Transient Current Detection': {
        'description': 'Predicts whether a transient current event is Normal (0) or Faulty (1 or 2) using a Random Forest classifier.',
        'json': [
            {'name': 'I', 'full_name': 'Current (A)', 'type': 'number', 'default': 105.0, 'min': 0, 'max': 200},
            {'name': 'ARF', 'full_name': 'Air Flow Rate (l/min)', 'type': 'number', 'default': 282.0, 'min': 0, 'max': 500},
            {'name': 'AIP1', 'full_name': 'Anode Inlet Pressure 1 (bar)', 'type': 'number', 'default': 1776.0, 'min': 0, 'max': 3000},
            {'name': 'AIP2', 'full_name': 'Anode Inlet Pressure 2 (bar)', 'type': 'number', 'default': 1776.0, 'min': 0, 'max': 3000},
            {'name': 'CAIF', 'full_name': 'Cathode Air Inlet Flow (l/min)', 'type': 'number', 'default': 1692.0, 'min': 0, 'max': 2000},
            {'name': 'CIP1', 'full_name': 'Cathode Inlet Pressure 1 (bar)', 'type': 'number', 'default': 1256.0, 'min': 0, 'max': 2000},
            {'name': 'CS', 'full_name': 'Current Stack (A)', 'type': 'number', 'default': 2.52, 'min': 0, 'max': 10},
            {'name': 'COT2', 'full_name': 'Cathode Outlet Temperature 2 (°C)', 'type': 'number', 'default': 75.0, 'min': 0, 'max': 100},
            {'name': 'CIT2', 'full_name': 'Cathode Inlet Temperature 2 (°C)', 'type': 'number', 'default': 45.0, 'min': 0, 'max': 100},
            {'name': 'COT1', 'full_name': 'Cathode Outlet Temperature 1 (°C)', 'type': 'number', 'default': 75.0, 'min': 0, 'max': 100},
            {'name': 'CIT1', 'full_name': 'Cathode Inlet Temperature 1 (°C)', 'type': 'number', 'default': 46.0, 'min': 0, 'max': 100},
            {'name': 'WIP2', 'full_name': 'Water Inlet Pressure 2 (bar)', 'type': 'number', 'default': 1936.0, 'min': 0, 'max': 3000},
            {'name': 'WIP1', 'full_name': 'Water Inlet Pressure 1 (bar)', 'type': 'number', 'default': 1912.0, 'min': 0, 'max': 3000},
            {'name': 'WIF2', 'full_name': 'Water Inlet Flow 2 (l/min)', 'type': 'number', 'default': 820.0, 'min': 0, 'max': 1000},
            {'name': 'WIF1', 'full_name': 'Water Inlet Flow 1 (l/min)', 'type': 'number', 'default': 820.0, 'min': 0, 'max': 1000},
            {'name': 'WIT', 'full_name': 'Water Inlet Temperature (°C)', 'type': 'number', 'default': 37.0, 'min': 0, 'max': 100}
        ],
        'json_endpoint': '/model4/predict',
        'instructions': (
            'Enter values for all 16 fields representing operating conditions. Ensure all fields are filled with valid numerical values within the specified ranges. '
            'Click Submit JSON to predict if the transient current is Normal or Faulty.'
        )
    },
    'Voltage Prediction Digital Twin': {
        'description': 'Predicts the total stack voltage (Utot) using an XGBoost model based on operating conditions.',
        'json': [
            {'name': 'Time_h', 'full_name': 'Time (hours)', 'type': 'number', 'default': 1.0},
            {'name': 'TinH2', 'full_name': 'Hydrogen Inlet Temperature (°C)', 'type': 'number', 'default': 60.0},
            {'name': 'ToutH2', 'full_name': 'Hydrogen Outlet Temperature (°C)', 'type': 'number', 'default': 62.0},
            {'name': 'TinAIR', 'full_name': 'Air Inlet Temperature (°C)', 'type': 'number', 'default': 25.0},
            {'name': 'ToutAIR', 'full_name': 'Air Outlet Temperature (°C)', 'type': 'number', 'default': 30.0},
            {'name': 'TinWAT', 'full_name': 'Water Inlet Temperature (°C)', 'type': 'number', 'default': 55.0},
            {'name': 'ToutWAT', 'full_name': 'Water Outlet Temperature (°C)', 'type': 'number', 'default': 57.0},
            {'name': 'PinAIR', 'full_name': 'Air Inlet Pressure (mbara)', 'type': 'number', 'default': 1500.0},
            {'name': 'PoutAIR', 'full_name': 'Air Outlet Pressure (mbara)', 'type': 'number', 'default': 1400.0},
            {'name': 'PoutH2', 'full_name': 'Hydrogen Outlet Pressure (mbara)', 'type': 'number', 'default': 1450.0},
            {'name': 'PinH2', 'full_name': 'Hydrogen Inlet Pressure (mbara)', 'type': 'number', 'default': 1500.0},
            {'name': 'DinH2', 'full_name': 'Hydrogen Inlet Flow Rate (l/min)', 'type': 'number', 'default': 0.8},
            {'name': 'DoutH2', 'full_name': 'Hydrogen Outlet Flow Rate (l/min)', 'type': 'number', 'default': 0.1},
            {'name': 'DinAIR', 'full_name': 'Air Inlet Flow Rate (l/min)', 'type': 'number', 'default': 300.0},
            {'name': 'DoutAIR', 'full_name': 'Air Outlet Flow Rate (l/min)', 'type': 'number', 'default': 290.0},
            {'name': 'DWAT', 'full_name': 'Water Flow Rate (l/min)', 'type': 'number', 'default': 2.0},
            {'name': 'HrAIRFC', 'full_name': 'Air Relative Humidity at Fuel Cell (%)', 'type': 'number', 'default': 50.0}
        ],
        'file_endpoint': '/model5/predict_utot/file',
        'json_endpoint': '/model5/predict_utot/json',
        'instructions': (
            'For JSON input, enter values for all fields and click Submit JSON to see the predicted Utot. '
            'For file input, upload a CSV or XLSX file with columns matching the model’s required features and click Submit File to see a table.'
        )
    }
}

# Layout
app.layout = html.Div(className="container-fluid p-4 bg-light min-vh-100", children=[
    html.H1("Fuel Cell Model Dashboard", className="text-center text-primary mb-4"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model, 'value': model} for model in MODEL_INPUTS.keys()],
        value='Anomaly Detection Stack Voltage',
        className="mb-4 w-50 mx-auto"
    ),
    html.Div(id='input-form', className="card p-4 mb-4"),
    html.Div(id='upload-status', className="text-muted mb-3"),
    html.Div(id='output-container', className="card p-4"),
    dcc.Store(id='table-data-store'),
    dcc.Store(id='uploaded-data-store'),
    dcc.Download(id="download-dataframe-csv")
])

# Callback to update input form based on model selection
@app.callback(
    Output('input-form', 'children'),
    Input('model-dropdown', 'value')
)
def update_input_form(selected_model):
    inputs = MODEL_INPUTS[selected_model]
    form_children = [
        html.H5("Model Description", className="card-title"),
        html.P(inputs['description'], className="card-text mb-3")
    ]

    if inputs.get('json'):
        form_children.extend([
            html.H6("JSON Input", className="mt-3"),
            html.Div([
                html.Div([
                    html.Label(field['full_name'], className="form-label"),
                    dcc.Input(
                        id={'type': 'dynamic-input', 'index': field['name']},
                        type=field['type'],
                        value=field['default'],
                        className="form-control mb-2",
                        min=field.get('min'),
                        max=field.get('max')
                    )
                ], className="col-md-6 col-12") for field in inputs['json']
            ], className="row g-3"),
            html.Div([
                html.Button(
                    [html.I(className="fas fa-code me-2"), "Submit JSON"],
                    id={'type': 'submit-json', 'index': selected_model},
                    n_clicks=0,
                    className="btn btn-primary mt-3",
                    style={'z-index': '1000', 'padding': '8px 16px'}
                )
            ], className="d-flex justify-content-end")
        ])

    if inputs.get('file_endpoint'):
        form_children.extend([
            html.H6("File Upload (CSV/XLSX)", className="mt-3"),
            dcc.Upload(
                id={'type': 'dynamic-upload', 'index': 'upload-data'},
                children=html.Button(
                    [html.I(className="fas fa-upload me-2"), "Upload File"],
                    className="btn btn-success"
                ),
                className="mb-3"
            ),
            html.Div([
                html.Button(
                    [html.I(className="fas fa-file-upload me-2"), "Generate Report" if selected_model == 'Anomaly Detection Stack Voltage' else "Submit File"],
                    id={'type': 'submit-file', 'index': selected_model},
                    n_clicks=0,
                    className="btn btn-info",
                    style={'z-index': '1000', 'padding': '8px 16px'}
                )
            ], className="d-flex justify-content-end"),
            html.Div([
                html.Label("Select Cell for Anomaly View", className="form-label mt-3"),
                dcc.Dropdown(
                    id='cell-dropdown',
                    options=[{'label': f'Cell{i}', 'value': f'Cell{i}'} for i in range(1, 97)],
                    value=None,
                    placeholder="Select a cell",
                    className="form-control"
                )
            ], className="mt-3", id='cell-dropdown-container', style={'display': 'block' if selected_model == 'Anomaly Detection Stack Voltage' else 'none'})
        ])
        logger.debug(f"Rendered {'Generate Report' if selected_model == 'Anomaly Detection Stack Voltage' else 'Submit File'} button for {selected_model}")

    form_children.append(html.H6("Instructions", className="mt-3"))
    form_children.append(html.P(inputs['instructions'], className="card-text"))
    return form_children

# Callback to update upload status
@app.callback(
    Output('upload-status', 'children'),
    Input({'type': 'dynamic-upload', 'index': ALL}, 'filename'),
    State('model-dropdown', 'value')
)
def update_upload_status(filenames, selected_model):
    button_label = "Generate Report" if selected_model == 'Anomaly Detection Stack Voltage' else "Submit File"
    if filenames and filenames[0]:
        return html.P(f"File '{filenames[0]}' uploaded, click {button_label}.", className="text-success")
    return html.P("No file uploaded.", className="text-muted")

# Callback to handle file and JSON submissions
@app.callback(
    [Output('output-container', 'children'),
     Output('table-data-store', 'data'),
     Output('uploaded-data-store', 'data'),
     Output('download-dataframe-csv', 'data')],
    [Input({'type': 'submit-json', 'index': ALL}, 'n_clicks'),
     Input({'type': 'submit-file', 'index': ALL}, 'n_clicks'),
     Input({'type': 'download-button', 'index': ALL}, 'n_clicks')],
    [State('model-dropdown', 'value'),
     State({'type': 'dynamic-upload', 'index': ALL}, 'contents'),
     State({'type': 'dynamic-upload', 'index': ALL}, 'filename'),
     State({'type': 'dynamic-input', 'index': ALL}, 'value'),
     State('table-data-store', 'data'),
     State('uploaded-data-store', 'data')],
    prevent_initial_call=True
)
def update_output(json_n_clicks, file_n_clicks, download_n_clicks, selected_model, upload_contents, upload_filenames, input_values, stored_table_data, stored_uploaded_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div("Select a model and provide inputs, then click Submit JSON or Submit File.", className="text-muted"), None, None, None

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    triggered_type = json.loads(triggered_id)['type'] if triggered_id else None
    logger.debug(f"Triggered button: {triggered_type}, Model: {selected_model}")

    inputs = MODEL_INPUTS[selected_model]
    json_inputs = inputs.get('json', []) if inputs.get('json') is not None else []
    input_dict = {field['name']: float(value) if value is not None and value != '' else field['default'] for field, value in zip(json_inputs, input_values)}

    # Handle download button click
    if triggered_type == 'download-button' and stored_table_data and selected_model != 'Anomaly Detection Stack Voltage':
        return dash.no_update, dash.no_update, dash.no_update, dcc.send_data_frame(pd.DataFrame(stored_table_data).to_csv, f"{selected_model.replace(' ', '_')}_results.csv")

    try:
        output = []
        table_data = []
        uploaded_data = stored_uploaded_data
        csv_data = None

        # Handle file submission
        if triggered_type == 'submit-file' and inputs.get('file_endpoint'):
            content = upload_contents[0] if upload_contents else None
            filename = upload_filenames[0] if upload_filenames else None
            if not content or not filename:
                return html.Div("No file uploaded. Please upload a valid CSV or XLSX file and click Submit File.", className="text-danger"), None, None, None

            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            if filename.endswith('.csv'):
                df = pd.read_csv(BytesIO(decoded))
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(BytesIO(decoded))
            else:
                return html.Div("Unsupported file format. Use CSV or XLSX.", className="text-danger"), None, None, None

            uploaded_data = df.to_dict('records')
            files = {'file': (filename, BytesIO(decoded), 'multipart/form-data')}
            if selected_model == 'Time Series Anomaly Detection':
                response = requests.post(f"{API_URL}{inputs['file_endpoint']}", files=files, data={'n_seconds': len(df)})
            else:
                response = requests.post(f"{API_URL}{inputs['file_endpoint']}", files=files)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"API Response for {selected_model}: {json.dumps(result, indent=2)}")

            if selected_model == 'Anomaly Detection Stack Voltage':
                if 'anomalies' in result and result['anomalies'] and isinstance(result['anomalies'], list):
                    table_data = result['anomalies']
                    output.append(html.P(f"{result['summary']['total_anomalies']} anomalies detected", className="text-muted"))
                    df_anomalies = pd.DataFrame(table_data)
                    cell_counts = df_anomalies['Min_Cell_Index'].value_counts().reset_index()
                    cell_counts.columns = ['Min_Cell_Index', 'Count']
                    all_cells = pd.DataFrame({'Min_Cell_Index': [f'Cell{i}' for i in range(1, 97)]})
                    cell_counts = all_cells.merge(cell_counts, on='Min_Cell_Index', how='left').fillna({'Count': 0})
                    cell_counts['Cell_Number'] = cell_counts['Min_Cell_Index'].str.extract('(\d+)').astype(int)
                    cell_counts = cell_counts.sort_values('Cell_Number')
                    fig_counts = px.bar(
                        cell_counts,
                        x='Min_Cell_Index',
                        y='Count',
                        title="Anomaly Count by Cell Number",
                        labels={'Min_Cell_Index': 'Cell Number', 'Count': 'Anomaly Count'}
                    )
                    count_plot_path = os.path.join(PLOT_DIR, "anomaly_cell_distribution.png")
                    fig_counts.write_image(count_plot_path)
                    output.append(dcc.Graph(figure=fig_counts))
                    output.append(dash_table.DataTable(
                        data=table_data,
                        columns=[{'name': col, 'id': col} for col in table_data[0].keys()] if table_data else [],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        page_size=10,
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                        ]
                    ))
                else:
                    output.append(html.P("No anomalies detected.", className="text-warning"))

            elif selected_model == 'Health State Classification':
                table_data = result.get('predictions', []) or [result.get('result', {})]
                output.append(html.P(f"Detected health states: {result.get('summary', {}).get('health_state_counts', 'N/A')}", className="text-muted"))
                output.append(dash_table.DataTable(
                    data=table_data,
                    columns=[{'name': col, 'id': col} for col in table_data[0].keys()] if table_data else [],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10,
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                    ]
                ))
                output.append(html.Button(
                    "Download CSV",
                    id={'type': 'download-button', 'index': selected_model},
                    className="btn btn-primary mt-3"
                ))

            elif selected_model == 'Time Series Anomaly Detection':
                anomaly_percentage = result['summary'].get('anomaly_percentage', 0.0)
                output.append(html.P(f"Anomaly Report: Detected {anomaly_percentage:.2f}% anomalies ({len(result.get('anomaly_indices', []))} total)", className="text-muted"))
                df_all = pd.DataFrame(uploaded_data)
                df_all['time_index'] = range(1, len(df_all) + 1)
                anomaly_indices = result.get('anomaly_indices', [])
                df_all['is_anomaly'] = df_all.index.isin(anomaly_indices)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_all['time_index'],
                    y=df_all['voltage'] * df_all['current'],
                    mode='markers',
                    name='Normal Points',
                    marker=dict(color='blue', size=8),
                    customdata=df_all[['time', 'current', 'voltage']],
                    hovertemplate='<b>Time:</b> %{customdata[0]}<br><b>Current:</b> %{customdata[1]}A<br><b>Voltage:</b> %{customdata[2]}V<br>'
                ))
                if anomaly_indices:
                    df_anomalies = df_all[df_all['is_anomaly']].copy()
                    fig.add_trace(go.Scatter(
                        x=df_anomalies['time_index'],
                        y=df_anomalies['voltage'] * df_anomalies['current'],
                        mode='markers',
                        name='Anomaly Points',
                        marker=dict(color='red', size=12),
                        customdata=df_anomalies[['time', 'current', 'voltage']],
                        hovertemplate='<b>Time:</b> %{customdata[0]}<br><b>Current:</b> %{customdata[1]}A<br><b>Voltage:</b> %{customdata[2]}V<br>'
                    ))
                fig.update_layout(
                    title="Time Series Power with Anomalies",
                    xaxis_title="Time Index",
                    yaxis_title="Power (W)",
                    showlegend=True
                )
                plot_path = os.path.join(PLOT_DIR, "time_series_power_plot.png")
                fig.write_image(plot_path)
                output.append(dcc.Graph(figure=fig))
                if anomaly_indices:
                    table_data = df_anomalies.to_dict('records')
                    output.append(dash_table.DataTable(
                        data=table_data,
                        columns=[{'name': col, 'id': col} for col in df_all.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        page_size=10,
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                        ]
                    ))
                    output.append(html.Button(
                        "Download CSV",
                        id={'type': 'download-button', 'index': selected_model},
                        className="btn btn-primary mt-3"
                    ))
                else:
                    logger.debug(f"No anomalies in Time Series response: {json.dumps(result, indent=2)}")
                    output.append(html.P("No anomalies detected.", className="text-warning"))

            elif selected_model == 'Voltage Prediction Digital Twin':
                table_data = result.get('predictions', []) or [result.get('result', {})]
                output.append(html.P(f"Mean Utot Predicted: {result.get('summary', {}).get('mean_utot_predicted', 'N/A'):.2f}", className="text-muted"))
                output.append(dash_table.DataTable(
                    data=table_data,
                    columns=[{'name': col, 'id': col} for col in table_data[0].keys()] if table_data else [],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10,
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                    ]
                ))
                output.append(html.Button(
                    "Download CSV",
                    id={'type': 'download-button', 'index': selected_model},
                    className="btn btn-primary mt-3"
                ))

        # Handle JSON submission
        elif triggered_type == 'submit-json' and inputs.get('json_endpoint') and json_inputs:
            if not all(v is not None and v != '' and isinstance(float(v) if v else 0, (int, float)) for v in input_values):
                return html.Div("All JSON fields must be filled with valid numerical values within the specified ranges.", className="text-danger"), None, None, None
            if len(input_dict) != len(json_inputs):
                return html.Div("All fields must be provided with valid numerical values.", className="text-danger"), None, None, None
            logger.debug(f"Sending JSON payload for {selected_model}: {json.dumps(input_dict, indent=2)}")
            payload = input_dict if selected_model != 'Transient Current Detection' else {'features': input_dict}
            response = requests.post(f"{API_URL}{inputs['json_endpoint']}", json=payload)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"API Response for {selected_model}: {json.dumps(result, indent=2)}")

            if selected_model == 'Health State Classification':
                health_state = result['result'].get('Health_State_Label')
                if health_state is None:
                    logger.error(f"Health_State_Label is None in API response: {json.dumps(result, indent=2)}")
                    output.append(html.P("Error: No health state received from API.", className="text-danger"))
                elif not isinstance(health_state, str):
                    logger.error(f"Health_State_Label is not a string: {health_state}")
                    output.append(html.P(f"Error: Invalid health state: {health_state}", className="text-danger"))
                else:
                    output.append(html.P(f"Health State: {health_state}", className="text-success h4"))

            elif selected_model == 'Transient Current Detection':
                prediction = result['prediction_status']
                label = result['prediction']
                confidence = result.get('confidence', 0.0)
                output.append(html.P(f"Prediction: {label} (Confidence: {confidence:.2f})", className="text-success h4"))
                if prediction != "Success":
                    logger.warning(f"Prediction failed: {json.dumps(result, indent=2)}")

            elif selected_model == 'Voltage Prediction Digital Twin':
                utot = result['result'].get('Utot_Predicted')
                if utot is None:
                    logger.error(f"Utot_Predicted is None in API response: {json.dumps(result, indent=2)}")
                    output.append(html.P("Error: No Utot prediction received from API.", className="text-danger"))
                elif not isinstance(utot, (int, float)):
                    logger.error(f"Utot_Predicted is not a number: {utot}")
                    output.append(html.P(f"Error: Invalid Utot prediction: {utot}", className="text-danger"))
                else:
                    output.append(html.P(f"Utot Predicted: {utot:.2f}", className="text-success h4"))

        return html.Div(output, className="mt-3"), table_data, uploaded_data, None

    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {str(e)}")
        return html.Div(f"API Error: {str(e)}", className="text-danger"), None, None, None
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="text-danger"), None, None, None

# Separate callback for cell-specific anomaly view (Model 1 only)
@app.callback(
    Output('output-container', 'children', allow_duplicate=True),
    Input('cell-dropdown', 'value'),
    State('model-dropdown', 'value'),
    State('table-data-store', 'data'),
    State('uploaded-data-store', 'data'),
    prevent_initial_call=True
)
def update_cell_specific_output(cell_selection, selected_model, stored_table_data, stored_uploaded_data):
    if selected_model != 'Anomaly Detection Stack Voltage' or not cell_selection or not stored_table_data:
        return dash.no_update

    try:
        output = [html.H5(f"Anomalies for {cell_selection}", className="card-title")]
        df_anomalies = pd.DataFrame(stored_table_data)
        df_all = pd.DataFrame(stored_uploaded_data) if stored_uploaded_data else pd.DataFrame()
        if not df_all.empty:
            df_all['Date_Time'] = pd.to_datetime(df_all['Date_Time'])
            df_anomalies['Date_Time'] = pd.to_datetime(df_anomalies['Date_Time'])
            cell_anomalies = df_anomalies[df_anomalies['Min_Cell_Index'] == cell_selection]
            if cell_selection in df_all.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_all['Date_Time'],
                    y=df_all[cell_selection],
                    mode='lines+markers',
                    name=f'{cell_selection} Voltage',
                    line=dict(color='blue')
                ))
                if not cell_anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=cell_anomalies['Date_Time'],
                        y=cell_anomalies['Min_Cell_Voltage'],
                        mode='markers',
                        name=f'{cell_selection} Anomalies',
                        marker=dict(color='red', size=10)
                    ))
                fig.update_layout(
                    title=f"{cell_selection} Voltage Time-Series with Anomalies",
                    xaxis_title="Date Time",
                    yaxis_title="Voltage (V)",
                    showlegend=True
                )
                plot_path = os.path.join(PLOT_DIR, "anomaly_cell_specific_plot.png")
                fig.write_image(plot_path)
                output.append(dcc.Graph(figure=fig))
            if not cell_anomalies.empty:
                output.append(dash_table.DataTable(
                    data=cell_anomalies.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in cell_anomalies.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    page_size=10,
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                    ]
                ))
            else:
                output.append(html.P(f"No anomalies detected for {cell_selection}.", className="text-warning"))
        else:
            output.append(html.P("No uploaded data available. Please upload a file first.", className="text-danger"))

        return html.Div(output, className="mt-3")

    except Exception as e:
        logger.error(f"Cell-specific Error: {str(e)}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)