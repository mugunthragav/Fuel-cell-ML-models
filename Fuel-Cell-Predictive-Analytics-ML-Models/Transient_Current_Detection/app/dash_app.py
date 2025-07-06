from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import requests
import io
import base64
import logging
from datetime import datetime
import os
import plotly.graph_objs as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI endpoint
API_URL = "http://localhost:8005/predict"

# Initialize Dash app
app = Dash(__name__)
app.title = "Transient Current Fault Detection"

app.layout = html.Div([
    html.H1("Transient Current Fault Detection"),
    html.P("Upload a CSV or Excel file to detect faults in fuel cell data using KFDA, Wavelet Packet, and SVD."),
    html.P("Note: Ensure the file has 21 columns (e.g., I, AIP1, CAIF, CIT1, fault). CSV must be UTF-8 encoded with comma delimiters."),

    # Excel to CSV converter
    html.H3("Convert Excel to CSV"),
    dcc.Upload(
        id='upload-excel',
        children=html.Button('Upload Excel File'),
        accept='.xlsx',
        multiple=False
    ),
    html.Div(id='excel-convert-output'),
    dcc.Download(id='download-converted-csv'),

    # Main file uploader
    html.H3("Fault Detection"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV or Excel File'),
        accept='.csv,.xlsx',
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    dcc.Graph(id='anomaly-plot'),
    html.Div(id='metrics'),
    dash_table.DataTable(id='prediction-table', page_size=10),
    html.Div(id='per-cell-reports'),
    dcc.Download(id='download-results-csv')
])

@app.callback(
    [Output('excel-convert-output', 'children'),
     Output('download-converted-csv', 'data')],
    Input('upload-excel', 'contents'),
    State('upload-excel', 'filename')
)
def convert_excel_to_csv(contents, filename):
    if contents is None:
        return "", None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        logger.info(f"Converted {filename} to CSV")
        return (
            html.P(f"Excel file {filename} converted to CSV. Download below."),
            dcc.send_string(csv_buffer.getvalue(), "converted_combined_all_files.csv")
        )
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
        return html.P(f"Error: {str(e)}", style={'color': 'red'}), None

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('anomaly-plot', 'figure'),
     Output('metrics', 'children'),
     Output('prediction-table', 'data'),
     Output('prediction-table', 'columns'),
     Output('per-cell-reports', 'children'),
     Output('download-results-csv', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def process_upload(contents, filename):
    if contents is None:
        return "", {}, [], [], [], None
    if not filename.endswith(('.csv', '.xlsx')):
        return html.P("Please upload a .csv or .xlsx file", style={'color': 'red'}), {}, [], [], [], None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Sending uploaded file to FastAPI")
        
        # Prepare file for API request
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        mime_type = 'text/csv' if filename.endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        files = {'file': (filename, io.BytesIO(decoded), mime_type)}
        
        # Send request to FastAPI
        response = requests.post(API_URL, files=files, timeout=600)
        if response.status_code != 200:
            logger.error(f"API error: {response.json()['detail']}")
            return html.P(f"API Error: {response.json()['detail']}", style={'color': 'red'}), {}, [], [], [], None
        
        result = response.json()
        
        # Parse API response
        fault_indices = result['fault_indices']
        plot_base64 = result['anomaly_plot'].replace("data:image/png;base64,", "")
        summary = result['summary']
        
        # Convert base64 plot to Plotly figure
        plot_bytes = base64.b64decode(plot_base64)
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{plot_base64}",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="stretch"
            )
        )
        fig.update_layout(
            width=1200,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        # Read results CSV
        results_csv_path = f"outputs/realtime_anomalies_kfda_{timestamp}.csv"
        if not os.path.exists(results_csv_path):
            return html.P(f"Results CSV not found: {results_csv_path}", style={'color': 'red'}), {}, [], [], [], None
        results_df = pd.read_csv(results_csv_path)
        
        # Prepare table data
        table_data = results_df.to_dict('records')
        table_columns = [{"name": col, "id": col} for col in results_df.columns]
        
        # Metrics
        metrics = [
            html.P(f"Detected {len(fault_indices)} faults"),
            html.P(f"Total Rows: {summary['total_rows']}"),
            html.P(f"Fault Percentage: {summary['fault_percentage']:.2f}%"),
            html.P(f"Features Used: {', '.join(summary['features_used'])}")
        ]
        
        # Per-cell reports
        cell_files = [f for f in os.listdir("outputs") if f.startswith(f"realtime_anomalies_cell_Cell") and f.endswith(f"{timestamp}.csv")]
        cell_reports = [html.P(f"Per-Cell Reports (Placeholder, 600 cells):")] + \
                       [html.P(cell_file) for cell_file in cell_files[:10]] + \
                       ([html.P(f"... and {len(cell_files) - 10} more")] if len(cell_files) > 10 else [])
        
        # Download results
        with open(results_csv_path, "rb") as f:
            csv_data = f.read()
        
        return (
            html.P(f"Processed {filename} successfully"),
            fig,
            metrics,
            table_data,
            table_columns,
            cell_reports,
            dcc.send_bytes(csv_data, f"realtime_anomalies_kfda_{timestamp}.csv")
        )
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        return html.P(f"Error: {str(e)}", style={'color': 'red'}), {}, [], [], [], None

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)