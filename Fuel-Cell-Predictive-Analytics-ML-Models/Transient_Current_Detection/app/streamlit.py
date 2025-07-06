import streamlit as st
import pandas as pd
import requests
import io
import base64
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI endpoint
API_URL = "http://localhost:8005/predict"

# Streamlit app
st.title("Transient Current Fault Detection")
st.markdown("""
Upload a CSV or Excel file to detect faults in fuel cell data using SelectKBest and Random Forest.
**Note**: Ensure the file has 21 columns (e.g., I, AIP1, CAIF, CIT1, fault). CSV must be UTF-8 encoded with comma delimiters.
""")

# Excel to CSV converter
st.header("Convert Excel to CSV")
excel_file = st.file_uploader("Upload Excel File", type=['xlsx'], key='excel')
if excel_file:
    try:
        df = pd.read_excel(excel_file)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        st.success(f"Converted {excel_file.name} to CSV")
        st.download_button(
            label="Download Converted CSV",
            data=csv_buffer.getvalue(),
            file_name="converted_combined_all_files.csv",
            mime="text/csv"
        )
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {str(e)}")
        st.error(f"Error: {str(e)}")

# Main file uploader
st.header("Fault Detection")
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=['csv', 'xlsx'], key='data')
if uploaded_file:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Sending uploaded file to FastAPI")
        
        # Prepare file for API request
        files = {'file': (uploaded_file.name, uploaded_file, 'text/csv' if uploaded_file.name.endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        
        # Send request to FastAPI
        response = requests.post(API_URL, files=files, timeout=600)
        if response.status_code != 200:
            logger.error(f"API error: {response.json()['detail']}")
            st.error(f"API Error: {response.json()['detail']}")
        else:
            result = response.json()
            
            # Parse API response
            fault_indices = result['fault_indices']
            plot_base64 = result['anomaly_plot'].replace("data:image/png;base64,", "")
            summary = result['summary']
            results_csv = result['results_csv']
            
            # Display plot
            st.image(base64.b64decode(plot_base64), caption="Transient Current Fault Detection", use_column_width=True)
            
            # Display metrics
            st.subheader("Metrics")
            st.write(f"**Detected Faults**: {len(fault_indices)}")
            st.write(f"**Total Rows**: {summary['total_rows']}")
            st.write(f"**Fault Percentage**: {summary['fault_percentage']:.2f}%")
            st.write(f"**Features Used**: {', '.join(summary['features_used'])}")
            
            # Display prediction table
            st.subheader("Predictions")
            results_df = pd.read_csv(results_csv)
            st.dataframe(results_df, height=300)
            
            # Download results
            with open(results_csv, "rb") as f:
                st.download_button(
                    label="Download Results CSV",
                    data=f,
                    file_name=f"realtime_anomalies_selectkbest_{timestamp}.csv",
                    mime="text/csv"
                )
            
            # Per-cell reports
            st.subheader("Per-Cell Reports (Placeholder, 600 cells)")
            cell_files = [f for f in os.listdir("outputs") if f.startswith("realtime_anomalies_cell_Cell") and f.endswith(f"{timestamp}.csv")]
            for cell_file in cell_files[:10]:
                st.write(cell_file)
            if len(cell_files) > 10:
                st.write(f"... and {len(cell_files) - 10} more")
            if cell_files:
                with open(f"outputs/realtime_anomalies_cell_Cell1_{timestamp}.csv", "rb") as f:
                    st.download_button(
                        label="Download Sample Cell Report (Cell1)",
                        data=f,
                        file_name=f"realtime_anomalies_cell_Cell1_{timestamp}.csv",
                        mime="text/csv"
                    )
            
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        st.error(f"Error: {str(e)}")