FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/main.py .
COPY ../../FC-final/Anomaly_Detection_Stack_Voltage/models ./models/model1
COPY ../../FC-final/Health_State_Classification/models ./models/model2
COPY ../../FC-final/Time_Series_Anomaly_Detection/models ./models/model3
COPY ../../FC-final/Transient_Current_Detection/models ./models/model4
COPY ../../FC-final/Voltage_Prediction_Digital_Twin/models ./models/model5
COPY ../../FC-final/Anomaly_Detection_Stack_Voltage/outputs ./outputs/model1
COPY ../../FC-final/Health_State_Classification/outputs ./outputs/model2
COPY ../../FC-final/Time_Series_Anomaly_Detection/outputs ./outputs/model3
COPY ../../FC-final/Transient_Current_Detection/outputs ./outputs/model4
COPY ../../FC-final/Voltage_Prediction_Digital_Twin/outputs ./outputs/model5

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]