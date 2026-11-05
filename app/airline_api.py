import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI,Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = 'airline API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLS = ['Gender', 'Age', 'Type_of_Travel', 'Flight_Distance',
        'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes',
        'satisfaction', 'Class_business', 'Class_eco', 'Class_eco_plus']

DATA_JSON = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\data_frame.pkl")[COLS].to_json(orient="records")

@app.get("/data")
def data():
    return Response(DATA_JSON, media_type="application/json")


model = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\data\airline_model.pkl")
scaler = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\scaler.pkl")
model_columns = joblib.load(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\model_columns.pkl")

class CustomerInput(BaseModel):
    Gender:int
    Customer_Type:int
    Age:int
    Type_of_Travel:int
    Flight_Distance:int|float
    Inflight_wifi_service:int
    Departure_Arrival_time_convenient:int
    Gate_location:int
    Food_and_drink:int
    Online_boarding:int
    Seat_comfort:int
    Inflight_entertainment:int
    On_board_service:int
    Leg_room_service:int
    Baggage_handling:int
    Checkin_service:int
    Inflight_service:int
    Cleanliness:int
    Departure_Delay_in_Minutes:int|float
    Arrival_Delay_in_Minutes:int|float
    Class_business:int
    Class_eco:int
    Class_eco_plus:int

@app.post('/predict')
def satisfaction_prediction(data: CustomerInput):
    df = pd.DataFrame([data.model_dump()])
    df = df.reindex(columns = model_columns , fill_value = 0)
    df = scaler.transform(df)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {'Prediction': prediction,'Probability': probability}