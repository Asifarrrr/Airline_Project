# ✈️ Airline Customer Satisfaction Predictor

An end-to-end machine learning project that predicts whether an airline passenger will be satisfied with their flight. A logistic regression model is served through a **FastAPI** backend and explored through an interactive **React** dashboard with a modern dark purple interface.

<!-- Add a screenshot: save it as docs/dashboard.png and uncomment the line below -->
<!-- ![Dashboard screenshot](docs/dashboard.png) -->

## ✨ Features

- **🎯 Satisfaction prediction:** Enter a customer profile, flight details, and 13 service ratings to receive a prediction with its probability score.
- **📊 Interactive analytics:** Filter customers dynamically by flight distance and class, watching KPIs and charts update instantly.
- **📈 Rich visualizations:** Explore flight distance vs. arrival delay, satisfaction shares by travel class, and gender-based breakdowns.
- **⚡ REST API:** A fully documented FastAPI backend featuring automatic request validation and interactive Swagger UI at `/docs`.
- **🌐 Single-file frontend:** Built using React 18 and Chart.js loaded directly from a CDN—requiring zero Node.js build steps.

## 🛠️ Tech Stack

| Layer | Tools & Frameworks |
|---|---|
| **Model** | Python 🐍, Pandas, Scikit-Learn (Logistic Regression, StandardScaler) |
| **Backend** | FastAPI 🚀, Uvicorn, Pydantic |
| **Frontend** | React 18 ⚛️, Chart.js 📊, HTML5 / CSS3 |

## 📂 Project Structure

airline-satisfaction/
├── app/
│   ├── airline_api.py           # FastAPI app (/predict, /data, dashboard view)
│   └── template/
│       └── airline_dashboard.html
├── data/
│   └── Airline-Dataset.csv      # Raw customer survey dataset
├── models/                      # Generated artifacts from training script
│   ├── airline_model.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   └── data_frame.pkl
├── airline_model.py             # Data preprocessing and model training pipeline
├── requirements.txt             # Python dependencies
└── README.md
💡 Note: Adjust the file paths above if your local repository structure differs.🚀 Getting Started & Setup CommandsTo get your application up and running locally from start to finish, run the following complete sequence of terminal commands:Bash# 1. Clone the repository and enter the directory
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux

# 3. Install required Python dependencies
pip install -r requirements.txt

# 4. Train the model (cleans data, trains, and serializes model artifacts into models/)
python airline_model.py

# 5. Start the FastAPI backend server
uvicorn app.airline_api:app --reload
Dashboard Access: Open http://127.0.0.1:8000 in your browser.API Docs: Explore interactive endpoints at http://127.0.0.1:8000/docs.Tip: The status pill in the top-right corner of the dashboard will display "API connected" when communication with the backend is successful. If it shows "API offline", verify that your Uvicorn server is actively running on port 8000.🔌 API ReferenceMethodEndpointDescriptionGET/Serves the interactive React dashboard UIGET/dataReturns the cleaned dataset as JSON to power dashboard analyticsPOST/predictEvaluates customer parameters and returns satisfaction status & probability📝 Example Request (POST /predict)JSON{
  "Gender": 1,
  "Customer_Type": 1,
  "Age": 35,
  "Type_of_Travel": 1,
  "Flight_Distance": 1000,
  "Inflight_wifi_service": 3,
  "Departure_Arrival_time_convenient": 3,
  "Gate_location": 3,
  "Food_and_drink": 3,
  "Online_boarding": 3,
  "Seat_comfort": 3,
  "Inflight_entertainment": 3,
  "On_board_service": 3,
  "Leg_room_service": 3,
  "Baggage_handling": 3,
  "Checkin_service": 3,
  "Inflight_service": 3,
  "Cleanliness": 3,
  "Departure_Delay_in_Minutes": 0,
  "Arrival_Delay_in_Minutes": 0,
  "Class_business": 1,
  "Class_eco": 0,
  "Class_eco_plus": 0
}
📨 Example ResponseJSON{ 
  "Prediction": 1, 
  "Probability": 0.87 
}
(ℹ️ Prediction evaluates to 1 for Satisfied and 0 for Neutral or Dissatisfied. Probability reflects the model's confidence score).🧠 Model SpecificationsAlgorithm: Logistic Regression 📐Preprocessing Pipeline: Missing arrival delays imputed with column mean, invalid rating zero-values dropped, categorical variables encoded, class categories one-hot encoded, and numerical features scaled using StandardScaler.Train/Test Split: 80% training / 20% testing (random_state=42) ✂️Performance: Achieved an accuracy of XX% on the test set. (📝 Replace "XX%" with your actual model evaluation score, e.g., 87.4%)📊 DatasetSourced from Kaggle's Airline Passenger Satisfaction collection. Each row maps out individual customer survey feedback detailing passenger demographics, travel specs, 1-to-5 service grading metrics, and final satisfaction labels.🗺️ Roadmap & Future Improvements[ ] Benchmark performance against alternative classification models (Random Forest 🌲, Gradient Boosting / XGBoost 🚀)[ ] Integrate feature importance metrics directly into the web dashboard UI[ ] Containerize the full service stack using Docker 🐳 and deploy to a cloud provider📝 LicenseDistributed under the terms of the MIT License. See LICENSE for more information.
