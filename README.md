# ✈️ Airline Customer Satisfaction Predictor

A small machine learning project that predicts whether an airline passenger will be **satisfied** or **not satisfied** with their flight.

I trained a Logistic Regression model, built an API for it with FastAPI, and made a dashboard to try it out and explore the data.

<!-- Add a screenshot: save it as dashboard.png in the project folder and uncomment the next line -->
<!-- ![Dashboard](dashboard.png) -->

## What it does

- Enter a passenger's details and service ratings, and the model predicts if they will be satisfied (with a probability)
- Filter the dataset by flight distance and class
- Shows charts: flight distance vs arrival delay, satisfaction by class, and satisfaction by gender

## Built with

- Python, pandas, scikit-learn
- FastAPI
- React and Chart.js (for the dashboard)

## Project files

```
├── app/
│   ├── airline_api.py              # the FastAPI server
│   └── template/
│       └── airline_dashboard.html  # the dashboard
├── data/
│   └── Airline-Dataset.csv         # the dataset
├── models/                         # saved model files (created by airline_model.py)
├── airline_model.py                # cleans the data and trains the model
├── requirements.txt
└── README.md
```

## How to run it

1. Clone the project and go into the folder

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

3. Train the model (this creates the files in the `models` folder)

```bash
python airline_model.py
```

4. Start the server

```bash
uvicorn app.airline_api:app --reload
```

5. Open http://127.0.0.1:8000 in your browser.

If the top right of the dashboard says **API connected**, everything is working.

## The model

- **Algorithm:** Logistic Regression
- **Data cleaning:** filled missing arrival delays with the mean, removed rows with a rating of 0, and converted text columns to numbers
- **Split:** 80% training and 20% testing
- **Accuracy:** XX%  <!-- put your real accuracy here -->

## Dataset

I used the Airline Passenger Satisfaction dataset from Kaggle. Each row is a customer survey with their info, flight details, service ratings from 1 to 5, and whether they were satisfied.

## What I learned

- Cleaning and preparing a real dataset with pandas
- Training and saving a model with scikit-learn
- Building an API that serves a model with FastAPI
- Connecting a frontend to an API

## Ideas for the future

- Try other models like Random Forest
- Show which features matter the most
- Put the project online

## Author

<your-name> · [GitHub](https://github.com/<your-username>)
