import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# __________________________________ Data Cleaning (Pandas) ____________________________________
df = None
try:
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\data\Airline-Dataset.csv")
except FileNotFoundError:
    pass

df = df.drop(columns = ['Unnamed: 0','id'])

text = ['Gender','Customer Type','Type of Travel','Class','satisfaction']
for col in text:
    df[col] = df[col].str.strip().str.lower()

useless = ['Inflight wifi service','Departure/Arrival time convenient'
    ,'Ease of Online booking','Gate location','Food and drink','Online boarding'
    ,'Online boarding','Seat comfort','Inflight entertainment','On-board service'
    ,'Leg room service','Baggage handling','Checkin service','Inflight service'
    ,'Cleanliness']
for col in useless:
    df.loc[df[col] == 0,col] = None
    df = df.dropna(subset = col)
    df[col] = df[col].astype(int)

df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
    df['Arrival Delay in Minutes'].mean()
)

df['Gender'] = df['Gender'].replace({
    'male': 1,
    'female': 0
})
df['Gender'] = df['Gender'].astype(int)

df['Customer Type'] = df['Customer Type'].replace({
    'loyal customer': 1,
    'disloyal customer': 0
})
df['Customer Type'] = df['Customer Type'].astype(int)

df['Type of Travel'] = df['Type of Travel'].replace({
    'business travel': 1,
    'personal travel': 0
})
df['Type of Travel'] = df['Type of Travel'].astype(int)

df['satisfaction'] = df['satisfaction'].replace({
    'satisfied': 1,
    'neutral or dissatisfied': 0
})
df['satisfaction']  = df['satisfaction'].astype(int)

df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].astype(int)

df = pd.get_dummies(df,columns = ['Class'],dtype = int)
df = df.rename(
    columns={
        'Customer Type': 'Customer_Type',
        'Type of Travel': 'Type_of_Travel',
        'Flight Distance': 'Flight_Distance',
        'Inflight wifi service': 'Inflight_wifi_service',
        'Departure/Arrival time convenient': (
            'Departure_Arrival_time_convenient'
        ),
        'Gate location': 'Gate_location',
        'Food and drink': 'Food_and_drink',
        'Online_boarding': 'Online_boarding',
        'Seat comfort': 'Seat_comfort',
        'Inflight entertainment': 'Inflight_entertainment',
        'On-board service': 'On_board_service',
        'Leg room service': 'Leg_room_service',
        'Baggage handling': 'Baggage_handling',
        'Checkin service': 'Checkin_service',
        'Inflight service': 'Inflight_service',
        'Departure Delay in Minutes': 'Departure_Delay_in_Minutes',
        'Arrival Delay in Minutes': 'Arrival_Delay_in_Minutes',
        'Class_eco plus': 'Class_eco_plus',
    }
)
# _____________________________________ LogisticRegression (Ai) _____________________________________
x = df.drop(columns = ['satisfaction'])
y = df['satisfaction']

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)

joblib.dump(df,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\data_frame.pkl")
joblib.dump(model,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\data\airline_model.pkl")
joblib.dump(scaler,r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\scaler.pkl")
joblib.dump(list(x.columns),r"C:\Users\pc\PycharmProjects\WelcomeScreen\Mini Projects\Airline_Project\models\model_columns.pkl")