import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/plant_disease_dataset/training_set_features.csv"
    df = pd.read_csv(url)
    return df

data = load_data()

# Preprocess data
def preprocess_data(df):
    le = LabelEncoder()
    for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
        df[col] = le.fit_transform(df[col])
    return df

data = preprocess_data(data)

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, le

model, acc, label_encoder = train_model(data)

# Sidebar input
st.sidebar.header("Input Plant Conditions")

def get_user_input():
    return pd.DataFrame([{
        'Temperature': st.sidebar.slider("Temperature (°C)", 10, 45, 25),
        'Humidity': st.sidebar.slider("Humidity (%)", 10, 100, 60),
        'Moisture': st.sidebar.slider("Soil Moisture (%)", 0, 100, 40),
        'Soil Type': st.sidebar.selectbox("Soil Type", [0, 1, 2, 3, 4, 5]),  # Encoded
        'Crop Type': st.sidebar.selectbox("Crop Type", [0, 1, 2, 3, 4, 5]),  # Encoded
        'Nitrogen': st.sidebar.slider("Nitrogen Level", 0, 140, 60),
        'Potassium': st.sidebar.slider("Potassium Level", 0, 140, 40),
        'Phosphorous': st.sidebar.slider("Phosphorous Level", 0, 140, 40),
        'Fertilizer Name': st.sidebar.selectbox("Fertilizer Used", [0, 1, 2, 3, 4])
    }])

input_df = get_user_input()

st.title("🌱 Plant Disease Prediction App")
st.markdown("Predict if a plant is **healthy** or has a **disease** based on its soil and environmental conditions.")

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    disease_label = label_encoder.inverse_transform([prediction])[0]
    st.subheader("Prediction:")
    st.success(f"The plant is likely: **{disease_label}**")

    st.subheader("Model Accuracy:")
    st.write(f"{acc:.2%}")
