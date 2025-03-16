import gradio as gr
import pandas as pd
import pickle
from geopy.distance import geodesic

# Load Model
with open("random_forest_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Load Transport Stations Data
stations = pd.read_csv("zurich_transport_stations.csv")

def compute_transport_distance(lat, lon):
    """Finds the nearest public transport station and calculates the distance"""
    min_distance = float("inf")

    for _, station in stations.iterrows():
        station_coords = (station["lat"], station["lon"])
        distance = geodesic((lat, lon), station_coords).km

        if distance < min_distance:
            min_distance = distance

    return min_distance

# Prediction Function
def predict(rooms, area, latitude, longitude):
    # Compute transport distance
    distance_to_transport = compute_transport_distance(latitude, longitude)

    # Prepare input DataFrame
    input_data = pd.DataFrame([[rooms, area, distance_to_transport]],
                              columns=["rooms", "area", "distance_to_transport"])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return f"Predicted Rent Price: CHF {prediction:.2f}"

# Gradio App Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Area (m²)"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
    ],
    outputs="text",
    title="Apartment Rent Price Prediction",
    description="Predict apartment rent price based on features including public transport proximity."
)

demo.launch()
