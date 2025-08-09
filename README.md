A machine learning project that predicts rat infestation hot spots across NYC ZIP codes using supervised classification and 311 service data.

Project Overview
Analyzes NYC 311 rat sighting complaints to identify ZIP codes with high rat activity. Hot spots are defined as the top 10% of ZIP codes by sighting frequency. The goal is to help city officials and pest control services prioritize interventions.

Objectives
Predict ZIP codes likely to become rat hot spots



NYC Rat Sightings Hot Spot Classifier
=====================================

Overview
--------
This project uses machine learning to predict rat infestation hot spots across New York City ZIP codes, leveraging NYC 311 service request data. The model classifies ZIP codes as hot spots (top 10% by sighting frequency) to help city officials and pest control services prioritize interventions.

Features
--------
- Predicts hot spot ZIP codes based on historical 311 rat sighting complaints
- Compares multiple models (Logistic Regression, Multi-Layer Perceptron)
- Feature engineering from temporal, geographic, and service data
- Interactive Streamlit app for analysis, mapping, and prediction
- Visualizes all rat sightings on an interactive NYC map

Project Structure
-----------------
- `streamlit_app.py` : Main Streamlit web app
- `v2-rat-sightings-classifier.ipynb` : Model training, feature engineering, and EDA
- `streamlit_files/` : Contains model, scaler, and feature files for app inference
- `Rat_Sightings.csv` : Raw data

Setup
-----
1. Clone the repository and install requirements:
	```
	pip install -r requirements.txt
	```
2. Run the Streamlit app:
	```
	streamlit run streamlit_app.py
	```

Usage
-----
Use the Streamlit app to:
- Explore data analysis and model results
- Visualize rat sightings and hot spots on a map
- Predict hot spot status for new data

Model Details
-------------
- Hot spots defined as top 10% of ZIP codes by rat sighting frequency
- Features: Borough, Location Type, Season, Response Time, Address Type, Month, Year, DayOfWeek, Hour
- Models: Logistic Regression, several MLP architectures (single and multi-layer)
- Best model and scaler are saved in `streamlit_files/` for app use

Contact
-------
For questions or contributions, please contact the repository owner.



