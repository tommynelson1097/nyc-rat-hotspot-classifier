from pyexpat import features
import streamlit as st
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- Custom Logo ---

st.set_page_config(
    page_title="NYC Rat Hot Spot Analysis",
    layout="centered"
)

# st.image("rat_nyc.jpg", width=120)
st.image("streamlit_files/rat_nyc.jpg", width=120)
st.markdown("""
# NYC Rat Hot Spot Analysis
Welcome to the interactive dashboard for exploring and analyzing rat sighting hot spots in New York City!

**Hot Spot Definition:**
Hot spots are defined as the top 10% of NYC ZIP codes by rat sighting frequency, based on 311 service request data. These are the areas with the highest reported rat activity.
""")

# Use st.cache for compatibility with older Streamlit versions
@st.cache
def load_data():
    # path = '../Rat_Sightings.csv'
    path = 'Rat_Sightings.csv'
    df = pd.read_csv(path)
    return df

df = load_data().copy()

# Merge hot spot classification back onto the dataframe 

zip_counts = df.groupby('Incident Zip')['Unique Key'].nunique().sort_values(ascending=False)
threshold_90th = zip_counts.quantile(0.9)

# Create the classification dataframe
zip_classification = zip_counts.reset_index()
zip_classification.columns = ['Incident_Zip', 'Rat_Sighting_Count']
zip_classification['Is_Hot_Spot'] = zip_classification['Rat_Sighting_Count'] >= threshold_90th
zip_classification['Hot_Spot_Category'] = zip_classification['Is_Hot_Spot'].map({True: 'Hot Spot', False: 'Normal'})

# Merge with the selected features dataframe
df_with_hotspots = df.merge(
    zip_classification[['Incident_Zip', 'Rat_Sighting_Count', 'Is_Hot_Spot', 'Hot_Spot_Category']], 
    left_on='Incident Zip', 
    right_on='Incident_Zip', 
    how='left'
)

# Drop the duplicate zip column
df_with_hotspots = df_with_hotspots.drop('Incident_Zip', axis=1)

df_with_hotspots['Created Date'] = pd.to_datetime(df_with_hotspots['Created Date'], errors='coerce')


# Feature Engineering and Data Preparation
# Extract additional date features from Create Date
df_with_hotspots['Created Date'] = pd.to_datetime(df_with_hotspots['Created Date'], errors='coerce')
df_with_hotspots['Month'] = df_with_hotspots['Created Date'].dt.month
df_with_hotspots['Year'] = df_with_hotspots['Created Date'].dt.year
df_with_hotspots['DayOfWeek'] = df_with_hotspots['Created Date'].dt.dayofweek
df_with_hotspots['Hour'] = df_with_hotspots['Created Date'].dt.hour

# Create Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Unknown'

df_with_hotspots['Season'] = df_with_hotspots['Month'].apply(get_season)


# Calculate Response Time (if Closed Date exists)
if 'Closed Date' in df_with_hotspots.columns:
    df_with_hotspots['Closed Date'] = pd.to_datetime(df_with_hotspots['Closed Date'], errors='coerce')
    df_with_hotspots['Created Date'] = pd.to_datetime(df_with_hotspots['Created Date'], errors='coerce')

    # Calculate response time in days
    df_with_hotspots['Response_Time'] = (df_with_hotspots['Closed Date'] - df_with_hotspots['Created Date']).dt.days

    # Replace negative or missing values with median of valid response times
    valid_response_times = df_with_hotspots['Response_Time'][df_with_hotspots['Response_Time'] >= 0]
    median_response_time = valid_response_times.median()

    df_with_hotspots['Response_Time'] = df_with_hotspots['Response_Time'].apply(
        lambda x: median_response_time if pd.isna(x) or x < 0 else x
    )
else:
    # Create a synthetic response time based on other factors
    np.random.seed(42)
    df_with_hotspots['Response_Time'] = np.random.exponential(scale=7, size=len(df_with_hotspots))






tab1, tab2, tab3 = st.tabs(["📊 Rat Sighting Distribution Analysis", "🏙️ NYC Map", "🧮 Interactive Rat Sighting Hot-Spot Predictor"])


# Tab 1: Distribution Analysis Page
with tab1:
    # Assuming df_with_hotspots is already loaded and preprocessed

    # Set plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Rat Sightings - Comprehensive Data Analysis', fontsize=16, fontweight='bold')

    # Color palette (NYC MTA subway line colors)
    colors = ['#0039A6', '#EE352E', '#00933C', '#FF6319', '#FCCC0A']


    # 1. Borough Distribution
    borough_counts = df_with_hotspots['Borough'].value_counts()
    axes[0, 0].bar(borough_counts.index, borough_counts.values, color=colors)
    axes[0, 0].set_title('Rat Sightings by Borough', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Sightings')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Location Type Distribution (top 10)
    location_counts = df_with_hotspots['Location Type'].value_counts().head(10)
    axes[0, 1].barh(location_counts.index, location_counts.values, color=colors[1])
    axes[0, 1].set_title('Top 10 Location Types', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Sightings')

    # 3. Seasonal Pattern
    season_counts = df_with_hotspots['Season'].value_counts()
    axes[0, 2].pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0, 2].set_title('Seasonal Distribution', fontweight='bold')

    # 4. Address Type Distribution
    address_counts = df_with_hotspots['Address Type'].value_counts()
    axes[1, 0].bar(address_counts.index, address_counts.values, color=colors[2])
    axes[1, 0].set_title('Sightings by Address Type', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Sightings')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Response Time Distribution
    axes[1, 1].hist(df_with_hotspots['Response_Time'], bins=30, color=colors[3], alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Response Time Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Response Time (days)')
    axes[1, 1].set_ylabel('Frequency')

    # 6. Hot Spot vs Normal Distribution by Borough
    hot_spot_borough = df_with_hotspots.groupby(['Borough', 'Hot_Spot_Category']).size().unstack(fill_value=0)
    hot_spot_borough.plot(kind='bar', ax=axes[1, 2], color=[colors[0], colors[4]], width=0.8)
    axes[1, 2].set_title('Hot Spots vs Normal by Borough', fontweight='bold')
    axes[1, 2].set_xlabel('Borough')
    axes[1, 2].set_ylabel('Number of Sightings')
    axes[1, 2].legend(title='Category')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)


# Tab 2: NYC Interactive Map Page
with tab2:
    st.subheader("NYC Interactive Heatmap")
    st.write("This heatmap shows the density of rat sightings across NYC. Brighter areas indicate more reports. For performance, a random sample of 5,000 sightings is shown.")
    # Filter for valid lat/lon
    df_map = df_with_hotspots.dropna(subset=["Latitude", "Longitude"])
    # Limit to 2000 points for performance
    max_points = 5000
    if len(df_map) > max_points:
        df_map = df_map.sample(n=max_points, random_state=42)
    # Create a map using Folium
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    # Prepare data for HeatMap
    heat_data = df_map[["Latitude", "Longitude"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=12, min_opacity=0.2, max_zoom=1).add_to(nyc_map)
    # Display the map
    st_folium(nyc_map, width=700, height=500)



# Tab 3: Interactive Rat Sighting Hot Spot Predictor
with tab3:
    st.subheader("Interactive Rat Sighting Hot Spot Predictor")
    st.write("Enter the features below to predict if a sighting is in a hot spot zip code. This predictor uses the top-performing single hidden layer neural network model for classification.")


    # Load the best model and feature columns
    @st.cache(allow_output_mutation=True)
    def load_best_model_and_features_and_scaler():
        model = joblib.load('streamlit_files/best_model.pkl')
        features = joblib.load('streamlit_files/model_features.pkl')
        scaler = joblib.load('streamlit_files/scaler.pkl')
        return model, features, scaler
    best_model, model_features, scaler = load_best_model_and_features_and_scaler()

    # Feature columns
    feature_columns = ['Borough', 'Location Type', 'Season','Response_Time','Address Type','Month', 'Year', 'DayOfWeek', 'Hour']

    # Get unique values for dropdowns from the data
    boroughs = sorted(df_with_hotspots['Borough'].dropna().unique())
    location_types = sorted(df_with_hotspots['Location Type'].dropna().unique())
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    address_types = sorted(df_with_hotspots['Address Type'].dropna().unique())

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            borough = st.selectbox("Borough", boroughs)
            location_type = st.selectbox("Location Type", location_types)
            season = st.selectbox("Season", seasons)
            response_time = st.number_input("Response Time (days)", min_value=0.0, value=7.0, format="%.2f")
        with col2:
            address_type = st.selectbox("Address Type", address_types)
            month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            month_name = st.selectbox("Month", month_names, index=5)
            month = month_names.index(month_name) + 1
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_name = st.selectbox("Day of Week", weekdays, index=0)
            dayofweek = weekdays.index(weekday_name)
            hour_labels = [f"{h:02d}:00" for h in range(24)]
            hour_label = st.selectbox("Hour", hour_labels, index=12)
            hour = hour_labels.index(hour_label)
        submitted = st.form_submit_button("Predict Hot Spot")


    if submitted:
        # Build input DataFrame (without latitude and longitude)
        input_dict = {
            'Borough': [borough],
            'Location Type': [location_type],
            'Season': [season],
            'Response_Time': [response_time],
            'Address Type': [address_type],
            'Month': [month],
            'Year': [year],
            'DayOfWeek': [dayofweek],
            'Hour': [hour]
        }
        input_df = pd.DataFrame(input_dict)

        # One-hot encode categorical variables
        categorical_cols = ['Borough', 'Location Type', 'Season', 'Address Type']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

        # Align columns with model_features
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        # Ensure correct column order
        input_encoded = input_encoded[model_features]



        input_scaled = scaler.transform(input_encoded)

        pred = best_model.predict(input_scaled)[0]
        pred_proba = best_model.predict_proba(input_scaled)[0][1]

        # Predict
        if pred:
            st.markdown(
            f"<div style='background-color:#FF6319;padding:1.2em 1em 1.2em 1em;border-radius:10px;text-align:center;'>"
            f"<span style='font-size:2em;'>🐀</span><br>"
            f"<span style='font-size:1.5em;font-weight:bold;color:white;'>HOT SPOT!</span><br>"
            f"<span style='font-size:1.1em;color:white;'>Probability of Hot Spot: <b>{pred_proba:.2%}</b></span>"
            f"</div>", unsafe_allow_html=True)
        else:
            st.markdown(
            f"<div style='background-color:#0039A6;padding:1.2em 1em 1.2em 1em;border-radius:10px;text-align:center;'>"
            f"<span style='font-size:2em;'>🏙️</span><br>"
            f"<span style='font-size:1.5em;font-weight:bold;color:white;'>Non-Hot Spot</span><br>"
            f"<span style='font-size:1.1em;color:white;'>Probability of Hot Spot: <b>{pred_proba:.2%}</b></span>"
            f"</div>", unsafe_allow_html=True)
            

        # Debug output
        st.subheader('Debug: Model Input Inspection')
        st.write('**Input DataFrame after encoding and alignment:**')
        st.dataframe(input_encoded)
        st.write('**Input values passed to model:**', input_encoded.values)

        # Scale input
        st.write('**Input after scaling:**')
        st.dataframe(pd.DataFrame(input_scaled, columns=input_encoded.columns))
        
        # st.markdown(f"### Prediction: {'🐀 Hot Spot' if pred else 'Non-Hot Spot'}")
        # st.markdown(f"**Probability of Hot Spot:** {pred_proba:.2%}")