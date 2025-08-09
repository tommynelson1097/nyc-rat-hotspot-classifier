
from pyexpat import features
import streamlit as st
# Imports 
# Built-in libraries
import warnings
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster


# Scikit-learn modules
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, StratifiedKFold

st.set_page_config(
	page_title="NYC Rat Hot Spot Analysis",
	page_icon="🐀",
	layout="centered"
)

st.markdown("""
# 🐀 NYC Rat Hot Spot Analysis
Welcome to the interactive dashboard for exploring and analyzing rat sighting hot spots in New York City!
""")

# Use st.cache for compatibility with older Streamlit versions
@st.cache
def load_data():
	path = 'Rat_Sightings.csv'
	df = pd.read_csv(path)
	return df

df = load_data().copy()

# Create zip code hot spot classification based on top 10% of rat sightings
zip_counts = df.groupby('Incident Zip')['Unique Key'].nunique().sort_values(ascending=False)

# Calculate the 90th percentile threshold (top 10%)
threshold_90th = zip_counts.quantile(0.9)

# Number of zip codes in top 10%
total_zip_codes = len(zip_counts)
top_10_percent_count = int(total_zip_codes * 0.1)

# Create hot spot classification
zip_counts_df = zip_counts.reset_index()
zip_counts_df.columns = ['Incident_Zip', 'Rat_Sighting_Count']

# Add hot spot classification column
zip_counts_df['Is_Hot_Spot'] = zip_counts_df['Rat_Sighting_Count'] >= threshold_90th
zip_counts_df['Hot_Spot_Category'] = zip_counts_df['Is_Hot_Spot'].map({True: 'Hot Spot', False: 'Normal'})

# Display hot spots
hot_spots = zip_counts_df[zip_counts_df['Is_Hot_Spot'] == True]


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






tab1, tab2, tab3 = st.tabs(["📊 Rat Sighting Distribution Analysis", "📈 NYC Interactive Map", "🧮 Interactive Rat Sighting Predictor"])


# Tab 1: Histogram Page
with tab1:


    # Assuming df_with_hotspots is already loaded and preprocessed

    st.subheader("Rat Sighting Distribution")
    st.write("Explore the distribution of rat sightings across NYC.")

    # Set plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Rat Sightings - Comprehensive Data Analysis', fontsize=16, fontweight='bold')

    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']

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


# # Tab 2: NYC Interactive Map Page
# with tab2:
#     st.subheader("NYC Interactive Map")
#     st.write("Visualize rat sightings across NYC on an interactive map.")

#     # Create a map using Folium
#     nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

#     # Use MarkerCluster for performance
#     marker_cluster = MarkerCluster().add_to(nyc_map)
#     for _, row in df_with_hotspots.iterrows():
#         if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
#             folium.Marker(
#                 location=[row['Latitude'], row['Longitude']],
#                 popup=f"Zip: {row['Incident Zip']}<br>Count: {row['Rat_Sighting_Count']}",
#                 icon=folium.Icon(color='red' if row['Is_Hot_Spot'] else 'blue')
#             ).add_to(marker_cluster)

#     # Display the map
#     st.write(nyc_map._repr_html_(), unsafe_allow_html=True)



with tab3:
    st.subheader("Interactive Rat Sighting Predictor")
    st.write("Predict the likelihood of a rat sighting based on various features.")

    # Model Building - Predict Hot Spot Classification
    print("Missing values in Is_Hot_Spot:", df_with_hotspots['Is_Hot_Spot'].isnull().sum())

    # Create a copy of the dataframe for modeling
    df_model = df_with_hotspots.copy()

    # Remove rows with missing target values (NaN in Is_Hot_Spot)
    df_model = df_model.dropna(subset=['Is_Hot_Spot'])

    # Now convert to integer
    y = df_model['Is_Hot_Spot'].astype(int)

    # Select features for modeling
    feature_columns = ['Borough', 'Location Type', 'Latitude', 'Longitude','Season','Response_Time','Address Type','Month', 'Year', 'DayOfWeek', 'Hour']
    X = df_model[feature_columns].copy()

    # Handle missing values in features
    print(f"\nMissing values in features:")
    print(X.isnull().sum())

    # Drop rows with missing coordinates and update both X and y consistently
    X_clean = X.dropna()
    y_clean = y[X_clean.index]

    X_clean = X_clean.copy()
    y_clean = y_clean.copy()

    print(f"Clean dataset shape: {X_clean.shape}")
    print(f"Features: {list(X_clean.columns)}")

    # Identify categorical and numerical columns
    categorical_cols = ['Borough', 'Location Type', 'Season', 'Address Type']
    numerical_cols = ['Latitude', 'Longitude', 'Response_Time']

    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=False)

    print(f"\nAfter encoding shape: {X_encoded.shape}")
    print(f"New feature count: {len(X_encoded.columns)}")

    # Display some encoded column names
    encoded_cols = [col for col in X_encoded.columns if any(cat in col for cat in categorical_cols)]
    print(f"Sample encoded columns: {encoded_cols[:10]}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    