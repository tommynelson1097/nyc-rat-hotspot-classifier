NYC Rat Sightings Hot Spot Classifier
A machine learning project that predicts rat infestation hot spots across NYC ZIP codes using supervised classification and 311 service data.

Project Overview
Analyzes NYC 311 rat sighting complaints to identify ZIP codes with high rat activity. Hot spots are defined as the top 10% of ZIP codes by sighting frequency. The goal is to help city officials and pest control services prioritize interventions.

Objectives
Predict ZIP codes likely to become rat hot spots

Optimize pest control resource allocation

Enable early intervention

Compare multiple ML models

Quick Start
Prerequisites
bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Steps
Clone the repo

Add Rat_Sightings.csv to the root directory

Launch Jupyter Notebook

Run all cells in v2-rat-sightings-classifier.ipynb

Project Structure
├── v2-rat-sightings-classifier.ipynb
├── Rat_Sightings.csv
├── zip_code_hot_spot_classification.csv
├── streamlit_app.py
└── README.md
Machine Learning Pipeline
Feature Engineering
Temporal: season, month, day, hour

Geographic: borough, coordinates, address type

Location: residential, commercial, parks

Response: time-to-resolution

Encoding: one-hot for categorical variables

Hot Spot Definition
Top 10% ZIP codes by sighting frequency

Binary classification: hot spot vs. normal

Model Architectures
Logistic Regression (baseline)

Neural Networks: shallow, deep, wide variants

Evaluation
5-fold stratified cross-validation

Metrics: accuracy, ROC AUC, precision, recall

Train/test split: 80/20

Feature scaling with StandardScaler

Model Performance
Accuracy: 85–90%

ROC AUC: 0.85–0.90

Consistent cross-validation results

Over 50 features after encoding

Comparison Criteria
Performance vs. complexity

Interpretability vs. predictive power

Key Insights
Geographic Patterns
Manhattan: fastest response

Staten Island: slowest

Borough effects are predictive

Location Type
Restaurants: fastest service

Residential: standard

Parks: lowest priority

Temporal Patterns
Seasonal trends matter

Response time is a strong predictor

Visualizations
Includes charts for:

Borough distribution

Location type frequency

Seasonal patterns

Address type breakdown

Response time distribution

Hot spot mapping

Business Applications
NYC Government
Data-driven pest control

Budget optimization

Service performance tracking

Policy development

Pest Control Services
Route planning

Proactive scheduling

Location-specific response strategies

Public Health
Risk identification

Targeted prevention

Health impact monitoring

Community outreach

Technical Implementation
Feature Engineering
python
df['Season'] = df['Month'].apply(get_season)
df['Response_Time'] = (df['Closed Date'] - df['Created Date']).dt.days
Model Training
python
lr_model = LogisticRegression(random_state=42, max_iter=500)
mlp = MLPClassifier(hidden_layer_sizes=(50, 30, 20, 10), early_stopping=True, random_state=42, max_iter=500)
Evaluation
python
comparison_df = pd.DataFrame({
    'Architecture': model_names,
    'Test Accuracy': accuracies,
    'Test AUC': auc_scores
}).sort_values('Test Accuracy', ascending=False)
Future Enhancements
Short Term
Streamlit dashboard

Real-time API

Model deployment

Performance monitoring

Medium Term
Geospatial clustering

Time series forecasting

Mobile app

System integration

Long Term
Deep learning (CNNs)

Expansion to other cities

IoT sensor integration

Causal analysis

Contributing
Fork the repo

Create a feature branch

Commit changes

Push and open a pull request

Guidelines
Follow PEP 8

Use docstrings

Add unit tests

Update documentation

Data Sources
NYC 311 Rat Sightings dataset

Format: CSV with coordinates and timestamps

Source: NYC Open Data Portal

License
MIT License — see LICENSE file

Contact
GitHub Issues for bugs and features

GitHub Discussions for general questions

Email: [your contact email]

Acknowledgments
NYC Open Data

Scikit-learn

Data science community

NYC Department of Health

Results Summary
Best accuracy: ~87–90%

Best AUC: ~0.86–0.89

Recommended models: Logistic Regression (interpretability), MLP (performance)

Key features: response time, borough, location type

Impact Potential
Proactive rat control

Smarter resource allocation

Improved public health

Evidence-based policy support
