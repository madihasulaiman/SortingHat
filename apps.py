import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.impute import SimpleImputer

# Load and preprocess the dataset
data_set = pd.read_csv("hogwarts_dataset.csv")

# Mapping dictionaries
personality_traits_mapping = {"Openness": 0, "Conscientiousness": 1, "Agreeableness": 2, "Extraversion": 3, "Neuroticism": 4}
behavioural_traits_mapping = {"Risk-Taking: Assesses the student's willingness to take risks and try new things.": 0, "Collaboration: Measures the student's ability to work effectively with others.": 1, "Discipline: Evaluates the student's self-control and adherence to rules and schedules.": 2, "Independence: Gauges the student's ability to work autonomously and make decisions on their own.": 3}
hobbies_mapping = {"Physical (e.g., active hobbies like dancing, yoga, hiking, sports, gardening, martial arts, singing)": 0, "Cerebral (e.g., activities like sudoku, reading, and puzzles can help another part of our minds by activating our concentration)": 1, "Creative (e.g., activities like writing, painting, singing, or cooking may provide a sense of accomplishment)": 2, "Community activities (e.g., volunteering, tutoring, helping people)": 3, "Collecting (e.g., coin /stamp collectors)": 4, "Making & Tinkering (e.g., self-motivated projects like building new things, self-restoration, and repairing stuff)": 5}
academic_mapping = {"1st Class": 0, "2nd Class": 1, "3rd Class": 2}
hometown_mapping = {"South": 0, "North": 1, "West": 2, "East": 3, "Borneo": 4}
inasis_mapping = {"Laluan A": 0, "Laluan B": 1, "Laluan C": 2, "Laluan D": 3, "Laluan E": 4, "Laluan F": 5, "Other": 6}
leadership_mapping = {"No": 0, "Yes": 1}
cocuriculum_mapping = {"Arts & Culture": 0, "Leadership & Volunteerism": 1, "Sports": 2, "Uniform": 3, "Martial Arts": 4, "Academic & Clubs": 5, "Others": 6}
fav_cuisine_mapping = {"Malay": 0, "Chinese": 1, "Indian": 2, "Western": 3, "Japanese": 4, "Korean": 5, "Thai": 6, "Exotic": 7}
income_mapping = {"t20": 0, "m40": 1, "b40": 2}
faculty_mapping = {"SOC": 1, "SEFB": 2, "SBM": 3, "SOB": 4, "SOE": 5, "TISSA": 6, "SOL": 7, "SOG": 8, "STML": 9, "IBS": 10, "STHEM": 11, "SQS": 12, "SOIS": 13, "SMMTC": 14}

# Data encoding
data_set['Personality Traits'] = data_set['Personality Traits'].map(personality_traits_mapping)
data_set['Behavioural Traits'] = data_set['Behavioural Traits'].map(behavioural_traits_mapping)
data_set['Hobbies'] = data_set['Hobbies'].map(hobbies_mapping)
data_set['Academic Performance'] = data_set['Academic Performance'].map(academic_mapping)
data_set['Co-curriculum Activities'] = data_set['Co-curriculum Activities'].map(cocuriculum_mapping)
data_set['Leadership'] = data_set['Leadership'].map(leadership_mapping)
data_set['Estimated Income'] = data_set['Estimated Income'].map(income_mapping)
data_set['Faculty'] = data_set['Faculty'].map(faculty_mapping)

# Separate numeric and categorical columns
numeric_columns = ['Personality Traits', 'Behavioural Traits', 'Hobbies', 'Academic Performance',
                   'Co-curriculum Activities', 'Leadership']
categorical_columns = ['Hogwarts House']

# Separate features and target variable
X = data_set[numeric_columns]
y = data_set[categorical_columns]

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute numeric columns
X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])

# Impute categorical columns
y[categorical_columns] = categorical_imputer.fit_transform(y[categorical_columns])

# Apply MinMaxScaler to scale data between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply SelectKBest with chi-squared test
k = 4
select_k_best = SelectKBest(score_func=chi2, k=k)
X_new = select_k_best.fit_transform(X_scaled, y.values.ravel())

# Get the selected features
selected_features = select_k_best.get_support(indices=True)
feature_names = X.columns
selected_feature_names = feature_names[selected_features]

# Apply RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=4)
rfe.fit(X_new, y.values.ravel())

# Get the selected features from RFE
rfe_selected_features = rfe.support_
rfe_feature_names = selected_feature_names[rfe_selected_features]

# Using the features selected by RFE
X_final = X_scaled[:, selected_features][:, rfe_selected_features]

# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=6, criterion='entropy', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
}

# Fit classifiers
for clf in classifiers.values():
    clf.fit(X_final, y.values.ravel())

def predict_house(input_data):
    # Preprocess input data
    input_data = np.array([input_data])
    norm_input_data = scaler.transform(input_data)
    input_data_final = norm_input_data[:, selected_features][:, rfe_selected_features]

    # Predict using classifiers
    votes = {house: 0 for house in np.unique(y.values.ravel())}
    for name, clf in classifiers.items():
        prediction = clf.predict(input_data_final)
        votes[prediction[0]] += 1

    final_house = max(votes, key=votes.get)
    percent_votes = (votes[final_house] / sum(votes.values())) * 100

    return final_house, percent_votes

# Streamlit interface
st.title("Hogwarts House Sorting")

# Input fields for the user
personality_traits = st.selectbox("Personality Traits", list(personality_traits_mapping.keys()))
behavioural_traits = st.selectbox("Behavioural Traits", list(behavioural_traits_mapping.keys()))
hobbies = st.selectbox("Hobbies", list(hobbies_mapping.keys()))
academic_performance = st.selectbox("Academic Performance", list(academic_mapping.keys()))
co_curriculum_activities = st.selectbox("Co-curriculum Activities", list(cocuriculum_mapping.keys()))
leadership = st.selectbox("Leadership", list(leadership_mapping.keys()))

# Convert inputs to numeric
input_data = [
    personality_traits_mapping[personality_traits],
    behavioural_traits_mapping[behavioural_traits],
    hobbies_mapping[hobbies],
    academic_mapping[academic_performance],
    cocuriculum_mapping[co_curriculum_activities],
    leadership_mapping[leadership]
]

if st.button("Predict House"):
    if None not in input_data:
        final_house, percent_votes = predict_house(input_data)
        st.write(f"The new data is classified into '{final_house}' house with {percent_votes:.2f}% votes.")
    else:
        st.error("Please fill in all the fields.")
