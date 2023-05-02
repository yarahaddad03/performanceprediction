import streamlit as st
import pickle
import numpy as np
from PIL import Image

def load_model():
    with open('C:/Users/lenovo/Desktop/Capstone/saved_steps.pkl', 'rb') as file:
        data=pickle.load(file)
    return data

data = load_model()

rf_loaded = data["model"]
le_Gender = data["le_Gender"]
le_Country = data["le_Country"]
le_Scholarship = data["le_Scholarship"]
le_Performance = data["le_Performance"]
le_Governorates = data["le_Governorates"]
le_Schools = data["le_Schools"]
le_Diploma = data["le_Diploma"]

def show_predict_page():
    st.title("Student Performance Prediction")
    image = Image.open('OSB-ROTATOR-05.png')
    st.image(image, caption='Sunrise by the mountains')

show_predict_page()

Gender = (
    "M",
    "F"
)

Countries = (
    "Lebanon",
    "United Arab Emirates",
    "Saudi Arabia",
    "United States of America",
    "Jordan",
    "Kuwait",
    "Qatar",
    "Syria",
    "United Kingdom",
    "AUB (Development use only)",
    "Canada",
    "France",
    "Cote D'Ivoire",
    "Switzerland",
    "Bahrain",
    "Oman",
    "Tunisia",
    "Australia",
    "Netherlands",
    "Italy",
    "Egypt",
    "Malaysia",
    "Spain",
    "Singapore",
    "AUBMC (Development use only)",
    "Belgium",
    "Romania",
    "Ghana",
    "Germany",
    "Cyprus",
    "Tanzania",
    "Algeria",
    "Turkey",
    "Japan",
    "Greece",
    "Hungary",
    "Hong Kong",
    "Ireland",
    "Morocco",
    "Venezuela",
    "Yemen",
    "Burkina Faso",
)

Scholarship = (
    "Yes",
    "No",
)

Governorates = (
    "Akkar",
    "Baalbeck-Hermel",
    "Beirut",
    "Bekaa",
    "Mount Lebanon",
    "North Lebanon",
    "South Lebanon",
)

School = ('International College, Ras Beirut',
          'GrandLyceeFrancolibanais,bey',
          'Col.Prot.Francais, Beirut',
          'Rawdah H.Sch., Beirut',
          'College Notre Dame de Jamhour',
          'Lycee Abdul-Kader, Beirut',
          'College Louise Wegmann, Beirut',
          'American Comm. Sc /Beirut', 
          'Sagesse High School, Ain Saade', 
          'College Notre Dame de Nazareth', 
          'College MARISTE Champville ,Di', 
          'Hariri High School II, Beirut', 
          'Rafic Hariri High School,Sidon', 
          'St.Joseph Sch.Cornet Shahwan', 
          'GrandLyceeFrancolibanais,bey'
          )

Diploma = ('Leb Bac II- Basic Life Sci',
           'Leb Bac II-Sociology & Economy',
           'Leb Bac II- General Sci',
           'French Bacc. -Econ.& Sociology',
           'French Bacc. - Mathematics',
           'International Baccalaureate - Science', 
           'Leb Bac II-Humanities & Philo', 
           'French Bacc.  - Literary', 
           'Lebanese Bacc.II Exp.Sci.'
           )

Age = st.slider("Age at admission", 16, 50 , 18)
Gender = st.selectbox("Gender", Gender)
Country = st.selectbox("Country", Countries)
Scholarship = st.selectbox("Scholarship", Scholarship)
SAT = st.slider("SAT Score", 500, 1600, 1100)
Governorates = st.selectbox("Governorates", Governorates)
School = st.selectbox("School", School)
Diploma = st.selectbox("Diploma", Diploma)

predict = st.button("Predict Performance")
if predict:
    X=np.array([[Age, Gender, Country, Scholarship, SAT, Governorates, School, Diploma]])
    X[:,1] = le_Gender.transform(X[:,1])
    X[:,2] = le_Country.transform(X[:,2])
    X[:,3] = le_Scholarship.transform(X[:,3])
    X[:,5] = le_Governorates.transform(X[:,5])
    X[:,6] = le_Schools.transform(X[:,6])
    X[:,7] = le_Diploma.transform(X[:,7])
    X=X.astype(float)
    
    performance = rf_loaded.predict(X)
    if performance[0] == 0:
        st.subheader("This student will fail")
    else:
        st.subheader("This student will pass")
    
    prediction_proba= rf_loaded.predict_proba(X)
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
