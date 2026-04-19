import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.metrics import accuracy_score, confusion_matrix 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 
from collections import Counter 

@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    df.fillna("None", inplace=True)
    return df
df = load_data()

mlb = MultiLabelBinarizer()

df['symptoms_list'] = df.iloc[:, 1:].values.tolist()
df['symptoms_list'] = df['symptoms_list'].apply(
    lambda x: [str(i) for i in x if str(i) != 'nan' and str(i) != 'None']
)

X = mlb.fit_transform(df['symptoms_list'])
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model()

st.set_page_config(page_title="AI Health Assistant", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Health Assistant")
st.caption("Smart symptom-based disease prediction system")

st.sidebar.title("About")
st.sidebar.info("Predict diseases based on symptoms using ML")
st.sidebar.warning("Not a medical diagnosis tool")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "Analytics"]
)

if menu == "Home":
    st.subheader("Welcome")
    st.write("Select symptoms and get predictions instantly.")

    col1, col2 = st.columns(2)
    col1.metric("Total Diseases", len(np.unique(y)))
    col2.metric("Total Symptoms", len(mlb.classes_))

elif menu == "Predict":
    st.subheader("Select Your Symptoms")

    search = st.text_input("Search symptoms")
    filtered = [s for s in mlb.classes_ if search.lower() in s.lower()]

    symptoms = st.multiselect("Symptoms", filtered)
    st.write(f"Selected: {len(symptoms)} symptoms")

    if st.button("Analyze Symptoms"):
        if not symptoms:
            st.warning("Please select symptoms")
        else:
            with st.spinner("Analyzing..."):
                X_input = mlb.transform([symptoms])
                prediction = model.predict(X_input)[0]
                probs = model.predict_proba(X_input)[0]

            st.success(f"Most Likely Condition: {prediction}")
            st.metric("Confidence", f"{max(probs)*100:.2f}%")

            top_indices = np.argsort(probs)[-5:][::-1]

            chart_data = pd.DataFrame({
                "Disease": model.classes_[top_indices],
                "Probability": probs[top_indices]
            })

            fig = px.bar(
                chart_data,
                x="Disease",
                y="Probability",
                color="Probability",
                title="Prediction Confidence",
                text="Probability"
            )

            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)

elif menu == "Analytics":
    st.subheader("Model Performance")

    y_pred = model.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(cm, cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Top Symptoms")

    all_symptoms = []
    for row in df['symptoms_list']:
        all_symptoms.extend(row)

    symptom_counts = Counter(all_symptoms)
    common = pd.DataFrame(symptom_counts.items(), columns=['Symptom','Count'])
    common = common.sort_values(by='Count', ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Count', y='Symptom', data=common, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Disease Distribution")

    fig3, ax3 = plt.subplots()
    df['Disease'].value_counts().head(8).plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax3
    )
    ax3.set_ylabel("")
    st.pyplot(fig3)
