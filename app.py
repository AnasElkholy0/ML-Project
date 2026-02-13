import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# =========================
# Page Config & Style
# =========================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS Design
st.markdown("""
<style>
    .main {
        background-color: #f0f7ff;
    }

    h1 {
        color: #004b8d !important;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #0077cc;
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #005fa3;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

train, test = load_data()

# =========================
# Helper Functions
# =========================
def extract_title(name):
    if pd.isna(name):
        return "Unknown"
    parts = name.split(',')
    if len(parts) > 1:
        return parts[1].split('.')[0].strip()
    return "Unknown"

def preprocess(df):
    df = df.copy()

    df['Title'] = df['Name'].apply(extract_title)

    title_map = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare',
        'Rev': 'Rare', 'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_map)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    age_median = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

# =========================
# Preprocess Data
# =========================
train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(columns=['Survived'])
y = train_proc['Survived']

X['Pclass'] = X['Pclass'].astype(str)
test_proc['Pclass'] = test_proc['Pclass'].astype(str)
X['IsAlone'] = X['IsAlone'].astype(str)
test_proc['IsAlone'] = test_proc['IsAlone'].astype(str)

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title', 'IsAlone']

num_transformer = SimpleImputer(strategy='median')
cat_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=200, random_state=42)
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
model.fit(X, y)

# =========================
# UI
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# User Form
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧍 Passenger Information")

Pclass = st.selectbox("Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Siblings aboard", 0, 10, 0)
Parch = st.number_input("Parents/children aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 30.0)
Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
if st.button("Predict"):
    data = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked,
        "Name": "Test Name",
        "Ticket": "0000",
        "Cabin": np.nan
    }])

    data_proc = preprocess(data)
    data_proc["Pclass"] = data_proc["Pclass"].astype(str)
    data_proc["IsAlone"] = data_proc["IsAlone"].astype(str)

    pred = model.predict(data_proc)[0]
    result = "🟢 Survived" if pred == 1 else "🔴 Did NOT Survive"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")
    st.success(result)
    st.markdown('</div>', unsafe_allow_html=True)
