import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("Student.csv")

# ----------------------------
# Train Model
# ----------------------------

X = df[["Maths", "Physics", "Chemistry"]]
y = df["Result"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Result Prediction System")

st.write("Predict Result using Maths, Physics and Chemistry Marks")

# ----------------------------
# Sidebar Inputs
# ----------------------------

st.sidebar.header("Enter Student Details")

student_name = st.sidebar.text_input(
    "Student Name",
    "Aditya"
)

maths = st.sidebar.slider(
    "Maths Marks",
    0,
    100,
    60
)

physics = st.sidebar.slider(
    "Physics Marks",
    0,
    100,
    60
)

chemistry = st.sidebar.slider(
    "Chemistry Marks",
    0,
    100,
    60
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("🚀 Predict Result"):

    input_data = pd.DataFrame({
        "Maths": [maths],
        "Physics": [physics],
        "Chemistry": [chemistry]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"📊 Predicted Result: {prediction:.2f}"
    )

    # ----------------------------
    # Grade
    # ----------------------------

    if prediction >= 90:
        grade = "A+"
    elif prediction >= 80:
        grade = "A"
    elif prediction >= 70:
        grade = "B"
    elif prediction >= 60:
        grade = "C"
    else:
        grade = "D"

    st.info(f"🏆 Grade: {grade}")

    # ----------------------------
    # Performance Status
    # ----------------------------

    if prediction >= 75:
        st.success("🟢 Excellent Performance")
    elif prediction >= 50:
        st.warning("🟡 Needs Improvement")
    else:
        st.error("🔴 At Risk")

    # ----------------------------
    # Personalized Advice
    # ----------------------------

    st.subheader("📚 Personalized Study Advice")

    avg_marks = (maths + physics + chemistry) / 3

    if avg_marks < 50:
        st.warning("""
        • Focus on basic concepts.
        • Study daily for 3-4 hours.
        • Solve previous year papers.
        • Ask teachers when in doubt.
        """)

    elif avg_marks < 75:
        st.info("""
        • Good performance.
        • Increase revision.
        • Practice numerical questions.
        • Improve weak subjects.
        """)

    else:
        st.success("""
        • Excellent work.
        • Maintain consistency.
        • Attempt advanced problems.
        • Prepare for competitive exams.
        """)

    # ----------------------------
    # Visualization
    # ----------------------------

    st.subheader("📈 Marks Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))

    subjects = ["Maths", "Physics", "Chemistry"]
    marks = [maths, physics, chemistry]

    ax.bar(subjects, marks)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Marks")
    ax.set_title("Subject Wise Performance")

    st.pyplot(fig)

    # ----------------------------
    # Report
    # ----------------------------

    st.subheader("📄 Student Report")

    report = pd.DataFrame({
        "Student": [student_name],
        "Maths": [maths],
        "Physics": [physics],
        "Chemistry": [chemistry],
        "Predicted Result": [round(prediction, 2)],
        "Grade": [grade]
    })

    st.dataframe(report)

# ----------------------------
# Dataset Preview
# ----------------------------

st.subheader("📚 Training Dataset")

st.dataframe(df)