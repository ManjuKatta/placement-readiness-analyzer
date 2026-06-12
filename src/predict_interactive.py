import joblib
import pandas as pd

# ----------------------------------
# Load Saved Model
# ----------------------------------

model = joblib.load(
    "models/random_forest_model.pkl"
)

print("\nSTUDENT PLACEMENT PREDICTION")
print("-" * 40)

# ----------------------------------
# User Friendly Inputs
# ----------------------------------

cgpa = float(input("CGPA: "))
prev_sem = float(input("Previous Semester Result: "))
projects = int(input("Projects Completed: "))
internship = int(input("Internship Experience (0 = No, 1 = Yes): "))

aptitude_score = int(
    input("Aptitude Score (0-100): ")
)

academic_rating = int(
    input("Academic Performance Rating (1-10): ")
)

communication_rating = int(
    input("Communication Confidence (1-10): ")
)

extra_curricular_rating = int(
    input("Extra Curricular Involvement (1-10): ")
)

# ----------------------------------
# Convert User Inputs
# To Dataset Features
# ----------------------------------

iq = aptitude_score

academic_performance = academic_rating * 10

communication_skills = communication_rating * 10

extra_curricular_score = (
    extra_curricular_rating * 10
)

# ----------------------------------
# Feature Engineering
# Same Logic Used During Training
# ----------------------------------

technical_score = (
    (iq * 0.30)
    +
    (academic_performance * 0.40)
    +
    (projects * 10 * 0.30)
)

readiness_score = (
    (cgpa * 10 * 0.40)
    +
    (communication_skills * 0.30)
    +
    (extra_curricular_score * 0.10)
    +
    (internship * 20 * 0.20)
)

profile_strength = (
    technical_score
    +
    readiness_score
)

# ----------------------------------
# Create Prediction DataFrame
# ----------------------------------

student = pd.DataFrame([{
    "IQ": iq,
    "Prev_Sem_Result": prev_sem,
    "CGPA": cgpa,
    "Academic_Performance": academic_performance,
    "Internship_Experience": internship,
    "Extra_Curricular_Score": extra_curricular_score,
    "Communication_Skills": communication_skills,
    "Projects_Completed": projects,
    "Technical_Score": technical_score,
    "Readiness_Score": readiness_score,
    "Profile_Strength": profile_strength
}])

# ----------------------------------
# Prediction
# ----------------------------------

prediction = model.predict(student)

probability = model.predict_proba(student)

# ----------------------------------
# Results
# ----------------------------------

print("\n" + "=" * 40)
print("RESULT")
print("=" * 40)

if prediction[0] == 1:
    print("Placement Prediction: PLACED")
else:
    print("Placement Prediction: NOT PLACED")

print(
    f"Placement Probability: {probability[0][1] * 100:.2f}%"
)

print(f"Technical Score: {technical_score:.2f}")
print(f"Readiness Score: {readiness_score:.2f}")
print(f"Profile Strength: {profile_strength:.2f}")