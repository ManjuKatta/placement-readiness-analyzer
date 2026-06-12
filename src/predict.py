import joblib
import pandas as pd

model = joblib.load(
    "models/random_forest_model.pkl"
)

print("\nENTER STUDENT DETAILS\n")

iq = int(input("IQ: "))
prev_sem = float(input("Previous Semester Result: "))
cgpa = float(input("CGPA: "))
academic = int(input("Academic Performance: "))
internship = int(input("Internship Experience (0/1): "))
extra = int(input("Extra Curricular Score: "))
communication = int(input("Communication Skills: "))
projects = int(input("Projects Completed: "))

technical_score = (
    iq * 0.30
    +
    academic * 0.40
    +
    projects * 10 * 0.30
)

readiness_score = (
    cgpa * 10 * 0.40
    +
    communication * 0.30
    +
    extra * 0.10
    +
    internship * 20 * 0.20
)

profile_strength = (
    technical_score
    +
    readiness_score
)

student = pd.DataFrame([{
    "IQ": iq,
    "Prev_Sem_Result": prev_sem,
    "CGPA": cgpa,
    "Academic_Performance": academic,
    "Internship_Experience": internship,
    "Extra_Curricular_Score": extra,
    "Communication_Skills": communication,
    "Projects_Completed": projects,
    "Technical_Score": technical_score,
    "Readiness_Score": readiness_score,
    "Profile_Strength": profile_strength
}])

prediction = model.predict(student)

probability = model.predict_proba(student)

print("\nRESULT")
print("-" * 30)

print(
    "Placement Prediction:",
    "Placed" if prediction[0] == 1 else "Not Placed"
)

print(
    "Placement Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)