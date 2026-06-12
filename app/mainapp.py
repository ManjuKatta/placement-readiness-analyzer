import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go


MODEL_PATH = "models/random_forest_model.pkl"
DEBUG_MODE = False

FEATURE_COLUMNS = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
    "Technical_Score",
    "Readiness_Score",
    "Profile_Strength",
]

TARGET_LEVELS = {
    "Academics": 80,
    "Communication": 80,
    "Aptitude": 75,
    "Projects": 80,
    "Industry Readiness": 80,
}


st.set_page_config(
    page_title="Placement Readiness Analyzer",
    page_icon="🚀",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --bg: #0d0f0e;
                --panel: #171a18;
                --panel-2: #1f2421;
                --line: #2d342f;
                --text: #f7fff9;
                --muted: #b7c4bb;
                --accent: #1ed760;
                --accent-2: #37f27a;
                --warn: #f5c542;
                --danger: #ff6868;
                --blue: #67b7ff;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 0%, rgba(30, 215, 96, 0.18), transparent 30rem),
                    radial-gradient(circle at 90% 10%, rgba(103, 183, 255, 0.10), transparent 24rem),
                    linear-gradient(180deg, #141815 0%, var(--bg) 45%);
                color: var(--text);
            }

            header[data-testid="stHeader"],
            #MainMenu,
            footer,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {
                display: none;
                visibility: hidden;
            }

            .block-container {
                max-width: 1160px;
                padding-top: 1.6rem;
                padding-bottom: 2.5rem;
            }

            h1, h2, h3, p, span, label,
            .stMarkdown, .stMarkdown p, .stMarkdown li {
                color: var(--text);
            }

            .hero {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1.1rem;
            }

            .eyebrow {
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .title {
                color: var(--text);
                font-size: 2.15rem;
                line-height: 1;
                font-weight: 900;
                letter-spacing: 0;
            }

            .subtitle {
                color: var(--muted);
                margin-top: 0.45rem;
                font-size: 0.98rem;
                max-width: 680px;
            }

            .section-title {
                color: var(--text);
                font-size: 1.05rem;
                font-weight: 850;
                margin: 1.2rem 0 0.55rem;
            }

            .panel {
                background: rgba(23, 26, 24, 0.92);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 1rem;
            }

            .kpi-card {
                min-height: 122px;
                background: linear-gradient(180deg, rgba(31, 36, 33, 0.98), rgba(20, 23, 21, 0.98));
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 750;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .kpi-value {
                color: var(--text);
                font-size: 1.9rem;
                font-weight: 900;
                line-height: 1.1;
                margin-top: 0.45rem;
            }

            .kpi-help {
                color: var(--muted);
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }

            .summary-card {
                background: rgba(23, 26, 24, 0.92);
                border: 1px solid var(--line);
                border-left: 4px solid var(--accent);
                border-radius: 10px;
                padding: 1rem 1.1rem;
                margin-top: 1rem;
            }

            .summary-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 850;
                margin-bottom: 0.45rem;
            }

            .summary-text {
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.55;
            }

            .tag-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.4rem;
            }

            .tag {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                background: rgba(30, 215, 96, 0.12);
                border: 1px solid rgba(30, 215, 96, 0.34);
                border-radius: 999px;
                color: var(--text);
                font-size: 0.86rem;
                font-weight: 750;
                padding: 0.42rem 0.7rem;
            }

            .insight {
                background: rgba(31, 36, 33, 0.78);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.72rem 0.85rem;
                margin-bottom: 0.5rem;
                color: var(--text);
                font-size: 0.92rem;
            }

            .good {
                border-left: 3px solid var(--accent);
            }

            .risk {
                border-left: 3px solid var(--warn);
            }

            .roadmap-card {
                background: rgba(31, 36, 33, 0.82);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 0.9rem;
                min-height: 138px;
            }

            .week {
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .roadmap-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 850;
                margin-top: 0.4rem;
            }

            .roadmap-text {
                color: var(--muted);
                font-size: 0.88rem;
                margin-top: 0.35rem;
            }

            div[data-testid="stWidgetLabel"] p {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 750;
            }

            div[data-testid="stSlider"],
            div[data-testid="stSelectbox"] {
                background: rgba(23, 26, 24, 0.9);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.65rem 0.8rem 0.55rem;
                margin-bottom: 0.65rem;
            }

            div[data-baseweb="select"] > div {
                min-height: 2.1rem;
                background: var(--panel-2);
                border-color: var(--line);
                border-radius: 8px;
                color: var(--text);
            }

            div[data-baseweb="select"] span,
            div[data-baseweb="select"] svg,
            div[data-testid="stSlider"] span,
            div[data-testid="stSlider"] div {
                color: var(--text);
                fill: var(--text);
            }

            div[data-testid="stSlider"] [role="slider"] {
                width: 0.9rem;
                height: 0.9rem;
                background: var(--accent);
                box-shadow: 0 0 0 4px rgba(30, 215, 96, 0.14);
            }

            .stButton > button {
                min-height: 2.45rem;
                border: 0;
                border-radius: 999px;
                background: var(--accent);
                color: #07130b;
                font-weight: 900;
                box-shadow: 0 12px 28px rgba(30, 215, 96, 0.22);
            }

            .stButton > button:hover {
                background: var(--accent-2);
                color: #07130b;
                transform: translateY(-1px);
            }

            div[data-testid="stAlert"] {
                background: rgba(31, 36, 33, 0.82);
                border: 1px solid var(--line);
                border-radius: 8px;
                color: var(--text);
            }

            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div,
            div[data-testid="stAlert"] span {
                color: var(--text);
            }

            hr {
                border-color: var(--line);
                margin: 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_student_features(inputs):
    internship_value = 1 if inputs["internship"] == "Yes" else 0
    iq = inputs["aptitude_score"]
    academic_performance = inputs["academic_rating"] * 10
    communication_skills = inputs["communication_rating"] * 10
    extra_curricular_score = inputs["extracurricular_rating"] * 10

    # Keep these formulas aligned with the training feature engineering.
    technical_score = (
        (iq * 0.30)
        + (academic_performance * 0.40)
        + (inputs["projects"] * 10 * 0.30)
    )

    readiness_score = (
        (inputs["cgpa"] * 10 * 0.40)
        + (communication_skills * 0.30)
        + (extra_curricular_score * 0.10)
        + (internship_value * 20 * 0.20)
    )

    profile_strength = technical_score + readiness_score

    return pd.DataFrame([{
        "IQ": iq,
        "Prev_Sem_Result": inputs["prev_sem"],
        "CGPA": inputs["cgpa"],
        "Academic_Performance": academic_performance,
        "Internship_Experience": internship_value,
        "Extra_Curricular_Score": extra_curricular_score,
        "Communication_Skills": communication_skills,
        "Projects_Completed": inputs["projects"],
        "Technical_Score": technical_score,
        "Readiness_Score": readiness_score,
        "Profile_Strength": profile_strength,
    }])[FEATURE_COLUMNS]


def get_grade(probability):
    if probability >= 90:
        return "Highly Competitive"
    if probability >= 80:
        return "Placement Ready"
    if probability >= 70:
        return "Competitive"
    if probability >= 60:
        return "Developing"
    return "Beginner"


def get_status(probability):
    if probability >= 80:
        return "Placement Ready", "Ready for interviews and applications"
    if probability >= 60:
        return "Needs Improvement", "Close to ready with focused practice"
    return "Placement Readiness: Low", "Needs a structured improvement plan"


def build_profile_summary(strengths, weaknesses, recommendations):
    if strengths:
        strength_text = "Your " + ", ".join(strengths[:2]).lower()
        strength_text += " are working in your favor."
    else:
        strength_text = "Your profile has clear room to grow with focused effort."

    if weaknesses and "No major gaps detected" not in weaknesses[0]:
        gap_text = "However, " + ", ".join(weaknesses[:2]).lower()
        gap_text += " are currently limiting placement readiness."
    else:
        gap_text = "No major profile gaps were detected, so the focus should shift to interview preparation and applications."

    focus = recommendations[0] if recommendations else "Maintain consistency and keep applying to relevant opportunities."

    return f"{strength_text} {gap_text} Focus next on this: {focus}"


def analyze_profile(inputs, features):
    skill_scores = {
        "Academics": round(((inputs["cgpa"] * 10) + (inputs["prev_sem"] * 10) + features["Academic_Performance"].iloc[0]) / 3, 1),
        "Communication": inputs["communication_rating"] * 10,
        "Aptitude": inputs["aptitude_score"],
        "Projects": min(inputs["projects"] * 20, 100),
        "Industry Readiness": min((inputs["projects"] * 14) + (features["Internship_Experience"].iloc[0] * 28) + (inputs["communication_rating"] * 4), 100),
    }

    strengths = []
    weaknesses = []
    recommendations = []

    if inputs["cgpa"] >= 8:
        strengths.append("Strong CGPA and academic consistency")
    elif inputs["cgpa"] < 7:
        weaknesses.append("Low CGPA compared with placement-ready profiles")
        recommendations.extend([
            "Improve academic consistency by revising core subjects weekly.",
            "Prioritize high-weight subjects and previous semester weak areas.",
        ])

    if inputs["academic_rating"] >= 8:
        strengths.append("Strong academic performance rating")
    elif inputs["academic_rating"] < 7:
        weaknesses.append("Academic performance needs improvement")
        recommendations.append("Create a subject-wise study tracker and review it every weekend.")

    if inputs["projects"] >= 4:
        strengths.append("Good project portfolio")
    else:
        weaknesses.append("Project portfolio is below the ideal placement benchmark")
        recommendations.append("Build 2 additional portfolio projects and publish them on GitHub.")

    if inputs["internship"] == "Yes":
        strengths.append("Internship experience improves industry readiness")
    else:
        weaknesses.append("No internship experience yet")
        recommendations.append("Apply for internships, open-source tasks, or freelancing work.")

    if inputs["communication_rating"] >= 8:
        strengths.append("Strong communication skills")
    elif inputs["communication_rating"] < 7:
        weaknesses.append("Communication score can reduce interview performance")
        recommendations.extend([
            "Practice mock interviews twice per week.",
            "Join group discussions or present one topic weekly.",
        ])

    if inputs["aptitude_score"] >= 75:
        strengths.append("Strong aptitude and analytical ability")
    elif inputs["aptitude_score"] < 70:
        weaknesses.append("Aptitude score is below common screening expectations")
        recommendations.append("Solve aptitude questions for 30 minutes daily.")

    if inputs["extracurricular_rating"] < 6:
        weaknesses.append("Low extracurricular score may weaken overall profile depth")
        recommendations.append("Add leadership, volunteering, club, or event experience to your profile.")

    if not strengths:
        strengths.append("Assessment completed with clear improvement opportunities")

    if not weaknesses:
        weaknesses.append("No major gaps detected. Keep sharpening interview readiness.")

    if not recommendations:
        recommendations.append("Maintain consistency and start applying to role-specific opportunities.")

    return skill_scores, strengths, weaknesses, recommendations


def career_dna(inputs, probability):
    tags = []

    if inputs["cgpa"] >= 8 or inputs["academic_rating"] >= 8:
        tags.append("📚 Academic Performer")
    if inputs["projects"] >= 4:
        tags.append("💻 Builder Mindset")
    if inputs["communication_rating"] >= 8:
        tags.append("🎤 Effective Communicator")
    if inputs["aptitude_score"] >= 75:
        tags.append("🧠 Analytical Thinker")
    if inputs["internship"] == "Yes":
        tags.append("🏢 Industry Exposed")
    if probability >= 80:
        tags.append("🚀 Career Ready")
    if not tags:
        tags.append("🌱 Growth Profile")

    return tags


def build_roadmap(weaknesses, inputs):
    plan = []

    if inputs["communication_rating"] < 7:
        plan.append(("Week 1", "Interview Communication", "Run 3 mock interviews, record answers, and improve clarity using STAR format."))
    elif inputs["cgpa"] < 7 or inputs["academic_rating"] < 7:
        plan.append(("Week 1", "Academic Recovery", "Revise core subjects, prepare short notes, and solve previous exam questions."))
    else:
        plan.append(("Week 1", "Profile Positioning", "Update resume headline, LinkedIn summary, and project descriptions."))

    if inputs["aptitude_score"] < 70:
        plan.append(("Week 2", "Aptitude Practice", "Practice quantitative, logical reasoning, and verbal questions for 30 minutes daily."))
    else:
        plan.append(("Week 2", "Interview Prep", "Practice coding, HR, and technical interview questions for target roles."))

    if inputs["projects"] < 4:
        plan.append(("Week 3", "Portfolio Project", "Build and deploy one project with a clean README and measurable outcome."))
    else:
        plan.append(("Week 3", "Project Polish", "Improve GitHub READMEs, add screenshots, and write impact-focused case studies."))

    if inputs["internship"] == "No":
        plan.append(("Week 4", "Industry Exposure", "Apply to internships, freelance tasks, and campus opportunities with a tailored resume."))
    else:
        plan.append(("Week 4", "Application Sprint", "Apply to 15 roles, message recruiters, and prepare company-specific interview notes."))

    return plan[:4]


def validate_model_input(model, student):
    expected = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    actual = list(student.columns)
    return expected == actual, expected


def render_kpi_card(label, value, helper):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(item, kind):
    icon = "✓" if kind == "good" else "⚠"
    st.markdown(
        f'<div class="insight {kind}">{icon} {item}</div>',
        unsafe_allow_html=True,
    )


def render_tags(tags):
    tag_html = "".join([f'<span class="tag">{tag}</span>' for tag in tags])
    st.markdown(f'<div class="tag-row">{tag_html}</div>', unsafe_allow_html=True)


def render_skill_scorecard(skill_scores):
    rows = []
    for skill, score in skill_scores.items():
        rows.append({
            "Skill": skill,
            "Level": "Current",
            "Score": score,
        })
        rows.append({
            "Skill": skill,
            "Level": "Target",
            "Score": TARGET_LEVELS[skill],
        })

    chart_df = pd.DataFrame(rows)
    fig = px.bar(
        chart_df,
        y="Skill",
        x="Score",
        color="Level",
        orientation="h",
        barmode="group",
        text="Score",
        color_discrete_map={
            "Current": "#1ed760",
            "Target": "#4b5563",
        },
    )

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f7fff9",
        margin=dict(l=16, r=16, t=12, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            range=[0, 105],
            gridcolor="#2d342f",
            title="Score",
            tickfont=dict(color="#b7c4bb"),
            title_font=dict(color="#b7c4bb"),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#f7fff9"),
        ),
    )

    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        textfont=dict(color="#f7fff9", size=12),
        hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x}<extra></extra>",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix": "%", "font": {"color": "#f7fff9", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#b7c4bb"},
            "bar": {"color": "#1ed760"},
            "bgcolor": "#171a18",
            "borderwidth": 1,
            "bordercolor": "#2d342f",
            "steps": [
                {"range": [0, 60], "color": "#3a1717"},
                {"range": [60, 80], "color": "#3b2f12"},
                {"range": [80, 100], "color": "#12351f"},
            ],
        },
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=16, r=16, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f7fff9",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_input_section():
    st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        prev_sem = st.slider("Previous Semester Result", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        projects = st.slider("Projects Completed", min_value=0, max_value=10, value=3)
        internship = st.selectbox("Internship Experience", ["No", "Yes"])

    with col2:
        aptitude_score = st.slider("Aptitude Score", min_value=0, max_value=100, value=70)
        academic_rating = st.slider("Academic Rating", min_value=1, max_value=10, value=7)
        communication_rating = st.slider("Communication Rating", min_value=1, max_value=10, value=7)
        extracurricular_rating = st.slider("Extra Curricular Rating", min_value=1, max_value=10, value=5)

    predict_button = st.button("🚀 Analyze My Profile", use_container_width=True)

    return {
        "cgpa": cgpa,
        "prev_sem": prev_sem,
        "projects": projects,
        "internship": internship,
        "aptitude_score": aptitude_score,
        "academic_rating": academic_rating,
        "communication_rating": communication_rating,
        "extracurricular_rating": extracurricular_rating,
        "predict_button": predict_button,
    }


def render_dashboard(model, inputs):
    student = build_student_features(inputs)

    is_valid, expected_columns = validate_model_input(model, student)

    # Prediction validation: the app sends the same feature names and order that
    # the Random Forest exposes through feature_names_in_.
    if not is_valid:
        st.error("Model input columns do not match the trained model.")
        st.write("Expected:", expected_columns)
        st.write("Actual:", list(student.columns))
        return

    prediction = model.predict(student)[0]
    probability = model.predict_proba(student)
    placement_probability = probability[0][1] * 100
    grade = get_grade(placement_probability)
    status, status_help = get_status(placement_probability)

    skill_scores, strengths, weaknesses, recommendations = analyze_profile(inputs, student)
    tags = career_dna(inputs, placement_probability)
    roadmap = build_roadmap(weaknesses, inputs)
    profile_summary = build_profile_summary(strengths, weaknesses, recommendations)
    profile_strength_score = min(round(student["Profile_Strength"].iloc[0] / 2, 1), 100)

    st.markdown('<div class="section-title">Results Dashboard</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi_card("Placement Probability", f"{placement_probability:.0f}%", "Model confidence for placement")
    with kpi2:
        render_kpi_card("Readiness Tier", grade, "Overall profile maturity")
    with kpi3:
        render_kpi_card("Placement Status", status, status_help)
    with kpi4:
        render_kpi_card("Profile Strength", f"{profile_strength_score:.0f}/100", "Engineered readiness score")

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">Profile Summary</div>
            <div class="summary-text">{profile_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gauge_col, dna_col = st.columns([1, 1.2])
    with gauge_col:
        render_probability_gauge(placement_probability)

    with dna_col:
        st.markdown('<div class="section-title">Career DNA</div>', unsafe_allow_html=True)
        render_tags(tags)
        if DEBUG_MODE:
            st.caption(f"Model class prediction: {prediction}. Use probability and explanations for decision support.")

    st.markdown('<div class="section-title">Why This Result?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top strengths contributing to the prediction**")
        for item in strengths:
            render_insight(item, "good")

    with col2:
        st.markdown("**Improvement areas affecting readiness**")
        for item in weaknesses:
            render_insight(item, "risk")

    st.markdown('<div class="section-title">Skill Gap Analysis</div>', unsafe_allow_html=True)
    render_skill_scorecard(skill_scores)

    st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)
    for recommendation in recommendations:
        render_insight(recommendation, "good")

    st.markdown('<div class="section-title">30-Day Improvement Roadmap</div>', unsafe_allow_html=True)
    week_cols = st.columns(4)
    for index, (week, title, details) in enumerate(roadmap):
        with week_cols[index]:
            st.markdown(
                f"""
                <div class="roadmap-card">
                    <div class="week">{week}</div>
                    <div class="roadmap-title">{title}</div>
                    <div class="roadmap-text">{details}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if DEBUG_MODE:
        with st.expander("Model validation details"):
            st.caption("Use this section while validating deployment or retraining. Hide it before a public demo if needed.")
            st.dataframe(student, use_container_width=True)
            st.write("Feature columns match trained model:", is_valid)
            st.write("Expected model columns:", expected_columns)
            st.write("Placement probability:", round(placement_probability, 2))


def main():
    inject_css()
    model = load_model()

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            """
            <div class="eyebrow">Career assessment platform</div>
            <div class="title">Placement Readiness Analyzer</div>
            <div class="subtitle">
                Evaluate placement probability, understand profile gaps, and get a practical improvement plan built around your inputs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.write("")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    inputs = render_input_section()

    if inputs["predict_button"]:
        st.divider()
        render_dashboard(model, inputs)
    else:
        st.info("Adjust your profile and click Analyze My Profile to generate the dashboard.")


if __name__ == "__main__":
    main()
