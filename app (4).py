"""AI Student Performance Predictor

A Streamlit app that takes a student's subject marks, attendance and daily
study hours, then:
  1. predicts the likely final academic score and performance level,
  2. identifies weak subjects and risk factors,
  3. gives personalized improvement suggestions (including "what-if" gains
     estimated by the trained model) and a suggested daily study plan.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FEATURES = ["avg_marks", "attendance", "study_hours"]
TARGET = "final_score"
MIN_CUSTOM_ROWS = 30

# (minimum score, level name, letter grade) - checked from highest to lowest
LEVELS = [
    (85, "Excellent", "A"),
    (70, "Good", "B"),
    (55, "Average", "C"),
    (40, "Below Average", "D"),
    (0, "At Risk", "F"),
]

DEFAULT_SUBJECTS = pd.DataFrame(
    {
        "Subject": ["Mathematics", "Physics", "Chemistry", "English", "Computer Science"],
        "Marks": [72.0, 58.0, 45.0, 80.0, 85.0],
    }
)


# --------------------------------------------------------------------------- #
# Data + model
# --------------------------------------------------------------------------- #
def score_to_level(score: float) -> tuple[str, str]:
    """Map a 0-100 score to (performance level, letter grade)."""
    for minimum, level, grade in LEVELS:
        if score >= minimum:
            return level, grade
    return LEVELS[-1][1], LEVELS[-1][2]


def generate_synthetic_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic training set.

    Final score rises with marks, attendance and study hours (with diminishing
    returns on study time) plus random noise.
    """
    rng = np.random.default_rng(seed)
    # Mostly "typical" students, plus ~25% uniformly spread values so the model
    # also sees extreme cases (a Random Forest cannot extrapolate beyond its data).
    extreme = rng.random((3, n)) < 0.25
    avg_marks = np.clip(np.where(extreme[0], rng.uniform(0, 100, n), rng.normal(65, 15, n)), 0, 100)
    attendance = np.clip(np.where(extreme[1], rng.uniform(0, 100, n), rng.normal(80, 12, n)), 0, 100)
    study_hours = np.clip(
        np.where(extreme[2], rng.uniform(0, 12, n), rng.gamma(shape=3.0, scale=1.1, size=n)), 0, 12
    )

    study_effect = 25 * (1 - np.exp(-study_hours / 3))
    noise = rng.normal(0, 3, n)
    final = 0.60 * avg_marks + 0.15 * attendance + study_effect + noise

    return pd.DataFrame(
        {
            "avg_marks": avg_marks.round(1),
            "attendance": attendance.round(1),
            "study_hours": study_hours.round(2),
            TARGET: np.clip(final, 0, 100).round(1),
        }
    )


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a user-supplied training DataFrame."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(required)}."
        )
    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < MIN_CUSTOM_ROWS:
        raise ValueError(
            f"Need at least {MIN_CUSTOM_ROWS} valid rows to train, found {len(df)}."
        )
    if (df[["avg_marks", "attendance", TARGET]] < 0).any().any() or (
        df[["avg_marks", "attendance", TARGET]] > 100
    ).any().any():
        raise ValueError("avg_marks, attendance and final_score must be between 0 and 100.")
    if (df["study_hours"] < 0).any() or (df["study_hours"] > 24).any():
        raise ValueError("study_hours must be between 0 and 24.")
    return df.reset_index(drop=True)


def train_models(df: pd.DataFrame, seed: int = 42) -> dict:
    """Train a regressor (score) and a classifier (performance level)."""
    X = df[FEATURES]
    y = df[TARGET]
    levels = y.apply(lambda s: score_to_level(s)[0])

    X_tr, X_te, y_tr, y_te, l_tr, _ = train_test_split(
        X, y, levels, test_size=0.2, random_state=seed
    )

    regressor = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )
    regressor.fit(X_tr, y_tr)
    pred = regressor.predict(X_te)

    classifier = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )
    classifier.fit(X_tr, l_tr)

    return {
        "regressor": regressor,
        "classifier": classifier,
        "r2": float(r2_score(y_te, pred)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "n_rows": int(len(df)),
        "importances": pd.Series(regressor.feature_importances_, index=FEATURES),
    }


def predict_student(models: dict, avg_marks: float, attendance: float, study_hours: float) -> dict:
    """Predict score, level, grade and level probabilities for one student."""
    X = pd.DataFrame([[avg_marks, attendance, study_hours]], columns=FEATURES)
    score = float(np.clip(models["regressor"].predict(X)[0], 0, 100))
    level, grade = score_to_level(score)
    clf = models["classifier"]
    proba = pd.Series(clf.predict_proba(X)[0], index=clf.classes_)
    return {"score": score, "level": level, "grade": grade, "proba": proba}


# --------------------------------------------------------------------------- #
# Analysis + suggestions
# --------------------------------------------------------------------------- #
def subject_status(marks: float) -> str:
    if marks < 40:
        return "Critical"
    if marks < 55:
        return "Weak"
    if marks < 70:
        return "Moderate"
    return "Strong"


def analyze_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
    out = subjects.copy()
    out["Status"] = out["Marks"].apply(subject_status)
    return out


def find_weaknesses(subjects: pd.DataFrame, attendance: float, study_hours: float) -> list[str]:
    """Return a list of plain-language problems found in the student's data."""
    issues: list[str] = []

    for _, row in subjects[subjects["Marks"] < 40].iterrows():
        issues.append(f"**{row['Subject']}** is at a critical level ({row['Marks']:.0f}/100) and is at risk of failing.")
    for _, row in subjects[(subjects["Marks"] >= 40) & (subjects["Marks"] < 55)].iterrows():
        issues.append(f"**{row['Subject']}** is weak ({row['Marks']:.0f}/100) and needs focused revision.")

    if len(subjects) >= 2:
        spread = float(subjects["Marks"].max() - subjects["Marks"].min())
        if spread >= 30:
            issues.append(
                f"Performance is uneven: a {spread:.0f}-point gap between your best and weakest subject."
            )

    if attendance < 60:
        issues.append(f"Attendance is very low ({attendance:.0f}%), so lessons and explanations are being missed.")
    elif attendance < 75:
        issues.append(f"Attendance ({attendance:.0f}%) is below the commonly required 75%.")

    if study_hours < 1:
        issues.append("Almost no daily study time (under 1 hour).")
    elif study_hours < 2:
        issues.append(f"Study time ({study_hours:g} h/day) is low for consistent progress.")
    elif study_hours > 9:
        issues.append(
            f"Study time ({study_hours:g} h/day) is very high; watch for burnout and make sure sleep and breaks are protected."
        )
    return issues


def what_if_gains(models: dict, avg_marks: float, attendance: float, study_hours: float, base_score: float) -> list[dict]:
    """Use the trained model to estimate the score gain of realistic changes."""
    scenarios = []
    if attendance < 100:
        new_att = min(100.0, attendance + 10)
        scenarios.append(
            ("Raise attendance to %.0f%%" % new_att, avg_marks, new_att, study_hours)
        )
    if study_hours < 12:
        scenarios.append(
            (f"Study {min(12.0, study_hours + 1):g} h/day (+1 h)", avg_marks, attendance, min(12.0, study_hours + 1))
        )
        scenarios.append(
            (f"Study {min(12.0, study_hours + 2):g} h/day (+2 h)", avg_marks, attendance, min(12.0, study_hours + 2))
        )
    if avg_marks < 100:
        new_marks = min(100.0, avg_marks + 5)
        scenarios.append(
            (f"Raise average marks by 5 points to {new_marks:.0f}", new_marks, attendance, study_hours)
        )

    results = []
    for label, m, a, h in scenarios:
        new_score = predict_student(models, m, a, h)["score"]
        results.append({"Action": label, "New predicted score": round(new_score, 1), "Gain": round(new_score - base_score, 1)})
    results.sort(key=lambda r: r["Gain"], reverse=True)
    return results


def generate_suggestions(subjects: pd.DataFrame, attendance: float, study_hours: float, gains: list[dict]) -> list[str]:
    tips: list[str] = []

    worst = subjects.sort_values("Marks").head(2)
    for _, row in worst.iterrows():
        if row["Marks"] < 70:
            tips.append(
                f"**{row['Subject']}**: revisit fundamentals, solve past papers, and ask your teacher about "
                f"the topics you lose marks on. Aim for +10 marks this term."
            )

    if attendance < 85:
        tips.append("Aim for 85%+ attendance: missed classes are the hardest content to recover on your own.")
    if study_hours < 3:
        tips.append("Build a fixed daily routine of 3 focused hours using 45-minute study blocks with 10-minute breaks.")

    useful = [g for g in gains if g["Gain"] > 0.5]
    if useful:
        best = useful[0]
        tips.append(
            f"Highest-impact change according to the model: **{best['Action']}** "
            f"(about {best['Gain']:+.1f} points)."
        )
    tips.append("Review each day's notes the same evening and test yourself weekly; active recall beats re-reading.")
    return tips


def study_plan(subjects: pd.DataFrame, study_hours: float) -> pd.DataFrame:
    """Split daily study time across subjects, giving weaker subjects more time."""
    weights = (100 - subjects["Marks"]).clip(lower=0) + 10
    share = weights / weights.sum()
    plan = subjects[["Subject"]].copy()
    plan["Share (%)"] = (share * 100).round(0).astype(int)
    plan["Hours / day"] = (share * study_hours).round(2)
    return plan.reset_index(drop=True)


def build_report(name: str, result: dict, avg: float, attendance: float, hours: float,
                 issues: list[str], tips: list[str]) -> str:
    clean = lambda s: s.replace("**", "")
    lines = [
        f"Student Performance Report - {name or 'Student'}",
        "=" * 50,
        f"Average marks : {avg:.1f}",
        f"Attendance    : {attendance:.0f}%",
        f"Study hours   : {hours:g} per day",
        f"Predicted score: {result['score']:.1f} ({result['level']}, grade {result['grade']})",
        "",
        "Weaknesses:",
        *([f"- {clean(i)}" for i in issues] or ["- None detected"]),
        "",
        "Suggestions:",
        *[f"- {clean(t)}" for t in tips],
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training the model...")
def get_default_models() -> dict:
    return train_models(generate_synthetic_data())


@st.cache_resource(show_spinner="Training on your data...")
def get_custom_models(df: pd.DataFrame) -> dict:
    return train_models(df)


def clean_subject_table(edited: pd.DataFrame) -> pd.DataFrame:
    df = edited.copy()
    df["Subject"] = df["Subject"].astype("string").str.strip()
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
    df = df.dropna(subset=["Subject", "Marks"])
    df = df[df["Subject"] != ""]
    df["Subject"] = df["Subject"].astype(str)
    df["Marks"] = df["Marks"].clip(0, 100).astype(float)
    return df.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="wide")
    st.title("🎓 AI Student Performance Predictor")
    st.caption("Enter marks, attendance and study hours to predict performance, find weaknesses and get a plan.")

    # ---- Sidebar: model source ------------------------------------------- #
    with st.sidebar:
        st.header("Model")
        uploaded = st.file_uploader(
            "Optional: train on your own CSV",
            type="csv",
            help="Columns required: avg_marks, attendance, study_hours, final_score",
        )
        models = None
        source = "synthetic demo data"
        if uploaded is not None:
            try:
                custom_df = validate_training_data(pd.read_csv(io.BytesIO(uploaded.getvalue())))
                models = get_custom_models(custom_df)
                source = f"your CSV ({models['n_rows']} rows)"
            except Exception as exc:  # noqa: BLE001 - show any parsing/validation problem
                st.error(f"Could not use that file: {exc}")
        if models is None:
            models = get_default_models()

        st.write(f"**Trained on:** {source}")
        st.metric("Model R² (hold-out)", f"{models['r2']:.3f}")
        st.metric("Mean abs. error", f"{models['mae']:.2f} pts")
        st.caption(
            "Without a CSV, the model learns from generated demo data, so predictions are "
            "illustrative estimates, not guarantees. Upload real records for better accuracy."
        )

    # ---- Inputs ----------------------------------------------------------- #
    st.subheader("1. Student details")
    name = st.text_input("Student name (optional)")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Subject marks (out of 100)** - add, edit or delete rows")
        edited = st.data_editor(
            DEFAULT_SUBJECTS,
            num_rows="dynamic",
            key="subjects_editor",
            column_config={
                "Subject": st.column_config.TextColumn("Subject", required=True),
                "Marks": st.column_config.NumberColumn("Marks", min_value=0, max_value=100, step=1, required=True),
            },
        )
    with right:
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)

    if not st.button("Predict performance", type="primary"):
        st.info("Fill in the details and press **Predict performance**.")
        return

    subjects = clean_subject_table(edited)
    if subjects.empty:
        st.error("Please enter at least one subject with marks.")
        return

    avg_marks = float(subjects["Marks"].mean())
    result = predict_student(models, avg_marks, float(attendance), float(study_hours))
    analysis = analyze_subjects(subjects)
    issues = find_weaknesses(subjects, float(attendance), float(study_hours))
    gains = what_if_gains(models, avg_marks, float(attendance), float(study_hours), result["score"])
    tips = generate_suggestions(subjects, float(attendance), float(study_hours), gains)

    # ---- Results ---------------------------------------------------------- #
    st.divider()
    st.subheader(f"2. Prediction{' for ' + name if name else ''}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted score", f"{result['score']:.1f} / 100")
    c2.metric("Grade", result["grade"])
    c3.metric("Level", result["level"])
    c4.metric("Current average", f"{avg_marks:.1f}")

    if result["level"] in ("Excellent", "Good"):
        st.success(f"Strong outlook: predicted **{result['level']}** performance.")
    elif result["level"] == "Average":
        st.warning("Average outlook: there is clear room to improve.")
    else:
        st.error(f"Warning: predicted **{result['level']}**. Early action is recommended.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Subject performance**")
        st.bar_chart(analysis.set_index("Subject")["Marks"])
        st.dataframe(analysis, hide_index=True)
    with col_b:
        st.markdown("**Model confidence by level**")
        st.bar_chart(result["proba"].round(3))
        st.markdown("**What influences the prediction**")
        st.bar_chart(models["importances"].rename({
            "avg_marks": "Marks", "attendance": "Attendance", "study_hours": "Study hours"}))

    st.subheader("3. Weaknesses identified")
    if issues:
        for issue in issues:
            st.markdown(f"- {issue}")
    else:
        st.success("No major weaknesses detected. Keep it up!")

    st.subheader("4. Personalized improvement suggestions")
    for tip in tips:
        st.markdown(f"- {tip}")

    if gains:
        st.markdown("**What-if scenarios (estimated by the model)**")
        st.dataframe(pd.DataFrame(gains), hide_index=True)

    st.markdown("**Suggested daily study plan**")
    if study_hours > 0:
        st.dataframe(study_plan(subjects, float(study_hours)), hide_index=True)
    else:
        st.caption("Set daily study hours above 0 to get a plan.")

    st.download_button(
        "Download report (.txt)",
        data=build_report(name, result, avg_marks, float(attendance), float(study_hours), issues, tips),
        file_name="student_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
