# 🎓 AI Student Performance Predictor

A Streamlit web app that predicts a student's likely academic performance from
subject marks, attendance and daily study hours, identifies weaknesses, and
gives personalized improvement suggestions.

## Features

- **Performance prediction**: predicted final score (0-100), letter grade and performance level
- **Model confidence**: probability of each performance level
- **Weakness detection**: weak/critical subjects, uneven results, low attendance, too little (or too much) study time
- **Personalized suggestions**: subject-specific tips plus model-estimated "what-if" gains (e.g. +10% attendance, +1 study hour)
- **Daily study plan**: splits study time across subjects, giving weaker subjects more time
- **Downloadable report** as a text file
- **Optional custom training data**: upload your own CSV to train the model on real records

## Tech stack

- [Streamlit](https://streamlit.io/): user interface
- [scikit-learn](https://scikit-learn.org/): Random Forest regressor and classifier
- [Pandas](https://pandas.pydata.org/) / NumPy: data handling

## How it works

1. A `RandomForestRegressor` predicts the final score from three features:
   average marks, attendance (%) and daily study hours.
2. A `RandomForestClassifier` estimates the probability of each performance
   level (Excellent, Good, Average, Below Average, At Risk).
3. Rule-based analysis flags weaknesses; the trained model is queried with
   modified inputs to estimate how much each improvement would help.

By default the models train on **synthetic demo data** generated inside the app
(so it works out of the box). Predictions are therefore illustrative. For more
meaningful results, upload a CSV of real student records.

### Custom CSV format

Upload a CSV (in the sidebar) with at least these columns, at least 30 rows:

| avg_marks | attendance | study_hours | final_score |
|-----------|------------|-------------|-------------|
| 72.5      | 85         | 3.0         | 74.0        |

Rules: `avg_marks`, `attendance`, `final_score` must be 0-100; `study_hours` 0-24.

## Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project structure

```
.
├── app.py             # Streamlit app + ML logic
├── requirements.txt   # Python dependencies
└── README.md
```

## Deploy on Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.

## Disclaimer

This tool gives data-driven estimates to support students and teachers. It is
not a guarantee of results and should not be used as the sole basis for any
academic decision.
