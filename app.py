import pandas as pd
import gradio as gr
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("punkt", quiet=True)

# Load dataset (must be in the same folder as this file)
df = pd.read_csv("final_medical_dataset.csv")

# Combine symptom columns
symptom_cols = [col for col in df.columns if "symptom_" in col]
df["symptoms"] = df[symptom_cols].fillna("").agg(" ".join, axis=1)

X = df["symptoms"]
y = df["disease"]

# Recommendations
if "recommendation" in df.columns:
    recommendations = dict(zip(df["disease"], df["recommendation"]))
else:
    recommendations = {
        disease: "Please consult a medical professional."
        for disease in df["disease"].unique()
    }

# Precautions
precautions_dict = {}
for _, row in df.drop_duplicates(subset=["disease"]).iterrows():
    disease = row["disease"]
    precautions = [
        row.get("Precaution_1", ""),
        row.get("Precaution_2", ""),
        row.get("Precaution_3", ""),
        row.get("Precaution_4", ""),
    ]
    precautions_dict[disease] = [
        p for p in precautions if pd.notna(p) and str(p).strip()
    ]

# Model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(X, y)

EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "unconscious",
    "stroke",
]

def virtual_doctor(user_symptoms, age, gender):

    # Required field validation
    if not user_symptoms.strip():
        return "❌ Symptoms are required."

    if age is None:
        return "❌ Age is required."

    if not gender:
        return "❌ Gender is required."

    # Emergency symptom check
    warning = ""
    if any(word in user_symptoms.lower() for word in EMERGENCY_KEYWORDS):
        warning = "🚨 Seek emergency medical attention immediately.\n\n"

    # Predict disease
    predicted_condition = model.predict([user_symptoms])[0]

    try:
        confidence = max(model.predict_proba([user_symptoms])[0]) * 100
    except Exception:
        confidence = None

    # Low confidence check
    if confidence is not None and confidence < 20:
        return """
⚠️ Not enough information.

Please enter multiple symptoms such as:
• fever cough headache
• vomiting nausea stomach pain
• chest pain breathing difficulty

instead of only one symptom.
"""

    # Get recommendation and precautions
    advice = recommendations.get(
        predicted_condition,
        "Consult a doctor."
    )

    precautions = precautions_dict.get(
        predicted_condition,
        ["No specific precautions found."]
    )

    precautions_text = "\n".join(
        f"- {str(p).replace('_', ' ')}"
        for p in precautions
    )

    # Confidence level
    level = "Unknown"

    if confidence is not None:
        if confidence >= 70:
            level = "High"
        elif confidence >= 40:
            level = "Medium"
        else:
            level = "Low"

    # Build response
    response = f"""{warning}
⚠️ MEDICAL DISCLAIMER:
This AI is for educational purposes only and is not a substitute for professional medical advice.

👤 Patient Info:
Age: {age}
Gender: {gender}

🩺 Possible Condition:
{predicted_condition}
"""

    if confidence is not None:
        response += f"\n📊 Confidence: {confidence:.2f}% ({level})\n"

    response += f"""
💡 Recommendation:
{advice}

🛡️ Precautions:
{precautions_text}

🚨 If symptoms are severe, persistent, or worsening,
please consult a qualified healthcare professional immediately.
"""

    return response

app = gr.Interface(
    fn=virtual_doctor,
    inputs=[
        gr.Textbox(
            lines=6,
            placeholder="Enter symptoms (e.g., fever cough headache)",
            label="Symptoms *"
        ),
        gr.Number(
            label="Age *",
            minimum=0,
            maximum=120,
            placeholder="Enter your age"
        ),
        gr.Radio(
            ["Male", "Female", "Other"],
            label="Gender *"
        )
    ],
    outputs=gr.Textbox(
        lines=15,
        label="Prediction & Recommendation",

    ),
    title="🩺 AI-Powered Virtual Doctor",
    description="Real-time symptom analysis using Machine Learning",
    theme=gr.themes.Default()
)

if __name__ == "__main__":
    app.launch()