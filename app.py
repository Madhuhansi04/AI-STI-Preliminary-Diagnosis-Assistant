from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np
import spacy
import re
from pymongo import MongoClient


app = Flask(__name__)


# =========================================================
# MONGODB CONNECTION
# =========================================================

try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)

    # Test connection
    mongo_client.admin.command("ping")

    mongo_db = mongo_client["sti_assessment_db"]
    assessments_collection = mongo_db["assessments"]

    print("========================================")
    print("MONGODB CONNECTED SUCCESSFULLY")
    print("Database: sti_assessment_db")
    print("Collection: assessments")
    print("========================================")

except Exception as e:
    print("========================================")
    print("MONGODB CONNECTION FAILED")
    print("Error:", e)
    print("========================================")

    assessments_collection = None

# =========================================================
# LOAD NLP MODEL
# =========================================================

try:

    nlp = spacy.load("en_core_web_sm")

    print("\n========================================")
    print("NLP MODEL LOADED SUCCESSFULLY")
    print("========================================")

except Exception as e:

    print("\n========================================")
    print("ERROR LOADING NLP MODEL")
    print("========================================")

    print(e)

    nlp = None


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# MODEL PATHS
# =========================================================

VIRAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "viral_sti_random_forest.pkl"
)

VIRAL_FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "viral_sti_features.pkl"
)


HPV_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hpv_biopsy_random_forest.pkl"
)

HPV_FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "hpv_biopsy_features.pkl"
)

HPV_IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "hpv_biopsy_imputer.pkl"
)

HPV_THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "hpv_biopsy_threshold.pkl"
)


# =========================================================
# LOAD VIRAL STI MODEL
# =========================================================

try:

    viral_model = joblib.load(
        VIRAL_MODEL_PATH
    )

    viral_features = joblib.load(
        VIRAL_FEATURES_PATH
    )


    print("\n========================================")
    print("VIRAL STI MODEL LOADED SUCCESSFULLY")
    print("========================================")

    print("Model:")
    print(viral_model)

    print("\nFeatures:")
    print(viral_features)

    print("\nNumber of features:")
    print(len(viral_features))

    print("\nModel classes:")
    print(viral_model.classes_)

    print("========================================\n")


except Exception as e:

    print("\n========================================")
    print("ERROR LOADING VIRAL STI MODEL")
    print("========================================")

    print(e)

    print("========================================\n")

    viral_model = None

    viral_features = []


# =========================================================
# LOAD HPV MODEL
# =========================================================

try:

    hpv_model = joblib.load(
        HPV_MODEL_PATH
    )

    hpv_features = joblib.load(
        HPV_FEATURES_PATH
    )

    hpv_imputer = joblib.load(
        HPV_IMPUTER_PATH
    )

    hpv_threshold = joblib.load(
        HPV_THRESHOLD_PATH
    )


    # -----------------------------------------------------
    # Convert threshold to numeric value
    # -----------------------------------------------------

    if isinstance(hpv_threshold, dict):

        possible_keys = [
            "threshold",
            "best_threshold",
            "optimal_threshold",
            "value"
        ]

        threshold_value = None

        for key in possible_keys:

            if key in hpv_threshold:

                threshold_value = hpv_threshold[key]

                break

        if threshold_value is None:

            threshold_value = list(
                hpv_threshold.values()
            )[0]

        hpv_threshold = float(
            threshold_value
        )

    else:

        hpv_threshold = float(
            np.asarray(
                hpv_threshold
            ).reshape(-1)[0]
        )


    print("\n========================================")
    print("HPV MODEL LOADED SUCCESSFULLY")
    print("========================================")

    print("Model:")
    print(hpv_model)

    print("\nHPV Features:")
    print(hpv_features)

    print("\nNumber of HPV features:")
    print(len(hpv_features))

    print("\nHPV Model classes:")
    print(hpv_model.classes_)

    print("\nHPV Threshold:")
    print(hpv_threshold)

    print("\nHPV Imputer:")
    print(hpv_imputer)

    print("========================================\n")


except Exception as e:

    print("\n========================================")
    print("ERROR LOADING HPV MODEL")
    print("========================================")

    print(e)

    print("========================================\n")

    hpv_model = None

    hpv_features = []

    hpv_imputer = None

    hpv_threshold = 0.5


# =========================================================
# USER FRIENDLY DISEASE NAMES
# =========================================================

DISPLAY_NAMES = {

    "AIDS": "HIV/AIDS",

    "HSV": "HSV",

    "Hepatitis B": "Hepatitis B",

    "HPV": "HPV"

}


# =========================================================
# NLP SYMPTOM KEYWORD MAPPING
# =========================================================
#
# Natural-language terms are mapped to the exact feature
# names used by the Viral STI Random Forest model.
#
# =========================================================

NLP_SYMPTOM_MAP = {

    "abdominal_pain": [
        "abdominal pain",
        "stomach pain",
        "belly pain",
        "pain in my abdomen"
    ],

    "dark_urine": [
        "dark urine",
        "brown urine"
    ],

    "fatigue": [
        "fatigue",
        "tired",
        "tiredness",
        "very tired",
        "extreme tiredness",
        "feeling tired"
    ],

    "high_fever": [
        "high fever",
        "very high fever",
        "fever"
    ],

    "itching": [
        "itching",
        "itchy",
        "itch"
    ],

    "lethargy": [
        "lethargy",
        "weakness",
        "feeling weak"
    ],

    "loss_of_appetite": [
        "loss of appetite",
        "no appetite",
        "poor appetite",
        "reduced appetite"
    ],

    "malaise": [
        "malaise",
        "feeling unwell",
        "feeling sick",
        "general discomfort"
    ],

    "muscle_wasting": [
        "muscle wasting",
        "muscle loss",
        "loss of muscle"
    ],

    "painful urination": [
        "painful urination",
        "pain when urinating",
        "pain while urinating",
        "painful urine",
        "burning urination",
        "burning when urinating",
        "burning while urinating"
    ],

    "patches_in_throat": [
        "patches in throat",
        "throat patches",
        "white patches in throat",
        "white patches on throat"
    ],

    "skin lesion": [
        "skin lesion",
        "skin lesions",
        "lesion on skin",
        "lesions on skin"
    ],

    "skin rash": [
        "skin rash",
        "rash",
        "skin rashes"
    ],

    "suprapubic pain": [
        "suprapubic pain",
        "lower abdominal pain",
        "pain above pubic area",
        "pain above the pubic area"
    ],

    "vaginal discharge": [
        "vaginal discharge",
        "discharge from vagina",
        "abnormal vaginal discharge"
    ],

    "vaginal itching": [
        "vaginal itching",
        "itchy vagina",
        "itching in vagina"
    ],

    "yellow_urine": [
        "yellow urine",
        "yellowish urine"
    ],

    "yellowing_of_eyes": [
        "yellowing of eyes",
        "yellow eyes",
        "yellow eyes",
        "eyes turning yellow"
    ],

    "yellowish_skin": [
        "yellowish skin",
        "yellow skin",
        "skin turning yellow"
    ]
}


# =========================================================
# NLP SYMPTOM EXTRACTION FUNCTION
# =========================================================

def extract_symptoms_from_text(text):

    if not text:

        return []


    # -----------------------------------------------------
    # Convert to lowercase
    # -----------------------------------------------------

    text_lower = text.lower().strip()


    # -----------------------------------------------------
    # Clean extra spaces
    # -----------------------------------------------------

    text_lower = re.sub(
        r"\s+",
        " ",
        text_lower
    )


    # -----------------------------------------------------
    # Run spaCy NLP
    # -----------------------------------------------------

    if nlp is not None:

        doc = nlp(
            text_lower
        )

        # Print tokens for development/testing
        print("\nNLP tokens:")

        for token in doc:

            print(
                token.text,
                end=" | "
            )

        print()


    # -----------------------------------------------------
    # Match known symptom phrases
    # -----------------------------------------------------

    extracted_symptoms = []


    for feature, keywords in NLP_SYMPTOM_MAP.items():

        for keyword in keywords:

            if keyword in text_lower:

                if feature not in extracted_symptoms:

                    extracted_symptoms.append(
                        feature
                    )

                break


    return extracted_symptoms


# =========================================================
# VIRAL STI PREDICTION HELPER
# =========================================================

def run_viral_prediction(selected_symptoms):

    if viral_model is None:

        raise Exception(
            "Viral STI model could not be loaded."
        )


    # -----------------------------------------------------
    # Create input dictionary
    # -----------------------------------------------------

    input_data = {}


    for feature in viral_features:

        if feature in selected_symptoms:

            input_data[feature] = 1

        else:

            input_data[feature] = 0


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # Ensure feature order
    # -----------------------------------------------------

    input_df = input_df[
        viral_features
    ]


    print("\nModel input:")

    print(input_df)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = viral_model.predict(
        input_df
    )[0]


    # -----------------------------------------------------
    # Prediction probabilities
    # -----------------------------------------------------

    probabilities = viral_model.predict_proba(
        input_df
    )[0]


    # -----------------------------------------------------
    # Create result dictionary
    # -----------------------------------------------------

    results = {}


    for disease, probability in zip(
        viral_model.classes_,
        probabilities
    ):

        results[disease] = round(
            float(probability) * 100,
            2
        )


    # -----------------------------------------------------
    # Sort highest first
    # -----------------------------------------------------

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # -----------------------------------------------------
    # Top prediction
    # -----------------------------------------------------

    top_disease = sorted_results[0][0]

    top_probability = sorted_results[0][1]


    # -----------------------------------------------------
    # Display names
    # -----------------------------------------------------

    display_results = []


    for disease, probability in sorted_results:

        display_name = DISPLAY_NAMES.get(
            disease,
            disease
        )

        display_results.append(
            (
                display_name,
                probability
            )
        )


    display_top_disease = DISPLAY_NAMES.get(
        top_disease,
        top_disease
    )


    return (
        prediction,
        display_results,
        display_top_disease,
        top_probability
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# =========================================================
# VIRAL STI SYMPTOM CHECKER
# =========================================================

@app.route("/symptom-checker")
def symptom_checker():

    return render_template(

        "symptom_checker.html",

        features=viral_features

    )


# =========================================================
# NLP VIRAL STI ASSESSMENT
# =========================================================

@app.route(
    "/nlp-predict",
    methods=["POST"]
)
def nlp_predict():

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if viral_model is None:

        return """
        <h2>Model Error</h2>
        <p>The Viral STI model could not be loaded.</p>
        """


    # -----------------------------------------------------
    # Get natural language input
    # -----------------------------------------------------

    symptom_text = request.form.get(
        "symptom_text",
        ""
    ).strip()


    print("\n========================================")
    print("NEW NLP STI ASSESSMENT")
    print("========================================")

    print("\nUser symptom description:")

    print(symptom_text)


    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not symptom_text:

        return """
        <h2>Input Required</h2>

        <p>
        Please describe your symptoms.
        </p>

        <a href="/symptom-checker">
        Back to Symptom Checker
        </a>
        """


    # -----------------------------------------------------
    # Extract symptoms using NLP
    # -----------------------------------------------------

    extracted_symptoms = extract_symptoms_from_text(
        symptom_text
    )


    print("\nNLP extracted symptoms:")


    if extracted_symptoms:

        for symptom in extracted_symptoms:

            print(
                "-",
                symptom
            )

    else:

        print(
            "No matching symptoms found."
        )


    # -----------------------------------------------------
    # No matching symptoms
    # -----------------------------------------------------

    if not extracted_symptoms:

        return render_template(

            "result.html",

            selected_symptoms=[],

            prediction="No matching symptoms",

            results=[],

            top_disease="No matching symptoms",

            top_probability=0,

            nlp_text=symptom_text,

            extracted_symptoms=[],

            nlp_mode=True

        )


    # -----------------------------------------------------
    # Run Random Forest prediction
    # -----------------------------------------------------

    try:

        (
            prediction,
            display_results,
            display_top_disease,
            top_probability

        ) = run_viral_prediction(
            extracted_symptoms
        )


    except Exception as e:

        print("\nNLP PREDICTION ERROR:")

        print(e)


        return f"""
        <h2>NLP Prediction Error</h2>

        <p>
        {e}
        </p>

        <a href="/symptom-checker">
        Back to Symptom Checker
        </a>
        """


    # -----------------------------------------------------
    # Terminal output
    # -----------------------------------------------------

    print("\n========================================")
    print("NLP VIRAL STI PREDICTION RESULTS")
    print("========================================")

    print(
        "Model prediction:",
        prediction
    )

    print(
        "Website prediction:",
        display_top_disease
    )

    print(
        "Top probability:",
        top_probability,
        "%"
    )

    print("\nAll probabilities:")


    for disease, probability in display_results:

        print(
            disease,
            ":",
            probability,
            "%"
        )


    print("========================================\n")


    # -----------------------------------------------------
    # SAVE VIRAL STI ASSESSMENT TO MONGODB
    # -----------------------------------------------------

    if assessments_collection is not None:

        try:

            assessment_data = {

                "assessment_type": "Viral STI",

                "extracted_symptoms": extracted_symptoms,

                "prediction": str(prediction),

                "top_disease": str(display_top_disease),

                "top_probability": float(top_probability),

                "probabilities": {
                    str(disease): float(probability)
                    for disease, probability in display_results
                }

            }

            assessments_collection.insert_one(
                assessment_data
            )

            print("MongoDB: Assessment saved successfully")

        except Exception as e:

            print("MongoDB save error:", e)

    else:

        print("MongoDB save skipped: database connection unavailable")


    # -----------------------------------------------------
    # Result page
    # -----------------------------------------------------

    return render_template(

        "result.html",

        selected_symptoms=extracted_symptoms,

        prediction=prediction,

        results=display_results,

        top_disease=display_top_disease,

        top_probability=top_probability,

        nlp_text=symptom_text,

        extracted_symptoms=extracted_symptoms,

        nlp_mode=True

    )


# =========================================================
# VIRAL STI CHECKBOX PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if viral_model is None:

        return """
        <h2>Model Error</h2>

        <p>
        The Viral STI model could not be loaded.
        </p>
        """


    # -----------------------------------------------------
    # Get selected symptoms
    # -----------------------------------------------------

    selected_symptoms = request.form.getlist(
        "symptoms"
    )


    print("\n========================================")
    print("NEW VIRAL STI PREDICTION")
    print("========================================")

    print("Selected symptoms:")


    for symptom in selected_symptoms:

        print(
            "-",
            symptom
        )


    # -----------------------------------------------------
    # Run model
    # -----------------------------------------------------

    try:

        (
            prediction,
            display_results,
            display_top_disease,
            top_probability

        ) = run_viral_prediction(
            selected_symptoms
        )


    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>

        <p>{e}</p>
        """


    # -----------------------------------------------------
    # Terminal output
    # -----------------------------------------------------

    print("\n========================================")
    print("VIRAL STI PREDICTION RESULTS")
    print("========================================")

    print(
        "Model prediction:",
        prediction
    )

    print(
        "Website prediction:",
        display_top_disease
    )

    print(
        "Top probability:",
        top_probability,
        "%"
    )

    print("\nAll probabilities:")


    for disease, probability in display_results:

        print(
            disease,
            ":",
            probability,
            "%"
        )


    print("========================================\n")


    # -----------------------------------------------------
    # SAVE CHECKBOX ASSESSMENT TO MONGODB
    # -----------------------------------------------------

    if assessments_collection is not None:

        try:

            assessment_data = {
                "assessment_type": "Checkbox",
                "selected_symptoms": selected_symptoms,
                "prediction": str(prediction),
                "top_disease": str(display_top_disease),
                "top_probability": float(top_probability),
                "probabilities": {
                    str(disease): float(probability)
                    for disease, probability in display_results
                }
            }

            assessments_collection.insert_one(assessment_data)
            print("MongoDB: Checkbox assessment saved successfully")

        except Exception as e:
            print("MongoDB save error:", e)


    # -----------------------------------------------------
    # Result page
    # -----------------------------------------------------

    return render_template(

        "result.html",

        selected_symptoms=selected_symptoms,

        prediction=prediction,

        results=display_results,

        top_disease=display_top_disease,

        top_probability=top_probability

    )


# =========================================================
# HPV ASSESSMENT PAGE
# =========================================================

@app.route("/hpv-assessment")
def hpv_assessment():

    if hpv_model is None:

        return """
        <h2>HPV Model Error</h2>

        <p>
        The HPV model could not be loaded.
        </p>
        """


    return render_template(

        "hpv_assessment.html",

        features=hpv_features

    )


# =========================================================
# HPV PREDICTION
# =========================================================

@app.route(
    "/predict-hpv",
    methods=["POST"]
)
def predict_hpv():

    # -----------------------------------------------------
    # Check HPV model
    # -----------------------------------------------------

    if hpv_model is None:

        return """
        <h2>HPV Model Error</h2>

        <p>
        The HPV model could not be loaded.
        </p>
        """


    if hpv_imputer is None:

        return """
        <h2>HPV Imputer Error</h2>

        <p>
        The HPV preprocessing imputer could not be loaded.
        </p>
        """


    # -----------------------------------------------------
    # Print request
    # -----------------------------------------------------

    print("\n========================================")
    print("NEW HPV PREDICTION")
    print("========================================")


    # -----------------------------------------------------
    # Read HPV form values
    # -----------------------------------------------------

    input_data = {}


    try:

        for feature in hpv_features:

            value = request.form.get(
                feature
            )


            if value is None or value.strip() == "":

                input_data[feature] = np.nan

            else:

                input_data[feature] = float(
                    value
                )


    except Exception as e:

        print(
            "\nERROR READING HPV FORM:"
        )

        print(e)


        return f"""
        <h2>Input Error</h2>

        <p>
        Unable to process the HPV assessment data.
        </p>

        <p>
        {e}
        </p>
        """


    # -----------------------------------------------------
    # Print received data
    # -----------------------------------------------------

    print("\nReceived HPV input:")


    for feature, value in input_data.items():

        print(
            feature,
            ":",
            value
        )


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # Ensure correct feature order
    # -----------------------------------------------------

    input_df = input_df[
        hpv_features
    ]


    print("\nOriginal HPV DataFrame:")

    print(input_df)


    # -----------------------------------------------------
    # Apply trained imputer
    # -----------------------------------------------------

    try:

        imputed_array = hpv_imputer.transform(
            input_df
        )

    except Exception as e:

        print(
            "\nERROR APPLYING HPV IMPUTER:"
        )

        print(e)


        return f"""
        <h2>Preprocessing Error</h2>

        <p>
        The HPV input could not be processed.
        </p>

        <p>
        {e}
        </p>
        """


    # -----------------------------------------------------
    # Convert imputed data to DataFrame
    # -----------------------------------------------------

    imputed_df = pd.DataFrame(

        imputed_array,

        columns=hpv_features

    )


    print("\nImputed HPV DataFrame:")

    print(imputed_df)


    # -----------------------------------------------------
    # Prediction probabilities
    # -----------------------------------------------------

    try:

        probabilities = hpv_model.predict_proba(
            imputed_df
        )[0]

    except Exception as e:

        print(
            "\nERROR DURING HPV PREDICTION:"
        )

        print(e)


        return f"""
        <h2>Prediction Error</h2>

        <p>
        The HPV model could not generate a prediction.
        </p>

        <p>
        {e}
        </p>
        """


    # -----------------------------------------------------
    # Find positive class
    # -----------------------------------------------------

    classes = list(
        hpv_model.classes_
    )


    positive_index = None


    for index, class_value in enumerate(classes):

        if str(class_value) == "1":

            positive_index = index

            break


    if positive_index is None:

        return """
        <h2>Model Configuration Error</h2>

        <p>
        The HPV model does not contain the expected
        positive class.
        </p>
        """


    # -----------------------------------------------------
    # Positive probability
    # -----------------------------------------------------

    positive_probability = float(
        probabilities[positive_index]
    )


    negative_index = None


    for index, class_value in enumerate(classes):

        if str(class_value) == "0":

            negative_index = index

            break


    if negative_index is None:

        return """
        <h2>Model Configuration Error</h2>

        <p>
        The HPV model does not contain the expected
        negative class.
        </p>
        """


    # -----------------------------------------------------
    # Negative probability
    # -----------------------------------------------------

    negative_probability = float(
        probabilities[negative_index]
    )


    # -----------------------------------------------------
    # Apply saved threshold
    # -----------------------------------------------------

    if positive_probability >= hpv_threshold:

        prediction = 1

        top_probability = positive_probability

    else:

        prediction = 0

        top_probability = negative_probability


    # -----------------------------------------------------
    # Convert probability to percentage
    # -----------------------------------------------------

    top_probability = round(
        top_probability * 100,
        2
    )


    # -----------------------------------------------------
    # Create probability results
    # -----------------------------------------------------

    results = {}


    for class_value, probability in zip(
        classes,
        probabilities
    ):

        results[str(class_value)] = round(
            float(probability) * 100,
            2
        )


    # -----------------------------------------------------
    # Sort results
    # -----------------------------------------------------

    sorted_results = sorted(

        results.items(),

        key=lambda x: x[1],

        reverse=True

    )


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n========================================")
    print("HPV PREDICTION RESULTS")
    print("========================================")

    print(
        "Positive probability:",
        round(
            positive_probability * 100,
            2
        ),
        "%"
    )

    print(
        "Negative probability:",
        round(
            negative_probability * 100,
            2
        ),
        "%"
    )

    print(
        "Classification threshold:",
        hpv_threshold
    )

    print(
        "Final prediction:",
        prediction
    )

    print(
        "Final probability:",
        top_probability,
        "%"
    )

    print("\nAll probabilities:")


    for label, probability in sorted_results:

        print(
            label,
            ":",
            probability,
            "%"
        )


    print("========================================\n")


    # -----------------------------------------------------
    # SAVE HPV ASSESSMENT TO MONGODB
    # -----------------------------------------------------

    if assessments_collection is not None:

        try:

            hpv_document = {
                "assessment_type": "HPV",
                "features": {
                    str(feature): None if pd.isna(value) else float(value)
                    for feature, value in input_data.items()
                },
                "prediction": int(prediction),
                "top_probability": float(top_probability),
                "probabilities": {
                    str(label): float(probability)
                    for label, probability in sorted_results
                },
                "threshold": float(hpv_threshold)
            }

            assessments_collection.insert_one(hpv_document)
            print("MongoDB: HPV assessment saved successfully")

        except Exception as e:
            print("MongoDB save error:", e)


    # -----------------------------------------------------
    # Result page
    # -----------------------------------------------------

    return render_template(

        "hpv_result.html",

        selected_features=input_data,

        prediction=prediction,

        top_probability=top_probability,

        results=sorted_results

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )