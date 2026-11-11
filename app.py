
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rental_model.joblib"

CATEGORICAL_COLS = [
    "Area Locality",
    "City",
    "Point of Contact"
]


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RentGuard Assistant",
    page_icon="R",
    layout="centered"
)


# ==========================================
# 3. FEATURE ENGINEERING
# Must match train_model.py
# ==========================================

def build_features(df):

    out = pd.DataFrame(index=df.index)

    out["BHK"] = pd.to_numeric(
        df["BHK"], errors="coerce"
    )

    out["Rent"] = pd.to_numeric(
        df["Rent"], errors="coerce"
    )

    out["Size"] = pd.to_numeric(
        df["Size"], errors="coerce"
    )

    out["Rent_per_sqft"] = (
        out["Rent"] /
        out["Size"].replace(0, np.nan)
    )

    out["Log_rent"] = np.log1p(
        out["Rent"].clip(lower=0)
    )

    for col in CATEGORICAL_COLS:

        out[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )

    return out


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_bundle():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model file not found. "
            "Please execute train_model.py first."
        )

    return joblib.load(MODEL_PATH)


try:

    bundle = load_bundle()
    pipeline = bundle["pipeline"]

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.info(
        "Ensure that train_model.py has been executed "
        "and rental_model.joblib exists in the application folder."
    )

    st.stop()


# ==========================================
# 5. PROFESSIONAL UI STYLING
# CSS ONLY - NO CUSTOM HTML CONTENT
# ==========================================

st.markdown("""
<style>

/* Application Background */

.stApp {
    background: linear-gradient(
        180deg,
        #edf4fc 0%,
        #ffffff 75%
    );
}

/* Main Content Width */

.block-container {
    max-width: 780px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header Styling */

h1 {
    color: #123b67 !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
}

h2, h3 {
    color: #183b60 !important;
    font-weight: 650 !important;
}

/* Paragraphs */

p, label {
    color: #344b63;
}

/* Input Fields */

div[data-baseweb="input"] input {
    border-radius: 8px;
}

div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Primary Buttons */

div[data-testid="stFormSubmitButton"] button {
    background: #1769aa;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: 650;
    transition: background 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #123b67;
    color: white;
}

/* Metric Containers */

div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 15px;
    border: 1px solid #dce7f3;
    border-radius: 10px;
}

/* Progress Bars */

div[data-testid="stProgressBar"] > div {
    background-color: #1769aa;
}

/* Dividers */

hr {
    border-color: #dce7f3;
}

/* Captions */

.stCaption {
    color: #61748a;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 6. APPLICATION HEADER
# NATIVE STREAMLIT COMPONENTS
# ==========================================

st.title("RentGuard Assistant")

st.markdown(
    "**Rental Listing Analysis and Fraud Detection System**"
)

st.caption(
    "System Status: Online | Machine Learning Model: Loaded"
)

st.divider()


# ==========================================
# 7. INTRODUCTION
# ==========================================

st.subheader("Welcome")

st.write(
    """
    RentGuard is a Machine Learning-based application
    designed to analyze rental property listings.

    Enter the property details below to receive an estimate
    of whether the listing appears genuine or suspicious.
    """
)

st.divider()


# ==========================================
# 8. RENTAL PROPERTY FORM
# ==========================================

st.subheader("Rental Property Details")

st.caption(
    "Enter the following information to analyze a rental listing."
)

with st.form("listing_form"):

    # Property Information

    st.markdown("**Property Information**")

    c1, c2, c3 = st.columns(3)

    with c1:

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

    with c2:

        rent = st.number_input(
            "Monthly Rent (INR)",
            min_value=0,
            max_value=10000000,
            value=15000,
            step=500
        )

    with c3:

        size = st.number_input(
            "Property Size (sq ft)",
            min_value=0,
            max_value=100000,
            value=800,
            step=50
        )

    st.divider()

    # Location Information

    st.markdown("**Location Information**")

    c4, c5 = st.columns(2)

    with c4:

        city = st.text_input(
            "City",
            placeholder="Enter city name"
        )

    with c5:

        locality = st.text_input(
            "Area / Locality",
            placeholder="Enter locality name"
        )

    # Contact Information

    st.markdown("**Contact Information**")

    contacts = bundle.get("contacts", [])

    contact_options = sorted(
        set(contacts) |
        {"contact owner", "contact agent"}
    )

    contact = st.selectbox(
        "Point of Contact",
        options=contact_options,
        format_func=lambda x: str(x).title()
    )

    st.write("")

    submitted = st.form_submit_button(
        "Analyze Rental Listing",
        use_container_width=True
    )


# ==========================================
# 9. PREDICTION AND ANALYSIS
# ==========================================

if submitted:

    # --------------------------------------
    # STEP 1: INPUT VALIDATION
    # --------------------------------------

    if not city.strip():

        st.warning("Please enter the city name.")

    elif not locality.strip():

        st.warning("Please enter the area or locality.")

    elif rent <= 0:

        st.warning(
            "Monthly rent must be greater than zero."
        )

    elif size <= 0:

        st.warning(
            "Property size must be greater than zero."
        )

    else:

        # ----------------------------------
        # STEP 2: PREPARE INPUT DATA
        # ----------------------------------

        raw = pd.DataFrame([{

            "BHK": bhk,
            "Rent": rent,
            "Size": size,
            "Area Locality": locality,
            "City": city,
            "Point of Contact": contact

        }])

        features = build_features(raw)

        # ----------------------------------
        # STEP 3: GENERATE PREDICTION
        # ----------------------------------

        try:

            probabilities = pipeline.predict_proba(
                features
            )[0]

            classes = list(pipeline.classes_)

            class_indices = {
                int(label): index
                for index, label in enumerate(classes)
            }

            if 0 not in class_indices or 1 not in class_indices:
                raise ValueError(
                    "The model must contain class labels 0 (fake) "
                    "and 1 (genuine)."
                )

            proba_real = float(
                probabilities[class_indices[1]]
            )

            proba_fake = float(
                probabilities[class_indices[0]]
            )

        except Exception as error:

            st.error(
                f"Prediction could not be generated: {error}"
            )

            st.stop()

        # ----------------------------------
        # STEP 4: CHECK TRAINING DATA
        # ----------------------------------

        known_cities = {
            str(x).strip().lower()
            for x in bundle.get("cities", [])
        }

        known_localities = {
            str(x).strip().lower()
            for x in bundle.get("localities", [])
        }

        known_contacts = {
            str(x).strip().lower()
            for x in bundle.get("contacts", [])
        }

        city_clean = city.strip().lower()

        locality_clean = locality.strip().lower()

        contact_clean = contact.strip().lower()

        unknown_city = (
            city_clean not in known_cities
        )

        unknown_locality = (
            locality_clean not in known_localities
        )

        unknown_contact = (
            contact_clean not in known_contacts
        )

        # ----------------------------------
        # STEP 5: NUMERICAL RANGE VALIDATION
        # ----------------------------------

        ranges = bundle["ranges"]

        outside_rent = not (
            ranges["Rent"][0]
            <= rent
            <= ranges["Rent"][1]
        )

        outside_size = not (
            ranges["Size"][0]
            <= size
            <= ranges["Size"][1]
        )

        outside_bhk = not (
            ranges["BHK"][0]
            <= bhk
            <= ranges["BHK"][1]
        )

        # ----------------------------------
        # STEP 6: CLASSIFICATION DECISION
        # ----------------------------------

        requires_review = any([

            unknown_city,
            unknown_locality,
            unknown_contact,
            outside_rent,
            outside_size,
            outside_bhk

        ])

        if requires_review:

            result = "Needs Manual Verification"

            result_type = "review"

        elif proba_real >= 0.75:

            result = "Likely Genuine"

            result_type = "genuine"

        elif proba_real <= 0.25:

            result = "Suspicious Listing"

            result_type = "suspicious"

        else:

            result = "Needs Manual Verification"

            result_type = "review"

        # ----------------------------------
        # STEP 7: DISPLAY SUBMITTED LISTING
        # ----------------------------------

        st.divider()

        st.subheader("Submitted Listing")

        st.write(
            f"**Property:** {bhk} BHK"
        )

        st.write(
            f"**Property Size:** {size:,} sq ft"
        )

        st.write(
            f"**Monthly Rent:** INR {rent:,}"
        )

        st.write(
            f"**Location:** {locality.strip().title()}, "
            f"{city.strip().title()}"
        )

        st.write(
            f"**Point of Contact:** {contact.title()}"
        )

        # ----------------------------------
        # STEP 8: DISPLAY PREDICTION RESULT
        # ----------------------------------

        st.divider()

        st.subheader("Analysis Result")

        if result_type == "genuine":

            st.success(
                f"Classification: {result}"
            )

        elif result_type == "suspicious":

            st.error(
                f"Classification: {result}"
            )

        else:

            st.warning(
                f"Classification: {result}"
            )

        st.markdown("**Model Prediction Scores**")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Genuine Listing Score",
                f"{proba_real * 100:.1f}%"
            )

            st.progress(proba_real)

        with col2:

            st.metric(
                "Fake Listing Score",
                f"{proba_fake * 100:.1f}%"
            )

            st.progress(proba_fake)

        st.caption(
            "These scores represent the model's estimated class "
            "probabilities. They do not independently establish "
            "whether a property is genuine or fraudulent."
        )

        # ----------------------------------
        # STEP 9: RENTAL STATISTICS
        # ----------------------------------

        st.divider()

        st.subheader("Rental Statistics")

        rent_per_sqft = rent / size

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Monthly Rent",
                f"INR {rent:,}"
            )

        with m2:

            st.metric(
                "Property Size",
                f"{size:,} sq ft"
            )

        with m3:

            st.metric(
                "Rent per Sq Ft",
                f"INR {rent_per_sqft:.2f}"
            )

        st.caption(
            "Rent per square foot is a calculated listing metric. "
            "It is not a verified local market-price comparison."
        )

        # ----------------------------------
        # STEP 10: RELIABILITY WARNINGS
        # ----------------------------------

        st.divider()

        st.subheader("Reliability Assessment")

        warnings = []

        if unknown_city:

            warnings.append(
                f"City '{city.strip()}' was not present "
                "in the training dataset."
            )

        if unknown_locality:

            warnings.append(
                f"Locality '{locality.strip()}' was not present "
                "in the training dataset."
            )

        if unknown_contact:

            warnings.append(
                "The selected contact category was not present "
                "in the training dataset."
            )

        if outside_rent:

            warnings.append(
                "Monthly rent is outside the training data range."
            )

        if outside_size:

            warnings.append(
                "Property size is outside the training data range."
            )

        if outside_bhk:

            warnings.append(
                "BHK value is outside the training data range."
            )

        if warnings:

            st.warning(
                "Additional verification is recommended."
            )

            for warning in warnings:

                st.write(f"- {warning}")

        else:

            st.success(
                "All entered categories and numerical values "
                "fall within the training data ranges."
            )

        # ----------------------------------
        # STEP 11: SAFETY DISCLAIMER
        # ----------------------------------

        st.divider()

        st.subheader("Important Disclaimer")

        st.info(
            """
            RentGuard provides a statistical estimate generated
            by a Machine Learning model.

            The application does not independently verify
            property ownership, landlord identity, property
            existence or payment authenticity.

            Always visit the property, verify ownership
            documents, review the rental agreement and
            independently confirm the landlord's identity.

            Avoid transferring advance payments before
            completing these checks.
            """
        )


# ==========================================
# 10. APPLICATION FOOTER
# ==========================================

st.divider()

st.caption(
    "RentGuard Assistant | Machine Learning-Based "
    "Rental Listing Analysis"
)

st.caption(
    f"Training Records: "
    f"{bundle.get('training_records', 'Unknown')}"
)
