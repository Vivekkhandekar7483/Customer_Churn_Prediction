
import os
import joblib
import sklearn
import numpy as np
import pandas as pd

# ---------------------------------
# 1. File paths
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "customer_churn_best_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "customer_churn_model_local.pkl"
)

# ---------------------------------
# 2. Display library versions
# ---------------------------------

print("Customer Churn Prediction")
print("-" * 40)

print("Scikit-learn:", sklearn.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

# ---------------------------------
# 3. Check existing model
# ---------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print("\nLoading already-trained model...")

# ---------------------------------
# 4. Load existing trained model
# ---------------------------------

try:
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

except Exception as error:
    print("\nModel loading failed.")
    print("Error:", error)

    print(
        "\nThis usually indicates a Scikit-learn "
        "version compatibility problem."
    )

    raise

# ---------------------------------
# 5. Save a local copy
# ---------------------------------

joblib.dump(model, OUTPUT_PATH)

print("\nModel saved successfully!")
print("Saved file:", OUTPUT_PATH)

# ---------------------------------
# 6. Verify saved model
# ---------------------------------

loaded_model = joblib.load(OUTPUT_PATH)

print("\nVerification completed!")
print("Model type:", type(loaded_model))
print("Existing trained model is ready to use.")
