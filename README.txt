
1. Install Python 3.10 or 3.11.
2. Place your trusted customer_churn_best_model.pkl beside app.py.
3. Open terminal in this folder.
4. Run: py -m venv .venv
5. Run: .venv\\Scripts\\activate
6. Run: pip install -r requirements.txt
7. Run: streamlit run app.py
8. Open the localhost URL printed in terminal (usually http://localhost:8501).

The app expects these ten inputs: Age, Gender, Tenure, Usage Frequency,
Support Calls, Payment Delay, Subscription Type, Contract Length,
Total Spend, Last Interaction. Make sure categories and package versions
match the training environment. Load only trusted pickle/joblib files.
