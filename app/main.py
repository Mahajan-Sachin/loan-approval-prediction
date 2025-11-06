 
# 🧠 Loan Approval Prediction Flask App (OOP + Production Ready)
# Author: Sachin Mahajan
# Description:
#   Flask + REST API that loads trained ML pipeline and predicts
#   loan approval, built for Railway deployment.
# ===============================================================

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os


# ===============================================================
# 🧩 CLASS DEFINITION
# ===============================================================
class LoanApprovalApp:
    """Encapsulated Flask App for Loan Prediction."""

    def __init__(self, model_path):
        """Initialize Flask app and load ML model."""
        self.app = Flask(__name__)
        self.model = self._load_model(model_path)
        self._add_routes()

    # ---------------------------
    # MODEL LOADING
    # ---------------------------
    def _load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Model file not found at {model_path}. Please ensure it exists."
            )
        except Exception as e:
            raise RuntimeError(f"⚠️ Error loading model: {str(e)}")

    # ---------------------------
    # ROUTES
    # ---------------------------
    def _add_routes(self):
        """Define Flask endpoints."""

        @self.app.route('/')
        def home():
            """Render main UI page."""
            try:
                return render_template('home.html')
            except Exception as e:
                return f"Error rendering home page: {str(e)}", 500

        @self.app.route('/predict_api', methods=['POST'])
        def predict_api():
            """Handle REST API JSON input."""
            try:
                data = request.get_json(force=True)
                df = pd.DataFrame([data])

                pred_proba = self.model.predict_proba(df)[0]
                prediction = self.model.predict(df)[0]
                confidence = round(pred_proba[int(prediction)] * 100, 2)
                result = "Approved" if prediction == 1 else "Rejected"

                return jsonify({
                    "status": "success",
                    "loan_status": result,
                    "confidence": f"{confidence}%",
                    "input_data": data
                })

            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Handle Web Form submission (home.html)."""
            try:
                # Extract and prepare form data
                form_data = {
                    'Gender': request.form['Gender'],
                    'Married': request.form['Married'],
                    'Dependents': request.form['Dependents'],
                    'Education': request.form['Education'],
                    'Self_Employed': request.form['Self_Employed'],
                    'ApplicantIncome': request.form['ApplicantIncome'],
                    'CoapplicantIncome': request.form['CoapplicantIncome'],
                    'LoanAmount': request.form['LoanAmount'],
                    'Loan_Amount_Term': request.form['Loan_Amount_Term'],
                    'Credit_History': request.form['Credit_History'],
                    'Property_Area': request.form['Property_Area']
                }

                df = pd.DataFrame([form_data])
                df = df.astype({
                    'ApplicantIncome': float,
                    'CoapplicantIncome': float,
                    'LoanAmount': float,
                    'Loan_Amount_Term': float,
                    'Credit_History': float
                })

                # Make prediction
                pred_proba = self.model.predict_proba(df)[0]
                prediction = self.model.predict(df)[0]
                confidence = round(pred_proba[int(prediction)] * 100, 2)
                result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

                return render_template(
                    'home.html',
                    prediction_text=result,
                    confidence=confidence,
                    form_data=form_data
                )

            except KeyError as e:
                return render_template('home.html', prediction_text=f"Missing input: {str(e)}")
            except ValueError as e:
                return render_template('home.html', prediction_text=f"Invalid input: {str(e)}")
            except Exception as e:
                return render_template('home.html', prediction_text=f"Unexpected error: {str(e)}")

    # ---------------------------
    # RUN SERVER (LOCAL)
    # ---------------------------
    def run(self, debug=True):
        """Run Flask server locally or on Railway."""
        try:
            port = int(os.environ.get("PORT", 8080))  # Railway sets PORT env
            self.app.run(host="0.0.0.0", port=port, debug=debug)
        except Exception as e:
            print(f"❌ Failed to start Flask server: {str(e)}")


# ===============================================================
# 🚀 ENTRY POINT (LOCAL + RAILWAY)
# ===============================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "../models/loan_approval_pipeline.joblib")

    loan_app = LoanApprovalApp(model_path)
    loan_app.run(debug=True)


# ===============================================================
# ✅ WSGI ENTRY FOR GUNICORN / RAILWAY
# ===============================================================
app = LoanApprovalApp(os.path.join(os.path.dirname(__file__), "../models/loan_approval_pipeline.joblib")).app