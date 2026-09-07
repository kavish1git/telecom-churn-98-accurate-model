# Deploying Telecom Churn AI to Vercel

This directory has been pre-configured for seamless serverless deployment on **Vercel** with:
- `api/index.py`: Serverless Python FastAPI backend serving predictions from the **99.12% Accuracy Decision Tree** model.
- `public/index.html`: Responsive, modern Tailwind CSS web interface with interactive 4-pillar input controls, real-time risk gauges, retention cards, and CSV batch scoring.
- `vercel.json`: Pre-configured serverless routes and static assets.
- `.vercelignore`: Automatically excludes large raw datasets to ensure instant build times.

---

## 🚀 Method 1: Deploy via GitHub & Vercel Dashboard (Recommended — No CLI Needed)

1. **Upload or Push to GitHub**:
   - Create a new repository on [GitHub](https://github.com/new) named `telecom-churn-prediction`.
   - Upload the project files from `C:\Users\Kavish\.gemini\antigravity\scratch\telecom_churn_prediction` (excluding the `data/` folder).

2. **Import to Vercel**:
   - Go to [vercel.com/new](https://vercel.com/new) and log in.
   - Under **"Import Git Repository"**, select `telecom-churn-prediction`.
   - Framework Preset: Leave as **Other** (Vercel automatically detects `vercel.json`).
   - Root Directory: Leave as `./`.

3. **Click "Deploy"**:
   - Vercel automatically builds the Python serverless environment using `api/requirements.txt` and publishes your site worldwide on a free `.vercel.app` domain (e.g. `https://telecom-churn-prediction.vercel.app`).

---

## 💻 Method 2: Deploy via Vercel CLI

If you have Node.js / npm installed on your machine:

1. **Install Vercel CLI**:
   ```powershell
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```powershell
   vercel login
   ```

3. **Deploy from Project Directory**:
   ```powershell
   cd C:\Users\Kavish\.gemini\antigravity\scratch\telecom_churn_prediction
   vercel --prod
   ```
   Follow the brief interactive prompts (press Enter to accept defaults). Vercel will output your live production URL!

---

## 📡 Testing the Serverless API Endpoints

Once deployed, your Vercel URL exposes both the web app and serverless API:

- **Frontend UI**: `https://<your-project>.vercel.app/`
- **Health Check**: `https://<your-project>.vercel.app/api/health`
- **Model Metrics**: `https://<your-project>.vercel.app/api/metrics`
- **Real-Time Prediction**: `POST https://<your-project>.vercel.app/api/predict`
  ```json
  {
    "telecom_partner": "Airtel",
    "gender": "F",
    "age": 34,
    "city": "Bangalore",
    "num_dependents": 1,
    "estimated_salary": 85000,
    "calls_made": 65,
    "sms_sent": 25,
    "data_used_gb": 12.5,
    "tenure_months": 8.0,
    "contract_type": "Month-to-month",
    "payment_method": "UPI / Auto-Debit",
    "monthly_charges": 599.0,
    "customer_service_calls": 4,
    "tech_support_tickets": 2,
    "satisfaction_rating": 2,
    "unresolved_complaints": 1
  }
  ```
