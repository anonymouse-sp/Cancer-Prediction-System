# OncoPredict: Cancer Risk & Action System

Simple web app for low-literacy users:
- Big, high-contrast UI with one-question-at-a-time flow
- FastAPI backend with GradientBoostingClassifier prediction endpoint
- Result page shows either doctor guidance (high risk) or healthy tips (low risk)

## Project Structure

- `backend/main.py` - FastAPI API and ML prediction logic
- `backend/requirements.txt` - Python dependencies
- `frontend/src/App.jsx` - Step-by-step React UI and result flow
- `frontend/src/index.css` - Large text and simple high-contrast styles

## Training With Fetched Datasets

The backend now supports automatic training from fetched CSV files.

- Place CSV files in project root, `backend/`, or `backend/data/`
- Start backend; it scans available CSVs and picks the best compatible dataset
- It maps common alias columns to the core model features:
	- age, gender, bmi, smoking, genetic risk, physical activity, alcohol intake, family history
- If some fields are missing, it imputes safely and continues training
- Check `GET /health` to confirm:
	- `training_source`
	- `samples`
	- `holdout_accuracy`
	- `feature_coverage` (how well fetched columns matched model features)

## Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend URL: `http://localhost:8000`

Note: in this workspace, backend is typically run on `http://localhost:8001` due to a local port conflict on 8000.

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: usually `http://localhost:5173`

## Important Disclaimer

THIS IS NOT A MEDICAL DIAGNOSIS. CONSULT A DOCTOR.
