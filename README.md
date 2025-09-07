# 🩺 MedAssist – Fitbit Data Prediction System

MedAssist is an AI pipeline that processes **Fitbit activity & health exports** (steps, sleep, calories, heart rate, etc.) and builds machine learning models to generate predictions for healthcare/fitness insights.

It is designed for **large Fitbit CSV exports** from `Fitabase` (like `mturkfitbit_export_4.12.16-5.12.16`) and can:
- Preprocess the raw data (normalize, encode, handle missing values)
- Train a deep learning regression model
- Save reusable artifacts (`.pkl`, `.h5`, `.yaml`, `.json`)
- Run memory-safe predictions on new Fitbit datasets
- Visualize accuracy graphs and heatmaps

---

## 📂 Project Structure

Med Assist/
│
├── archive/ # Raw Fitbit CSV exports
│ ├── mturkfitbit_export_4.12.16-5.12.16/
│ └── mturkfitbit_export_3.12.16-4.11.16/
│
├── medassist_preprocess.pkl # Preprocessing pipeline (scaler + encoder)
├── medassist_model.h5 # Trained Keras model
├── medassist_config.yaml # Model & preprocessing configuration
├── medassist_metadata.json # Metadata summary (features, shape, etc.)
│
├── medassist_predictions.csv # Prediction output (with new "prediction" column)
├── medassist_prediction_summary.json # Metrics summary (MAE, RMSE, R²)
│
├── medassist_accuracy.png # Training accuracy graph (MAE/Loss curves)
├── medassist_corr_heatmap.png # Correlation heatmap of numeric features
├── medassist_pred_vs_actual.png # Predicted vs Actual heatmap
│
├── train_medassist.py # Script to train model and generate artifacts
├── predict_medassist.py # Script to load PKL+H5 and run predictions
└── README.md

yaml
Copy code

---

## ⚙️ Installation

```bash
# Clone repo (if hosted on GitHub)
git clone https://github.com/yourname/medassist.git
cd medassist

# Install dependencies
pip install -r requirements.txt
requirements.txt

text
Copy code
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
joblib
🚀 Training
If you want to re-train the model on your Fitbit CSVs:

bash
Copy code
python train_medassist.py
This will:

Load Fitbit CSVs from the archive/ folders

Preprocess + merge them

Train a Keras regression model

Save artifacts:

medassist_preprocess.pkl

medassist_model.h5

medassist_config.yaml

medassist_metadata.json

Save history curves (medassist_accuracy.png)

🔮 Prediction
To generate predictions on new Fitbit datasets:

bash
Copy code
python predict_medassist.py
It will:

Load preprocess pipeline (.pkl) and model (.h5)

Force sparse OneHotEncoder → avoids RAM issues

Batch predictions (default: 10,000 rows, batch=1024)

Save results:

medassist_predictions.csv

medassist_prediction_summary.json

Show preview in console:

yaml
Copy code
=== Prediction Preview (first 10 rows) ===
 Calories  Steps  prediction
     2200   7534      2301.5
     1800   5230      1754.7
     1950   6000      1923.2
...
📊 Visualization
Accuracy Graph (Loss & MAE)

Shows training vs validation curves.

Correlation Heatmap

Highlights which Fitbit features are most correlated.

Predicted vs Actual Heatmap

2D density plot of actual vs predicted values.

🧪 Example Datasets
The project works with Kaggle Fitbit datasets too:
🔗 Kaggle Fitbit Fitness Tracker Data

📈 Metrics
From prediction summary (medassist_prediction_summary.json):

json
Copy code
{
  "rows_scored": 9850,
  "mae": 53.27,
  "mse": 4125.38,
  "rmse": 64.26,
  "r2": 0.82
}
✨ Key Features
✅ Handles huge Fitbit CSVs safely (sampling + batch prediction)

✅ Saves reusable model artifacts

✅ RAM-safe prediction with sparse OneHotEncoder

✅ Ready for expansion (add heart-rate prediction, sleep analysis, etc.)

✅ Visualization included (accuracy curves, heatmaps)

📌 Next Steps
Add deep sequence models (LSTMs) for time-series Fitbit data.

Build a Flask/FastAPI dashboard for interactive predictions.

Deploy on Streamlit for easy use by clinicians.

👨‍💻 Author
Built by Sagnik Patra (M.Tech CSE, IIIT)
