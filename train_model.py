from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "conveyor_shift_dataset_v1_v2.csv"
df = pd.read_csv(DATA)
df = df.drop_duplicates().dropna().copy()
df["motor_load_ratio"] = df.motor_current_A / df.rated_motor_current_A.clip(lower=0.01)
df["belt_wear_pressure_index"] = (df.belt_age_months / 60) * (1 + df.dust_accumulation_index / 100)
features = ["motor_current_A","rated_motor_current_A","vibration_mm_s","belt_age_months",
"parcels_per_minute","dust_accumulation_index","days_since_maintenance","humidity_percent",
"motor_load_ratio","belt_wear_pressure_index"]
X, y = df[features], df.next_shift_jam.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models = {
"Logistic Regression": Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1500,class_weight="balanced"))]),
"Random Forest": RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=2,class_weight="balanced",random_state=42)
}
results=[]
for name, model in models.items():
    model.fit(X_train,y_train); pred=model.predict(X_test)
    results.append({"model":name,"accuracy":accuracy_score(y_test,pred),
    "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),
    "confusion_matrix":confusion_matrix(y_test,pred).tolist()})
results.sort(key=lambda r:(r["recall"],r["f1"]),reverse=True)
chosen=models[results[0]["model"]]
(ROOT/"models").mkdir(exist_ok=True)
joblib.dump(chosen,ROOT/"models"/"best_model.pkl")
(ROOT/"models"/"model_metadata.json").write_text(json.dumps({"selected_model":results[0]["model"],"features":features},indent=2))
(ROOT/"reports").mkdir(exist_ok=True)
(ROOT/"reports"/"model_metrics.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
