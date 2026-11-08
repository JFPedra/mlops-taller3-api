import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
X = pd.DataFrame({
    "sensor_temperatura": np.random.randn(1000) * 10 + 20,
    "sensor_presion": np.random.rand(1000) * 50 + 100,
})
X.loc[15:65, "sensor_temperatura"] = np.nan
y = ((X["sensor_temperatura"] + X["sensor_presion"]) > 130).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline_productivo = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=200, random_state=42)),
])
pipeline_productivo.fit(X_train, y_train)

print(classification_report(y_test, pipeline_productivo.predict(X_test)))
print("Pasos del pipeline:", [n for n, _ in pipeline_productivo.steps])
print("Hiperparámetros del modelo:", pipeline_productivo.named_steps["clasificador"].get_params())

joblib.dump(pipeline_productivo, "pipeline.joblib")
print("Guardado: pipeline.joblib")