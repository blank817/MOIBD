# Запуск: python predict_test.py
# Положите рядом: model_pipeline.joblib и test_real_estate_samples.csv
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIO = "Салманов Илья"        # впишите свои данные
GROUP = "23П-4"


def room_to_number(x):
    if pd.isna(x):
        return np.nan
    try:
        return sum(float(p) for p in str(x).split("+"))
    except ValueError:
        return np.nan


def range_to_number(x, marker, cap):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if marker in x:
        return cap
    if "-" in x:
        try:
            a, b = x.split("-")
            return (float(a) + float(b.split()[0])) / 2
        except ValueError:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


art = joblib.load("model_pipeline.joblib")
pipe = art["pipeline"]
cols = art["feature_columns"]
print("Модель:", art["best_model_name"], "| признаки:", cols)

ts = pd.read_csv("test_real_estate_samples.csv")
f = ts.copy()
f["listing_days"] = (pd.to_datetime(f["end_date"], errors="coerce")
                     - pd.to_datetime(f["start_date"], errors="coerce")).dt.days
f["size_log"] = np.log1p(f["size"].clip(lower=0))
f["room_count_num"] = f["room_count"].apply(room_to_number)
f["building_age_num"] = f["building_age"].apply(lambda x: range_to_number(x, "40 ve", 40))
f["total_floor_num"] = f["total_floor_count"].apply(lambda x: range_to_number(x, "20 ve", 20))

ts["predicted_price"] = pipe.predict(f[cols]).round(2)

y, p = ts["price"].to_numpy(float), ts["predicted_price"].to_numpy(float)
n = len(y)
manual = {
    "MAE": float(np.abs(y - p).sum() / n),
    "RMSE": float(((y - p) ** 2).sum() / n) ** 0.5,
    "R2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
}
lib = {
    "MAE": float(mean_absolute_error(y, p)),
    "RMSE": float(mean_squared_error(y, p) ** 0.5),
    "R2": float(r2_score(y, p)),
}
print("вручную:", manual)
print("sklearn:", lib)

ts.insert(0, "Группа", GROUP)
ts.insert(0, "ФИО", FIO)
ts.to_csv("result_real_estate_samples.csv", index=False, encoding="utf-8-sig")
with open("metrics.json", "w", encoding="utf-8") as fh:
    json.dump({"model": art["best_model_name"], "manual": manual, "sklearn": lib,
               "val": art.get("metrics_validation"), "test": art.get("metrics_test_best_model")},
              fh, ensure_ascii=False, indent=1, default=float)
print("Готово: result_real_estate_samples.csv и metrics.json")