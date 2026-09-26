import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_FILE = "data/features.csv"
MODEL_DIR = "model"
SPLIT_YEAR = 2024
SEED = 42

mlflow.set_experiment("f1-winner-prediction")
NUMERIC = ["grid","constructor_strength"]
CATEGORICAL = ["circuit","constructor"]

def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_FILE)
    df["grid"] = df["grid"].astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL, dtype=float)

    return df

def temporal_split(df: pd.DataFrame):
    train = df[df["season"]<= SPLIT_YEAR -1]
    test = df[df["season"]>= SPLIT_YEAR]
    drop = ["season","round","driver_id","won"]
    X_train = train.drop(columns=drop)
    y_train = train["won"] 
    X_test = test.drop(columns=drop)
    y_test = test["won"]    
    return train,test,X_train, y_train, X_test, y_test


def build_models():
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(random_state=SEED, max_iter=1000)),    
    ])
    rf = RandomForestClassifier(n_estimators = 300,random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)

    return {"logistic_regression":lr, "random_forest":rf, "gradient_boosting":gb}

def race_top1_accuracy(test_df:pd.DataFrame, y_prob) -> float:
    """Per race: did the highest-probability poick actually win?"""
    tmp = test_df[["season","round","won"]].copy()
    tmp["prob"] = y_prob
    top = tmp.sort_values("prob", ascending=False).groupby(["season","round"]).head(1)
    return (top["won"]==1).mean()


def main() -> None:
    df = load_data()
    train,test,X_train, y_train, X_test, y_test = temporal_split(df)
    print(f"Train rows : {len(X_train)} |test rows: {len(X_test)}")
    print(f"Test races: {test.groupby(['season','round']).ngroups}")

    pole = test["grid"]==1
    print(f"\n Baseline 'always polesitter': driver-acc {pole.mean():.3f},"
          f"race-top1 {test[pole]['won'].mean():.3f}")
    print(f"Baseline 'random 1/20': driver-acc {y_test.mean():.3f}")
    results = {}

    for name, model in build_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, f"{name}_model")
            y_prob = model.predict_proba(X_test)[:,1]
            acc = accuracy_score(y_test, y_prob>=0.5)
            ll = log_loss(y_test, y_prob)
            top1 = race_top1_accuracy(test, y_prob)
            results[name] = {"accuracy":acc, "log_loss":ll, "race_top1_accuracy":top1}
            print(f"\n{name}: driver-acc {acc:.3f}, log-loss {ll:.3f}, race-top1 {top1:.3f}")
            mlflow.log_param("split_year", SPLIT_YEAR)
            mlflow.log_param("SEED",SEED)
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_metrics({"accuracy":acc, "log_loss":ll, "race_top1_accuracy":top1})
            mlflow.sklearn.log_model(model,artifact_path="model")
        
        
        best = min(results, key = lambda k:results[k]["log_loss"])
        print(f"\nChampion: {best}")
        build_models()[best].fit(X_train, y_train)
        joblib.dump(
            {"model":build_models()[best],f"feature_columns":
            list(X_train.columns)},
              f"{MODEL_DIR}/model.joblib",)
        with open(f"{MODEL_DIR}/metrics.json","w") as f:
            json.dump({"champion": best, "results": results}, f, indent=2)
        print("Saved model/model.joblib + model/metrics.json")

if __name__ == "__main__":
    main()

