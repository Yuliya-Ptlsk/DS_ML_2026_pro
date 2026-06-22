import optuna
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)


RANDOM_STATE = 42

def tune_train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_class,
    n_trials=50,
):
    study = tune_model(
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        n_trials=n_trials,
    )

    best_model = build_model(
        model_class,
        study.best_params,
    )

    best_model.fit(
        X_train,
        y_train,
    )

    metrics, conf_matrix = evaluate_model(
        model=best_model,
        model_name=model_class.__name__,
        X_test=X_test,
        y_test=y_test,
    )

    return (
        metrics,
        conf_matrix,
        study.best_params,
        best_model,
    )

def get_search_space(trial, model_class):

    if model_class == CatBoostClassifier:

        return {
            "iterations": trial.suggest_int(
                "iterations",
                200,
                1000,
            ),
            "depth": trial.suggest_int(
                "depth",
                4,
                10,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "l2_leaf_reg": trial.suggest_int(
                "l2_leaf_reg",
                1,
                20,
            ),
            "random_strength": trial.suggest_int(
                "random_strength",
                1,
                10,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0,
                10,
            ),
        }

    elif model_class == GradientBoostingClassifier:

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                2,
                8,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                50,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                30,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),
        }

    raise ValueError(
        f"Unsupported model: {model_class}"
    )

def build_model(model_class, params):

    if model_class == CatBoostClassifier:

        return CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            loss_function="Logloss",
            verbose=False,
            random_state=RANDOM_STATE,
        )

    elif model_class == GradientBoostingClassifier:

        return GradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unsupported model: {model_class}"
    )

def objective(
    trial,
    model_class,
    X_train,
    y_train,
):

    params = get_search_space(
        trial,
        model_class,
    )

    model = build_model(
        model_class,
        params,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    return scores.mean()

def tune_model(
    model_class,
    X_train,
    y_train,
    n_trials=50,
):

    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_class,
            X_train,
            y_train,
        ),
        n_trials=n_trials,
    )

    return study

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame([
        {
            "model": model_name,
            "accuracy": accuracy_score(
                y_test,
                y_pred,
            ),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                y_proba,
            ),
            "pr_auc": average_precision_score(
                y_test,
                y_proba,
            ),
        }
    ])

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    conf_matrix = pd.DataFrame([
        {
            "model": model_name,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    ])

    return metrics, conf_matrix
