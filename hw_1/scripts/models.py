import pandas as pd
from sklearn.metrics import  (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# # ==========================================================
# # FEATURES
# # ==========================================================
#
# num_features = [
#     'Administrative',
#     'Administrative_Duration',
#     'Informational',
#     'Informational_Duration',
#     'ProductRelated',
#     'ProductRelated_Duration',
#     'BounceRates',
#     'ExitRates',
#     'PageValues',
#     'SpecialDay'
# ]
# cat_features = [
#     "Month",
#     "VisitorType",
#     "Weekend",
#     "Browser",
#     "OperatingSystems",
#     "Region",
#     "TrafficType"
# ]
#
# target = "Revenue"
#
# # ==========================================================
# # LOAD DATA
# # ==========================================================
#
# df = load_data_from_db()
#
# X = df.drop(target, axis=1)
# y = df[target]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # ==========================================================
# # PREPROCESSORS
# # ==========================================================
#
# # for models sensitive for scale
# preprocessor_scaled = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
#     ]
# )
#
# # for tree and boost models
# preprocessor_tree_boost = ColumnTransformer(
#     transformers=[
#         ("num", "passthrough" , num_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
#     ]
# )
#
# # ==========================================================
# # PREPARE DATA
# # ==========================================================
# X_train_scaled = preprocessor_scaled.fit_transform(X_train)
# X_test_scaled = preprocessor_scaled.transform(X_test)
#
# X_train_tree_boost = preprocessor_tree_boost.fit_transform(X_train)
# X_test_tree_boost = preprocessor_tree_boost.transform(X_test)
def train_models(*args):
    X_train_scaled, X_test_scaled, X_train_tree_boost, X_test_tree_boost, y_train, y_test  = args
    # ==========================================================
    # MODELS REQUIRING SCALING
    # ==========================================================

    linear_distance_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "KNeighbors": KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
            metric="euclidean",
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    }

    # ==========================================================
    # TREE AND BOOST MODELS
    # ==========================================================

    tree_boost_models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.5,   # negative class / positive class: 0.8453 / 0.1547
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }

    # ==========================================================
    # METRICS STORAGE
    # ==========================================================
    metrics_rows = []
    conf_matrix_rows = []

    # ==========================================================
    # HELPERS
    # ==========================================================
    def save_metrics(model_name, y_pred, y_proba, metrics_data ):
        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        metrics_data.append(row)

    def save_conf_matrix(model_name, y_pred, matrix_data ):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        row = {
            "model": model_name,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        matrix_data.append(row)

    # ==========================================================
    # TRAIN SCALED MODELS
    # ==========================================================
    for name, model in linear_distance_models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]

        save_metrics(name, y_pred, y_proba, metrics_rows)
        save_conf_matrix(name, y_pred, conf_matrix_rows)

    # ==========================================================
    # TRAIN TREE AND BOOST MODELS
    # ==========================================================
    for name, model in tree_boost_models.items():
        model.fit(X_train_tree_boost, y_train)

        y_pred = model.predict(X_test_tree_boost)
        y_proba = model.predict_proba(X_test_tree_boost)[:,1]

        save_metrics(name, y_pred, y_proba, metrics_rows)
        save_conf_matrix(name, y_pred, conf_matrix_rows)

    # ==========================================================
    # RESULTS
    # ==========================================================
    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values(
        by="pr_auc",
        ascending=False,
    )

    conf_matrix = pd.DataFrame(conf_matrix_rows)

    return (
        metrics,
        conf_matrix,
    )