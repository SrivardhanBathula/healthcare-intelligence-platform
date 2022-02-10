import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from lifelines import CoxPHFitter
import mlflow
import mlflow.xgboost
import logging

logger = logging.getLogger(__name__)


class ICURiskModel:
    """ICU readmission risk scoring using XGBoost + Cox survival analysis."""

    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
            eval_metric="aucpr", use_label_encoder=False, tree_method="hist"
        )
        self.cox_model = CoxPHFitter(penalizer=0.1)
        self.fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            survival_df: pd.DataFrame = None):
        with mlflow.start_run(run_name="icu_risk_model_v2"):
            self.xgb_model.fit(X_train, y_train,
                              eval_set=[(X_train, y_train)],
                              verbose=50, early_stopping_rounds=30)
            if survival_df is not None and "duration" in survival_df.columns:
                self.cox_model.fit(survival_df, duration_col="duration",
                                  event_col="readmission_30d")
                mlflow.log_param("cox_penalizer", 0.1)
            preds = self.xgb_model.predict(X_train)
            probs = self.xgb_model.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, probs)
            f1 = f1_score(y_train, preds)
            mlflow.log_metrics({"train_auc": auc, "train_f1": f1})
            mlflow.xgboost.log_model(self.xgb_model, "icu_model",
                                    registered_model_name="icu_risk_predictor")
            logger.info(f"ICU model trained: AUC={auc:.4f}, F1={f1:.4f}")
            self.fitted = True
        return self

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.xgb_model.predict_proba(X)[:, 1]
        risk_levels = pd.cut(probs, bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        return pd.DataFrame({"risk_score": probs, "risk_level": risk_levels})
