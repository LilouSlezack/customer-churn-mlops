import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


from prepare_data import X_train, X_val, X_test, y_train, y_val, y_test


mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Customer_Churn_Optuna")


X_train_val = np.concatenate((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))
print(X_train_val.shape)


foret_finale = RandomForestClassifier(n_estimators=242, max_depth=17, min_samples_split=11,
                                      min_samples_leaf=3, max_features='sqrt',
                                      class_weight='balanced', random_state=42)
foret_finale.fit(X_train_val, y_train_val)


y_pred = foret_finale.predict(X_test)
y_proba = foret_finale.predict_proba(X_test)[:, 1]


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(classification_report(y_test, y_pred))
print(auc)


with mlflow.start_run(run_name="modele_final_RandomForest") as run:
    parametres = {"n_estimators": 242, "max_depth": 17, "min_samples_split": 11,
                  "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"}
    mlflow.log_params(parametres)

    resultats = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    mlflow.log_metrics(resultats)

    infos_modele = mlflow.sklearn.log_model(sk_model=foret_finale, input_example=X_train_val,
                                            name="modele_rf_final",
                                            skops_trusted_types=["sklearn.tree._tree.Tree"])

    matrice = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = matrice)
    disp.plot()
    plt.savefig("matrice_confusion_test.png")
    plt.close()
    mlflow.log_artifact("matrice_confusion_test.png")


mlflow.register_model(model_uri=infos_modele.model_uri, name="churn_random_forest")




