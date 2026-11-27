import mlflow
import mlflow.sklearn
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve



from prepare_data import X_train, X_val, X_test, y_train, y_val, y_test

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Customer_Churn_Optuna")


def objective(trial):
    with mlflow.start_run(nested=True, run_name="trial_" + str(trial.number)) as child_run:

        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        foret = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       class_weight=class_weight, random_state=42)

        foret.fit(X_train, y_train)

        y_pred = foret.predict(X_val)
        y_proba = foret.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)

        parametres = {"n_estimators": n_estimators, "max_depth": max_depth,
                      "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                      "max_features": max_features, "class_weight": class_weight}
        mlflow.log_params(parametres)

        resultats = {"accuracy": acc, "precision": prec, "recall": rec,
                     "f1": f1, "auc": auc}
        mlflow.log_metrics(resultats)

        mlflow.sklearn.log_model(sk_model=foret, input_example=X_train, name="modele_rf",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

        matrice = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = matrice)
        disp.plot()
        plt.savefig("matrice_confusion.png")
        plt.close()
        mlflow.log_artifact("matrice_confusion.png")

        fpr, tpr, seuils = roc_curve(y_val, y_proba)
        plt.plot([0,1],[0,1],'--')
        plt.plot(fpr,tpr)
        plt.xlabel('Taux Faux positif')
        plt.ylabel('Taux Vrai positif')
        plt.title('Courbe ROC foret aleatoire')
        plt.savefig("courbe_roc.png")
        plt.close()
        mlflow.log_artifact("courbe_roc.png")

    return f1


with mlflow.start_run(run_name="study_RandomForest") as run:
    study = optuna.create_study(direction="maximize", study_name="study_RandomForest")
    study.optimize(objective, n_trials=30)

    print(study.best_trial.number)
    print(study.best_value)
    print(study.best_params)
    
    
    
    
    
    
    
    
    