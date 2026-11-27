# customer-churn-mlops

TP MLOps (gestion de projet IA) : L'objectif est de prédire si un client va quitter son operateur 

Dataset : Telco Customer Churn (sur kaggle)

- Lilou : partie random forest
- Jules : partie regression logistique

## Les fichiers

- data : fichier csv du dataset
- figures : graphique de l'analyse exploratoire (nombre de clients par classe de Churn)
- prepare_data.py : la preparation des donnees

Random Forest :
- optuna_random_forest.py : optimisation des hyperparametres de la random forest avec optuna, 30 trials, tout est enregistre dans mlflow
- modele_final.py : le modele final avec les meilleurs hyperparametres, teste sur le test et enregistre dans le registry

Regression logistique :


## Installer

conda create -n mlflow-env python=3.12
conda activate mlflow-env
pip install mlflow
pip install optuna

## Lancer le serveur mlflow

conda activate mlflow-env
mlflow ui

ca ouvre mlflow sur http://127.0.0.1:5000

dans les scripts on met :
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Customer_Churn_Optuna")

## Refaire le meilleur modele

1. lancer mlflow ui
2. lancer optuna_random_forest.py (ca fait le run parent study_RandomForest et les 30 runs trial_0 a trial_29)
3. lancer modele_final.py (ça re entraine le modele sur train + validation et ça enregistre dans le registry sous le nom churn_random_forest)

(attention il faut changer le chemin du csv dans prepare_data.py car c est le chemin de mon ordi)

Le meilleur trial c est le trial_24 avec :
- n_estimators = 242 
- max_depth = 17
- min_samples_split = 11
- min_samples_leaf = 3
- max_features = sqrt
- class_weight = balanced

Resultats sur le test : 
- accuracy = 0.76
- precision = 0.53
- recall = 0.77
- f1 = 0.63
- auc = 0.84
