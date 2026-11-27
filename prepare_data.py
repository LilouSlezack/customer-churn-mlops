import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df=pd.read_csv("/Users/lilouslezack/Desktop/Gestion-Projet-IA/customer-churn-mlops/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")



print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df['Churn'].value_counts())

df['Churn'].value_counts().plot(kind='bar')
plt.title('Nombre de clients par classe')
plt.show()



#Nettoyage
df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum())


df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
print(df.isnull().sum())
print(df.dtypes)



#Encodage
df['Churn'] = LabelEncoder().fit_transform(df['Churn'])
print(df['Churn'].value_counts())

categorical_columns = df.select_dtypes(include='str').columns
print(categorical_columns)

df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
print(df.shape)
print(df.head())



#Découpage et normalisation
X = df.drop('Churn', axis=1)
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)









