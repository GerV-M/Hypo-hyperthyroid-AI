import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import kagglehub
import warnings

warnings.filterwarnings("ignore")

# Чтение файлов, обработка датасетов
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE_DIR, "thyroidDF.csv")):
    kagglehub.dataset_download(
        "emmanuelfwerr/thyroid-disease-data",
        path="thyroidDF.csv",
        output_dir=BASE_DIR
    )

if not os.path.exists(os.path.join(BASE_DIR, "hypothyroid.csv")):
    kagglehub.dataset_download(
        "abdelazizsami/thyroid-gland-dataset",
        path="hypothyroid.csv",
        output_dir=BASE_DIR
    )

df1 = pd.read_csv(os.path.join(BASE_DIR, 'thyroidDF.csv'))
df2 = pd.read_csv(os.path.join(BASE_DIR, 'hypothyroid.csv'))

df2.replace('?', np.nan, inplace=True)

df1 = df1.drop(columns=['referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG', 'patient_id'])
df2 = df2.drop(columns=['referral source', 'query on thyroxine', 'query hypothyroid', 'query hyperthyroid', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured', 'TBG'])

df2 = df2.rename(columns={'binaryClass':'target', 'on thyroxine' : 'on_thyroxine', 'on antithyroid medication' : 'on_antithyroid_meds', 'thyroid surgery' : 'thyroid_surgery', 'I131 treatment' : 'I131_treatment'})

target_map = {'-': 0, 'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 2, 'F': 2, 'G': 2, 'H': 2, 'P': 2}
df1 = df1[df1['target'].isin(target_map.keys())].copy()
df1['target'] = df1['target'].map(target_map)

df2 = df2[df2['target'].isin(target_map.keys())].copy()
df2['target'] = df2['target'].map(target_map)

numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
categorical_cols = [col for col in df1.columns if col not in numeric_cols + ['target']]

for col in numeric_cols:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

# Объединение датасетов, очистка от выбросов в возрасте и от повторных значений
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['age'] < 120.0]
df = df.drop_duplicates(subset=['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])

# Оценка выбросов с помощью графика
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numeric_cols])
plt.show()

# Оценка выбросов текстовых строк
for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())

# Распределение гормонов по болезни
hormones = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'target']
sns.set_style("whitegrid")
pair = sns.pairplot(df[hormones], hue='target', plot_kws={'alpha': 0.5})

plt.show()

# Количество пациентов с той или иной целевой переменной
plt.figure(figsize=(6,4))
sns.countplot(x = df['target'])
plt.title('Распределение классов')
plt.xlabel('Класс')
plt.ylabel('Количество пациентов')

plt.show()

# Гистограмма
df[numeric_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle('Распределение числовых признаков')

plt.show()

sns.boxplot(data=df[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']], showfliers=False)
plt.title('Распределение числовых признаков без отображения выбросов')
plt.show()

# Подготовка данных, стандартизация
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

# Заполнение пустых значений
mode = X_train[categorical_cols].mode().iloc[0]
X_train[categorical_cols] = X_train[categorical_cols].fillna(mode)
X_test[categorical_cols] = X_test[categorical_cols].fillna(mode)

imputer = KNNImputer(n_neighbors=5, weights='distance')

X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# Бинаризация
binary_cols = X_train.select_dtypes(include='object').columns

for col in binary_cols:
    if col != 'target':
        X_train[col] = X_train[col].map({'t':1, 'f':0, 'F':1, 'M':0})
        X_test[col] = X_test[col].map({'t':1, 'f':0, 'F':1, 'M':0})

# Скэйлер
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Гендер в числа для матриц корреляции
df['sex'] = df['sex'].map({'F':1, 'M':0})

#Матрица корреляции гипертиреоза
df_corr1 = df[df['target'].isin([0, 1])].select_dtypes(include=['number'])
plt.figure(figsize = (20, 5))
plt.title('Матрица корреляции для гипертиреоза')
sns.heatmap(df_corr1.corr(), annot = True)

plt.show()

#Матрица корреляции гипотиреоза
df_corr2 = df[df['target'].isin([0, 2])].select_dtypes(include=['number'])
plt.figure(figsize = (20, 5))
plt.title('Матрица корреляции для гипотиреоза')
sns.heatmap(df_corr2.corr(), annot = True)

plt.show()

import time
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
results = []

# 1. Логистическая регрессия
print("Модель логистической регрессии")

start_time = time.perf_counter()

log_model = LogisticRegression(random_state=42, class_weight='balanced')
log_params = {'penalty' : ['l1', 'l2'], 'C' : [0.1, 1], 'solver' : ['lbfgs', 'liblinear', 'saga'], 'max_iter' : [100, 500]}

log_grid = GridSearchCV(log_model, log_params, cv=5, scoring='f1_macro', n_jobs=-1)

log_grid.fit(X_train, y_train)
end_time = time.perf_counter()
log_best = log_grid.best_estimator_
log_pred = log_best.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")

print("Лучшие параметры: ", log_grid.best_params_)

print("Accuracy: ", accuracy_score(y_test, log_pred))
print("Precision", precision_score(y_test, log_pred, average='macro'))
print("Recall", recall_score(y_test, log_pred, average='macro'))
print("F1-score: ", f1_score(y_test, log_pred, average='macro'))

y_proba = log_best.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
print("ROC-AUC: ", roc_auc)

print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

results.append({
    "Model": "Логистическая регрессия",
    "Accuracy": accuracy_score(y_test, log_pred),
    "Precision": precision_score(y_test, log_pred, average='macro'),
    "Recall": recall_score(y_test, log_pred, average='macro'),
    "F1": f1_score(y_test, log_pred, average='macro'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
})


# 2. Метод опорных векторов
print("Модель опорных векторов")

start_time = time.perf_counter()

svm_model = SVC(random_state=42, probability=True, class_weight='balanced')
svm_params = {'C' : [1, 10], 'kernel' : ['linear', 'rbf', 'sigmoid'], 'gamma' : ['scale', 'auto']}

svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring='f1_macro', n_jobs=-1)

svm_grid.fit(X_train, y_train)
end_time = time.perf_counter()
svm_best = svm_grid.best_estimator_
svm_pred = svm_best.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")

print("Лучшие параметры: ", svm_grid.best_params_)

print("Accuracy: ", accuracy_score(y_test, svm_pred))
print("Precision", precision_score(y_test, svm_pred, average='macro'))
print("Recall", recall_score(y_test, svm_pred, average='macro'))
print("F1-score: ", f1_score(y_test, svm_pred, average='macro'))

y_proba = svm_best.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
print("ROC-AUC: ", roc_auc)

print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

results.append({
    "Model": "Опорные вектора",
    "Accuracy": accuracy_score(y_test, svm_pred),
    "Precision": precision_score(y_test, svm_pred, average='macro'),
    "Recall": recall_score(y_test, svm_pred, average='macro'),
    "F1": f1_score(y_test, svm_pred, average='macro'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
})


# 3. Случайный лес
print("Модель случайного леса")

start_time = time.perf_counter()

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_params = {'n_estimators': [100, 200, 300], 'max_features': ['sqrt', 'log2'], 'max_depth': [5, 10, 15]}

rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='f1_macro', n_jobs=-1)

rf_grid.fit(X_train, y_train)
end_time = time.perf_counter()
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")
print("Лучшие параметры: ", rf_grid.best_params_)

print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("Precision: ", precision_score(y_test, rf_pred, average='macro'))
print("Recall: ", recall_score(y_test, rf_pred, average='macro'))
print("F1-score: ", f1_score(y_test, rf_pred, average='macro'))

y_proba = rf_best.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

print("ROC-AUC: ", roc_auc)

print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

results.append({
    "Model": "Случайный лес",
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred, average='macro'),
    "Recall": recall_score(y_test, rf_pred, average='macro'),
    "F1": f1_score(y_test, rf_pred, average='macro'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
})


# 4. XGBoost
print("Модель XGBoost")

start_time = time.perf_counter()

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb_model = XGBClassifier(random_state=42)
xgb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}

xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1_macro', n_jobs=-1 )

xgb_grid.fit(X_train, y_train, sample_weight=sample_weights)
end_time = time.perf_counter()
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")
print("Лучшие параметры: ", xgb_grid.best_params_)

print("Accuracy: ", accuracy_score(y_test, xgb_pred))
print("Precision: ", precision_score(y_test, xgb_pred, average='macro'))
print("Recall: ", recall_score(y_test, xgb_pred, average='macro'))
print("F1-score: ", f1_score(y_test, xgb_pred, average='macro'))

y_proba = xgb_best.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

print("ROC-AUC: ", roc_auc)

print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

results.append({
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "Precision": precision_score(y_test, xgb_pred, average='macro'),
    "Recall": recall_score(y_test, xgb_pred, average='macro'),
    "F1": f1_score(y_test, xgb_pred, average='macro'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
})

# Наглядный график для сравнения
results_df = pd.DataFrame(results)
plot_df = results_df.set_index('Model').T
plot_df.plot(kind='bar', figsize=(10, 6))

plt.title("Сравнение моделей")
plt.ylabel("Значение метрики")
plt.xlabel("Метрика")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(title='Модель')
plt.tight_layout()
plt.show()

import joblib

joblib.dump(rf_best, "RFMDump.joblib")

preprocessing = {"imputer": imputer, "scaler": scaler, "categorical_cols": categorical_cols, "numeric_cols": numeric_cols, "binary_cols": binary_cols, "mode_values": mode}

joblib.dump(preprocessing, "RFMPreprocessing.joblib")

from sklearn.inspection import permutation_importance

result = permutation_importance(rf_best, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({"Feature": X_test.columns, "Importance": result.importances_mean}).sort_values("Importance", ascending=False)

print(importance_df)

plt.figure(figsize=(8,6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Permutation Importance")
plt.savefig("importance.png") 
plt.show()

import shap

explainer = shap.TreeExplainer(rf_best)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

sample_idx = 0

class_id = rf_best.predict(X_test.iloc[[sample_idx]])[0]
