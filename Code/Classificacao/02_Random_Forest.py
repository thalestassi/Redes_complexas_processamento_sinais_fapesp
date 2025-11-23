import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

metricas_autistas_p = np.load("Metricas/Metricas_autistas_pearson.npy")
metricas_controle_p = np.load("Metricas/Metricas_controle_pearson.npy")


# 0 = Autism
# 1 = Control

metricas_autistas_p_pre = np.zeros(
    (np.shape(metricas_autistas_p)[1], np.shape(metricas_autistas_p)[0] * np.shape(metricas_autistas_p)[-1] + 1))
for paciente in range(np.shape(metricas_autistas_p)[1]):
    for metrica in range(np.shape(metricas_autistas_p)[0]):
        for onda in range(np.shape(metricas_autistas_p)[-1]):
            metricas_autistas_p_pre[paciente, metrica * np.shape(metricas_autistas_p)[-1] + onda] = metricas_autistas_p[
                metrica, paciente, onda]

metricas_controle_p_pre = np.ones(
    (np.shape(metricas_controle_p)[1], np.shape(metricas_controle_p)[0] * np.shape(metricas_controle_p)[-1] + 1))
for paciente in range(np.shape(metricas_controle_p)[1]):
    for metrica in range(np.shape(metricas_controle_p)[0]):
        for onda in range(np.shape(metricas_controle_p)[-1]):
            metricas_controle_p_pre[paciente, metrica * np.shape(metricas_controle_p)[-1] + onda] = metricas_controle_p[
                metrica, paciente, onda]

metricas_controle_p_pre = np.delete(metricas_controle_p_pre, 17, axis=0)
metricas_controle_todos_p = np.vstack([metricas_autistas_p_pre, metricas_controle_p_pre])

# lista de métricas
metricas = [
    "Grau_Medio",
    "Variancia",
    "Mutual_Information",
    "Kurtosis",
    "Entropia",
    "Agrupamento_Medio",
    "Menor_Caminho_Medio",
]
ondas = ["delta", "theta", "alpha", "beta", "gamma"]
colunas = [f"{metrica}_{onda}" for metrica in metricas for onda in ondas]
colunas.append("Condicao")
metricas_tabela = pd.DataFrame(metricas_controle_todos_p, columns=colunas)

x = metricas_tabela[[x for x in colunas if x != "Condicao"]]
y = metricas_tabela['Condicao']

kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=40)

pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(random_state=40))
])
hiperparametros_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_leaf': [1, 3, 5, 7],
    'model__min_samples_split': [2, 5, 10],
    'model__max_features': ['sqrt', 'log2'],
    'model__class_weight': [None, 'balanced']}
grid_rf = GridSearchCV(estimator=pipeline_rf,
                       param_grid=hiperparametros_rf,
                       cv=kf,
                       scoring='accuracy',
                       n_jobs=-1)
grid_rf.fit(x, y)

# print("GridSearchCV para Random Forest concluído.")
# print(f"Melhor Acurácia (Cross-Validation): {grid_rf.best_score_:.4f}")
# print(f"Melhores Hiperparâmetros: {grid_rf.best_params_}")

model_rf = grid_rf.best_estimator_

cv_roc = StratifiedKFold(n_splits=4, shuffle=True, random_state=23)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10, 7))


for i, (train_index, test_index) in enumerate(cv_roc.split(x, y)):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model_rf.fit(X_train, y_train)
    y_probs = model_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, lw=1, alpha=0.3,
            label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)


ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
        label='Chance level (AUC = 0.5)')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color='blue',
        label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
        lw=2.5, alpha=0.8)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       xlabel="False Positive Rate",
       ylabel="True Positive Rate",
       title="Random Forest")

ax.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6, color='purple')
plt.show()
fig.savefig('ROC_rf.png', dpi=150)
