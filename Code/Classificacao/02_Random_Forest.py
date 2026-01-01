import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score

metricas_autistas_p = np.load("Metricas/Metricas_autistas_pearson.npy")
metricas_controle_p = np.load("Metricas/Metricas_controle_pearson.npy")

ondas = ["delta", "theta", "alpha", "beta", "gamma"]
metricas = ["Grau_Medio", "Variancia", "Mutual_Information", "Kurtosis", "Entropia", "Agrupamento_Medio",
            "Menor_Caminho_Medio"]

metricas_autistas_p_pre = np.ones(
    (np.shape(metricas_autistas_p)[1], np.shape(metricas_autistas_p)[0] * np.shape(metricas_autistas_p)[-1] + 1))
for paciente in range(np.shape(metricas_autistas_p)[1]):
    for metrica in range(np.shape(metricas_autistas_p)[0]):
        for onda in range(np.shape(metricas_autistas_p)[-1]):
            metricas_autistas_p_pre[paciente, metrica * np.shape(metricas_autistas_p)[-1] + onda] = metricas_autistas_p[
                metrica, paciente, onda]

metricas_controle_p_pre = np.zeros(
    (np.shape(metricas_controle_p)[1], np.shape(metricas_controle_p)[0] * np.shape(metricas_controle_p)[-1] + 1))
for paciente in range(np.shape(metricas_controle_p)[1]):
    for metrica in range(np.shape(metricas_controle_p)[0]):
        for onda in range(np.shape(metricas_controle_p)[-1]):
            metricas_controle_p_pre[paciente, metrica * np.shape(metricas_controle_p)[-1] + onda] = metricas_controle_p[
                metrica, paciente, onda]

metricas_controle_p_pre = np.delete(metricas_controle_p_pre, 1, axis=0)
metricas_controle_todos_p = np.vstack([metricas_autistas_p_pre, metricas_controle_p_pre])

colunas = [f"{metrica}_{onda}" for metrica in metricas for onda in ondas]
colunas.append("Condicao")
metricas_tabela = pd.DataFrame(metricas_controle_todos_p, columns=colunas)

x = metricas_tabela[[c for c in colunas if c != "Condicao"]]
y = metricas_tabela['Condicao']

tprs = []
aucs = []
acuracias = []
mean_fpr = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(10, 8))

pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(random_state=None))  # random_state varia a cada fit
])

hiperparametros_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_leaf': [1, 3, 5, 7],
    'model__min_samples_split': [2, 5, 10],
    'model__max_features': ['sqrt', 'log2'],
    'model__class_weight': [None, 'balanced']
}

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=i)

    grid_rf = GridSearchCV(estimator=pipeline_rf,
                           param_grid=hiperparametros_rf,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_model = grid_rf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_probs = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    acuracias.append(acc)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
mean_acc = np.mean(acuracias)
std_acc = np.std(acuracias)

ax.plot(mean_fpr, mean_tpr, color='green',
        label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
        lw=2.5)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                label=r'$\pm$ 1 std. dev.')
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
        label='Chance level (AUC = 0.5)')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       xlabel="False Positive Rate",
       ylabel="True Positive Rate",
       title=f"Random Forest")

ax.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
# plt.savefig('ROC_rf.png', dpi=150)
plt.show()
