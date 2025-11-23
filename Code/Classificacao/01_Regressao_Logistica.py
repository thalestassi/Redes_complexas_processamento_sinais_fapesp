import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

# 0 = Autism
# 1 = Control

metricas_autistas_p = np.load("Metricas/Metricas_autistas_pearson.npy")
metricas_controle_p = np.load("Metricas/Metricas_controle_pearson.npy")

ondas = ["delta", "theta", "alpha", "beta", "gamma"]

metricas = [
    "Grau_Medio",
    "Variancia",
    "Mutual_Information",
    "Kurtosis",
    "Entropia",
    "Agrupamento_Medio",
    "Menor_Caminho_Medio",
]

metricas_autistas_p_pre = np.zeros((np.shape(metricas_autistas_p)[1], np.shape(metricas_autistas_p)[0] * np.shape(metricas_autistas_p)[-1] + 1))
for paciente in range(np.shape(metricas_autistas_p)[1]):
    for metrica in range(np.shape(metricas_autistas_p)[0]):
        for onda in range(np.shape(metricas_autistas_p)[-1]):
            metricas_autistas_p_pre[paciente, metrica * np.shape(metricas_autistas_p)[-1] + onda] = metricas_autistas_p[metrica, paciente, onda]

metricas_controle_p_pre = np.ones((np.shape(metricas_controle_p)[1], np.shape(metricas_controle_p)[0] * np.shape(metricas_controle_p)[-1] + 1))
for paciente in range(np.shape(metricas_controle_p)[1]):
    for metrica in range(np.shape(metricas_controle_p)[0]):
        for onda in range(np.shape(metricas_controle_p)[-1]):
            metricas_controle_p_pre[paciente, metrica * np.shape(metricas_controle_p)[-1] + onda] = metricas_controle_p[metrica, paciente, onda]
metricas_controle_p_pre = np.delete(metricas_controle_p_pre, 17, axis=0)
metricas_controle_todos_p = np.vstack([metricas_autistas_p_pre, metricas_controle_p_pre])


colunas = [f"{metrica}_{onda}" for metrica in metricas for onda in ondas]
colunas.append("Condicao")
metricas_tabela = pd.DataFrame(metricas_controle_todos_p, columns=colunas)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
print(metricas_tabela)


# x = metricas_tabela[[x for x in colunas if x != "Condicao"]]
# y = metricas_tabela['Condicao']
#
# kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=40)
# fold_generator = kf.split(x, y)
#
# hiperparametro_C = {'model__C': np.logspace(-2, 2, 1000)}
# pipeline_l1 = Pipeline([('scaler', StandardScaler()),('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=40))])
#
# grid_l1 = GridSearchCV(estimator=pipeline_l1, param_grid=hiperparametro_C, cv=kf, scoring='accuracy', n_jobs=-1)
# grid_l1.fit(x, y)
# pipeline_l2 = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=40))])
#
# grid_l2 = GridSearchCV(estimator=pipeline_l2, param_grid=hiperparametro_C, cv=kf, scoring='accuracy', n_jobs=-1)
# grid_l2.fit(x, y)
#
# results_l1 = pd.DataFrame(grid_l1.cv_results_)
# results_l1 = results_l1.sort_values(by='param_model__C')
# c_values_l1 = results_l1['param_model__C'].astype(float)
# accuracy_l1 = results_l1['mean_test_score']
# best_model_l1 = grid_l1.best_estimator_.named_steps['model']
# coef_l1 = pd.Series(best_model_l1.coef_[0], index=x.columns)
#
# results_l2 = pd.DataFrame(grid_l2.cv_results_)
# results_l2 = results_l2.sort_values(by='param_model__C')
# c_values_l2 = results_l2['param_model__C'].astype(float)
# accuracy_l2 = results_l2['mean_test_score']
# best_model_l2 = grid_l2.best_estimator_.named_steps['model']
# coef_l2 = pd.Series(best_model_l2.coef_[0], index=x.columns)
#
#
# fig, ax = subplots(1, 2, figsize=(16, 6))
# ax[0].plot(c_values_l1, accuracy_l1, marker='o', linestyle='-')
# ax[0].set_xscale('log')
# ax[0].set_xlabel('Parâmetro de Regularização (C)')
# ax[0].set_ylabel('Acurácia Média (Cross-Validation)')
# ax[0].set_title('Lasso (L1)')
# ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
# best_c_l1 = grid_l1.best_params_['model__C']
# best_acc_l1 = grid_l1.best_score_
# ax[0].axvline(best_c_l1, color='red', linestyle='--', label=f'Melhor Acurácia Média: {best_acc_l1:.4f}\nParâmetro de Regularização: {best_c_l1:.1e}')
# ax[0].legend()
#
# ax[1].plot(c_values_l2, accuracy_l2, marker='s', linestyle='-', color='purple')
# ax[1].set_xscale('log')
# ax[1].set_xlabel('Parâmetro de Regularização (C)')
# ax[1].set_ylabel('Acurácia Média (Cross-Validation)')
# ax[1].set_title('Ridge (L2)')
# ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
# best_c_l2 = grid_l2.best_params_['model__C']
# best_acc_l2 = grid_l2.best_score_
# ax[1].axvline(best_c_l2, color='red', linestyle='--', label=f'Melhor Acurácia Média: {best_acc_l2:.4f}\nParâmetro de Regularização: {best_c_l2:.1e}')
# ax[1].legend()
# fig.tight_layout()
# fig.savefig('Acuracia_vs_C.png', dpi=150)
# plt.show()
#
# model = LogisticRegression(C=best_c_l2, penalty='l2', solver='lbfgs', max_iter=1000)
# cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=30)
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
#
# fig, ax = plt.subplots(figsize=(10, 7))
#
#
# scaler = StandardScaler()
# for i, (train_index, test_index) in enumerate(cv.split(x, y)):
#     X_train, X_test = x.iloc[train_index], x.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     scaler.fit(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     model.fit(X_train_scaled, y_train)
#     y_probs = model.predict_proba(X_test_scaled)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_test, y_probs)
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     ax.plot(fpr, tpr, lw=1, alpha=0.3,
#             label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
#     interp_tpr = np.interp(mean_fpr, fpr, tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#
# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
#         label='Chance level (AUC = 0.5)')
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# std_tpr = np.std(tprs, axis=0)
# mean_auc = np.mean(aucs)
# std_auc = np.std(aucs)
# ax.plot(mean_fpr, mean_tpr, color='blue',
#         label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
#         lw=2.5, alpha=0.8)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
#                 label=r'$\pm$ 1 std. dev.')
# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#        xlabel="False Positive Rate",
#        ylabel="True Positive Rate",
#        title="Regressão Logística")
#
# ax.legend(loc="lower right")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()
# fig.savefig('ROC_ridge.png', dpi=150)
#


# random_states_para_testar = [20, 42, 100, 555, 1984, 7, 99, 123, 777, 404]
# n_splits = 6
#
# acuracias_l1 = []
# params_c_l1 = []
# acuracias_l2 = []
# params_c_l2 = []
#
#
# for state in random_states_para_testar:
#     kf_iterativo = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)
#
#     grid_l1_iterativo = GridSearchCV(estimator=pipeline_l1,
#                                      param_grid=hiperparametro_C,
#                                      cv=kf_iterativo,
#                                      scoring='accuracy',
#                                      n_jobs=-1)
#     grid_l1_iterativo.fit(x, y)
#
#     acuracias_l1.append(grid_l1_iterativo.best_score_)
#     params_c_l1.append(grid_l1_iterativo.best_params_['model__C'])
#
#     # --- Modelo L2 (Ridge) ---
#     grid_l2_iterativo = GridSearchCV(estimator=pipeline_l2,
#                                      param_grid=hiperparametro_C,
#                                      cv=kf_iterativo,  # Usa o kf iterativo
#                                      scoring='accuracy',
#                                      n_jobs=-1)
#     grid_l2_iterativo.fit(x, y)
#
#     acuracias_l2.append(grid_l2_iterativo.best_score_)
#     params_c_l2.append(grid_l2_iterativo.best_params_['model__C'])



# print(f"Testes realizados com {len(random_states_para_testar)} diferentes random_states e {n_splits} folds.")
#
# print("\n--- Modelo L1 (Lasso) ---")
# print(f"Melhores Acurácias encontradas: {[round(a, 4) for a in acuracias_l1]}")
# print(f"Acurácia Média Robusta: {np.mean(acuracias_l1):.4f}")
# print(f"Desvio Padrão da Acurácia: {np.std(acuracias_l1):.4f}")
# print(f"Melhores 'C' encontrados: {[round(c, 4) for c in params_c_l1]}")
# print(f"'C' mais frequente: {max(set(params_c_l1), key=params_c_l1.count):.4f}")
#
# print("\n--- Modelo L2 (Ridge) ---")
# print(f"Melhores Acurácias encontradas: {[round(a, 4) for a in acuracias_l2]}")
# print(f"Acurácia Média Robusta: {np.mean(acuracias_l2):.4f}")
# print(f"Desvio Padrão da Acurácia: {np.std(acuracias_l2):.4f}")
# print(f"Melhores 'C' encontrados: {[round(c, 4) for c in params_c_l2]}")
# print(f"'C' mais frequente: {max(set(params_c_l2), key=params_c_l2.count):.4f}")
