import numpy as np
from nilearn import datasets

# Parâmetros

n_pacientes_geral = 500
tipo_analise = 'ccs'
qualidade = True
regressao_global = True


# Carregando o atlas

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
labels = atlas.labels  # lista de regiões cerebrais
regioes = labels[1:]

# Carregando os dados
x_a_lista = np.load(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True)
x_c_lista = np.load(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy', allow_pickle=True)

dicio_series_a = {}
for i in range(len(x_a_lista)):
    dicio_series_a[(x_a_lista[i][0], x_a_lista[i][1])] = x_a_lista[i][2]

dicio_series_c = {}
for i in range(len(x_c_lista)):
    dicio_series_c[(x_c_lista[i][0], x_c_lista[i][1])] = x_c_lista[i][2]

import numpy as np

def dicionario_para_array(dicio_series):
    # 1. Listar pacientes e regiões únicos e ordenados
    pacientes = sorted(set(chave[0] for chave in dicio_series))
    regioes = sorted(set(chave[1] for chave in dicio_series))

    # 2. Obter o tamanho esperado da série temporal
    n_tempos = None
    for v in dicio_series.values():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            n_tempos = v.shape[0]
            break
    if n_tempos is None:
        raise ValueError("Não há séries temporais válidas no dicionário.")

    # 3. Listas para armazenar os dados e pacientes válidos
    dados_validos = []
    pacientes_validos = []

    for paciente in pacientes:
        paciente_valido = True
        paciente_dados = []
        for regiao in regioes:
            chave = (paciente, regiao)
            if chave in dicio_series:
                dados = np.asarray(dicio_series[chave]).flatten()  # Garante 1D
                if dados.shape[0] != n_tempos:
                    paciente_valido = False
                    break
                paciente_dados.append(dados)
            else:
                paciente_valido = False
                break
        if paciente_valido:
            dados_validos.append(paciente_dados)
            pacientes_validos.append(paciente)
        else:
            print(f"Aviso: paciente '{paciente}' ignorado por shape incorreto ou região ausente.")

    # 4. Inicializar o array
    x = np.zeros((len(pacientes_validos), len(regioes), n_tempos), dtype=np.float32)

    # 5. Preencher o array
    for i, paciente_dados in enumerate(dados_validos):
        for j, dados in enumerate(paciente_dados):
            x[i, j, :] = dados

    return x



# Listas onde armazenaremos os dados [Patient ID, Região, Série Temporal]
dados_autistas = []
dados_controle = []

# Aplicar a conversão
X_autistas = dicionario_para_array(dicio_series_a)
X_controle = dicionario_para_array(dicio_series_c)

# Salvar arrays com shape reorganizado
np.save(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy', X_autistas)
np.save(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}_tensor.npy', X_controle)
print("Arquivos Tensores salvos com sucesso!")