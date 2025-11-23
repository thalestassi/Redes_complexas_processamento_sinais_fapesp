import numpy as np
from nilearn import datasets, input_data
import time

n_pacientes_geral = 50
tipo_analise = 'ccs'
qualidade = True
regressao_global = True


inicio = time.time()
abide = datasets.fetch_abide_pcp(n_subjects=n_pacientes_geral,
                                 pipeline=tipo_analise,
                                 quality_checked=qualidade,
                                 global_signal_regression=regressao_global)
fim = time.time()
print(f'{fim - inicio} segundos')

# Extrair informações
func_files = abide.func_preproc
phenotypic = abide.phenotypic
subject_ids = phenotypic['SUB_ID']
dx_groups = phenotypic['DX_GROUP']  # 1 = Autista, 2 = Controle
idades = phenotypic['AGE_AT_SCAN']

indices_filtrados = [i for i, idade in enumerate(idades) if 7 <= idade <= 10]

# Usar .iloc para acessar por posição, já que i é posição (enumerate)
autistas_ids = [i for i in indices_filtrados if dx_groups.iloc[i] == 1]
controle_ids = [i for i in indices_filtrados if dx_groups.iloc[i] == 2]


print(f'Número autistas: {len(autistas_ids)}')
print(f'Número controles: {len(controle_ids)}')

# Carregar atlas Harvard-Oxford Cortical
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps
labels = atlas.labels

# Criar masker para extrair os sinais
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

# Listas onde armazenaremos os dados [Patient ID, Região, Série Temporal]
dados_autistas = []
dados_controle = []


# Função para armazenar dados
def extrair_dados(indices, destino):
    for i in indices:
        patient_id = f"Patient_{subject_ids.iloc[i]}"
        func_file = func_files[i]

        # Extrair séries temporais das regiões
        time_series = masker.fit_transform(func_file)  # shape: (n_timepoints, n_regions)

        # Adicionar cada região como uma entrada
        for region_idx, region_label in enumerate(labels[1:]):  # labels[0] = background
            region_ts = time_series[:, region_idx]  # Série temporal dessa região
            destino.append([patient_id, region_label, region_ts])
        fim = time.time()
        print(f'{fim - inicio} segundos')


# Extrair dados
extrair_dados(autistas_ids, dados_autistas)
extrair_dados(controle_ids, dados_controle)

# Salvar arrays brutos
np.save(f'Matrizes_Adjacencia_Abide/x_a_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy',
        np.array(dados_autistas, dtype=object))
np.save(f'Matrizes_Adjacencia_Abide/x_c_abide_{n_pacientes_geral}_{tipo_analise}_{qualidade}_{regressao_global}.npy',
        np.array(dados_controle, dtype=object))
print("Arquivos brutos salvos com sucesso!")
