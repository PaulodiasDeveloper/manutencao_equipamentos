# %%        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
dados = pd.read_excel("Planilha de ocorrências.xlsx")
dados.head(20)
# %%
'''Local com maior número de ocorrências é o Pátio de Valorização de Resíduos (57,89%) seguido de AGR Cabiúnas (31,58%).

Os locais TIMS e Área 400 Depósito 421 aparecem apenas uma vez cada (5,26% cada).'''

freq_local_table = pd.DataFrame({
    "Frequência": dados["Local"].value_counts(),
    "Percentual (%)": dados["Local"].value_counts(normalize=True) * 100
})
print(freq_local_table)
# %%
# Visualização rápida (opcional)

dados["Local"].value_counts().plot(kind="bar", title="Ocorrências por Local")
plt.xlabel("Local")
plt.ylabel("Frequência")
plt.show()

# %%
freq_equip_table = pd.DataFrame({
    "Frequência": dados["Identificação"].value_counts(),
    "Percentual (%)": dados["Identificação"].value_counts(normalize=True) * 100
})
print(freq_equip_table)

# %%
'''Equipamento mais envolvido é a Manipuladora, responsável por 47,37% das ocorrências. Outros equipamentos com frequência menor incluem Empilhadeiras (diversos tipos), Retroescavadeira, Carreta Dedicada, e Empilhadeira Elétrica (cada um com cerca de 5–10%).'''

freq_equip_table = pd.DataFrame({
    "Frequência": dados["Equipamento"].value_counts(),
    "Percentual (%)": dados["Equipamento"].value_counts(normalize=True) * 100
})
print(freq_equip_table)
# %%
# Visualização rápida (opcional)

dados["Equipamento"].value_counts().plot(kind="bar", title="Ocorrências por Equipamentos")
plt.xlabel("Equipamento")
plt.ylabel("Frequência")
plt.show()

# %%
horas_equip = dados[['Tempo de Parada (h)', 'Equipamento']]
horas_equip
# %%
resumo = horas_equip.groupby('Equipamento')['Tempo de Parada (h)'].sum().reset_index()
print(resumo)
# %%
import matplotlib.pyplot as plt

# 'resumo' já contém o agrupamento dos dados
plt.figure(figsize=(10,6))
plt.bar(resumo['Equipamento'], resumo['Tempo de Parada (h)'])
plt.xlabel('Equipamento')
plt.ylabel('Soma das Horas de Parada')
plt.title('Soma das Horas de Parada por Equipamento')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()