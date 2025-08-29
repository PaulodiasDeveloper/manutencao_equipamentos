import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Importando Seaborn para gráficos mais elaborados

# Configuração da página para um layout mais amplo e um título
st.set_page_config(page_title="Dashboard de Análise de Ocorrências", layout="wide")

# Define o estilo dos gráficos do Seaborn
sns.set_style("whitegrid")

# Título principal do dashboard
st.title("📊 Análise Integrada de Ocorrências: Locais, Equipamentos e Paradas")
st.markdown("---")

# Componente para upload do arquivo Excel na barra lateral
uploaded_file = st.sidebar.file_uploader("Carregue a sua planilha de ocorrências (.xlsx)", type="xlsx")

# Verifica se um arquivo foi carregado
if uploaded_file is not None:
    # Leitura dos dados do arquivo Excel carregado
    try:
        dados = pd.read_excel(uploaded_file)
        # Remove linhas onde 'Tempo de Parada (h)' é nulo para evitar erros nos gráficos
        dados.dropna(subset=['Tempo de Parada (h)'], inplace=True)


        st.sidebar.success("Planilha carregada com sucesso!")

        # Opção para exibir os dados brutos
        if st.sidebar.checkbox("Mostrar dados brutos"):
            st.subheader("Visualização dos Dados Brutos")
            st.dataframe(dados)
            st.markdown("---")

        # Layout em duas colunas para os gráficos principais
        col1, col2 = st.columns(2)

        # --- ANÁLISE POR LOCAL ---
        with col1:
            st.header("Análise por Local")

            # Cria a tabela de frequência e percentual
            freq_local_table = pd.DataFrame({
                "Frequência": dados["Local"].value_counts(),
                "Percentual (%)": (dados["Local"].value_counts(normalize=True) * 100).round(2)
            })
            st.table(freq_local_table)

            # Gera o gráfico de ocorrências por local
            st.subheader("Gráfico de Ocorrências por Local")
            fig1, ax1 = plt.subplots()
            dados["Local"].value_counts().plot(kind="bar", ax=ax1, color='skyblue')
            ax1.set_title("Ocorrências por Local")
            ax1.set_xlabel("Local")
            ax1.set_ylabel("Frequência")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig1)

        # --- ANÁLISE POR EQUIPAMENTO ---
        with col2:
            st.header("Análise por Equipamento")

            # Cria a tabela de frequência e percentual por equipamento
            freq_equip_table = pd.DataFrame({
                "Frequência": dados["Equipamento"].value_counts(),
                "Percentual (%)": (dados["Equipamento"].value_counts(normalize=True) * 100).round(2)
            })
            st.table(freq_equip_table)

            # Gera o gráfico de ocorrências por equipamento
            st.subheader("Gráfico de Ocorrências por Equipamentos")
            fig2, ax2 = plt.subplots()
            dados["Equipamento"].value_counts().plot(kind="bar", ax=ax2, color='lightgreen')
            ax2.set_title("Ocorrências por Equipamentos")
            ax2.set_xlabel("Equipamento")
            ax2.set_ylabel("Frequência")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig2)

        st.markdown("---")

        # --- ANÁLISE DE HORAS DE PARADA ---
        st.header("Análise de Horas de Parada por Equipamento")

        # Agrupa os dados para somar as horas de parada por equipamento
        horas_equip = dados[['Tempo de Parada (h)', 'Equipamento']]
        resumo = horas_equip.groupby('Equipamento')['Tempo de Parada (h)'].sum().sort_values(ascending=False).reset_index()

        # Layout em duas colunas para a tabela e o gráfico de horas de parada
        col3, col4 = st.columns([1, 2]) # A segunda coluna é mais larga

        with col3:
             st.subheader("Tabela de Soma de Horas")
             st.table(resumo)

        with col4:
            # Gera o gráfico da soma das horas de parada
            st.subheader("Gráfico da Soma das Horas de Parada por Equipamento")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.bar(resumo['Equipamento'], resumo['Tempo de Parada (h)'], color='salmon')
            ax3.set_xlabel('Equipamento')
            ax3.set_ylabel('Soma das Horas de Parada')
            ax3.set_title('Soma das Horas de Parada por Equipamento')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3)

        st.markdown("---")

        # --- NOVA SEÇÃO: ANÁLISE DA DISTRIBUIÇÃO DO TEMPO DE PARADA ---
        st.header("Análise da Distribuição do Tempo de Parada")
        st.write("Esta seção ajuda a entender como os valores de tempo de parada estão distribuídos.")

        col5, col6 = st.columns(2)

        with col5:
            # --- HISTOGRAMA COM LINHA DE DENSIDADE (KDE) ---
            st.subheader("Histograma do Tempo de Parada")
            fig4, ax4 = plt.subplots()
            sns.histplot(dados['Tempo de Parada (h)'], kde=True, ax=ax4, color='darkcyan')
            ax4.set_title('Distribuição das Horas de Parada')
            ax4.set_xlabel('Tempo de Parada (h)')
            ax4.set_ylabel('Frequência')
            st.pyplot(fig4)
            st.caption("O histograma mostra a frequência de ocorrências para cada faixa de tempo de parada. A linha representa a estimativa de densidade da distribuição.")

        with col6:
            # --- BOXPLOT POR EQUIPAMENTO ---
            st.subheader("Boxplot do Tempo de Parada por Equipamento")
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Equipamento', y='Tempo de Parada (h)', data=dados, ax=ax5, palette='viridis')
            ax5.set_title('Distribuição do Tempo de Parada por Equipamento')
            ax5.set_xlabel('Equipamento')
            ax5.set_ylabel('Tempo de Parada (h)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig5)
            st.caption("O boxplot mostra a mediana (linha central), os quartis (caixa) e os outliers (pontos) para cada equipamento, facilitando a comparação da dispersão e de valores atípicos.")


    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
        st.warning("Verifique se as colunas 'Local', 'Equipamento' e 'Tempo de Parada (h)' existem na sua planilha.")

else:
    # Mensagem exibida quando nenhum arquivo foi carregado ainda
    st.info("👈 Por favor, carregue uma planilha no menu à esquerda para iniciar a análise.")
