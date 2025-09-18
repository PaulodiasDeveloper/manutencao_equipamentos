import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import openpyxl  # necessário para ler arquivos Excel

# Configuração da página
st.set_page_config(
    page_title="Análise de Paradas de Equipamentos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do dashboard
st.title("📊 Dashboard de Análise de Paradas de Equipamentos")

# Função para calcular dias e horas a partir de horas decimais
def calcular_dias_horas(horas):
    if pd.isna(horas):
        return ""
    dias = int(horas // 24)
    horas_restantes = horas % 24
    return f"{dias} dias e {horas_restantes:.2f} horas"

# Função para carregar dados do arquivo Excel
@st.cache_data
def load_data(uploaded_file):
    try:
        # Ler o arquivo Excel
        df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        
        # Converter para datetime
        df['Data Início'] = pd.to_datetime(df['Data Início'], errors='coerce')
        df['Data Fim'] = pd.to_datetime(df['Data Fim'], errors='coerce')
        
        # Calcular tempo de parada se não existir
        if 'Tempo de Parada (h)' not in df.columns:
            df['Tempo de Parada (h)'] = np.where(
                df['Data Fim'].notna(),
                (df['Data Fim'] - df['Data Início']).dt.total_seconds() / 3600,
                np.nan
            )
        
        # Calcular duração para registros abertos com data atual
        now = pd.Timestamp.now()
        df['Tempo Calculado (h)'] = np.where(
            df['Status'] == 'Aberto',
            (now - df['Data Início']).dt.total_seconds() / 3600,
            df['Tempo de Parada (h)']
        )
        
        # Criar coluna Parada_dias_horas se não existir
        if 'Parada_dias_horas' not in df.columns:
            df['Parada_dias_horas'] = df['Tempo Calculado (h)'].apply(calcular_dias_horas)
        
        # Extrair mês e ano
        df['Mês'] = df['Data Início'].dt.month
        df['Ano'] = df['Data Início'].dt.year
        df['Mês-Ano'] = df['Data Início'].dt.to_period('M').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        return None

# Upload do arquivo
st.sidebar.header("Carregar Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça upload da planilha de paradas (Excel)", 
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        # Sidebar com filtros
        st.sidebar.header("Filtros")
        locais = st.sidebar.multiselect(
            "Selecione os Locais:",
            options=df['Local'].unique(),
            default=df['Local'].unique()
        )

        status_options = st.sidebar.multiselect(
            "Selecione os Status:",
            options=df['Status'].unique(),
            default=df['Status'].unique()
        )

        equipamentos = st.sidebar.multiselect(
            "Selecione os Equipamentos:",
            options=df['Equipamento'].unique(),
            default=df['Equipamento'].unique()
        )

        # Aplicar filtros
        df_filtrado = df[
            (df['Local'].isin(locais)) &
            (df['Status'].isin(status_options)) &
            (df['Equipamento'].isin(equipamentos))
        ]

        # Métricas principais
        st.header("Métricas Principais")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_paradas = len(df_filtrado)
            st.metric("Total de Paradas", total_paradas)

        with col2:
            paradas_abertas = len(df_filtrado[df_filtrado['Status'] == 'Aberto'])
            st.metric("Paradas em Aberto", paradas_abertas)

        with col3:
            tempo_total_parada = df_filtrado['Tempo Calculado (h)'].sum()
            st.metric("Tempo Total de Parada (h)", f"{tempo_total_parada:.2f}")

        with col4:
            tempo_medio_parada = df_filtrado['Tempo Calculado (h)'].mean()
            st.metric("Tempo Médio de Parada (h)", f"{tempo_medio_parada:.2f}")

        # Gráficos
        st.header("Análise Visual dos Dados")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Visão Geral", 
            "Tendência Temporal", 
            "Análise de Equipamentos", 
            "Detalhes das Paradas"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    df_filtrado['Local'].value_counts().reset_index(),
                    x='Local',
                    y='count',
                    title="Número de Paradas por Local",
                    labels={'Local': 'Local', 'count': 'Número de Paradas'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                fig2 = px.pie(
                    df_filtrado,
                    names='Status',
                    title="Distribuição por Status"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            # Top 10 equipamentos com mais paradas
            fig3 = px.bar(
                df_filtrado['Equipamento'].value_counts().reset_index().head(10),
                x='Equipamento',
                y='count',
                title="Top 10 Equipamentos com Mais Paradas",
                labels={'Equipamento': 'Equipamento', 'count': 'Número de Paradas'}
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            # Agrupar por mês
            df_mensal = df_filtrado.groupby(['Ano', 'Mês', 'Mês-Ano']).agg({
                'Nº': 'count',
                'Tempo Calculado (h)': 'sum'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig4 = px.line(
                    df_mensal,
                    x='Mês-Ano',
                    y='Nº',
                    title="Evolução do Número de Paradas",
                    labels={'Nº': 'Número de Paradas', 'Mês-Ano': 'Mês'}
                )
                st.plotly_chart(fig4, use_container_width=True)
                
            with col2:
                fig5 = px.line(
                    df_mensal,
                    x='Mês-Ano',
                    y='Tempo Calculado (h)',
                    title="Evolução do Tempo Total de Parada (h)",
                    labels={'Tempo Calculado (h)': 'Tempo de Parada (h)', 'Mês-Ano': 'Mês'}
                )
                st.plotly_chart(fig5, use_container_width=True)
                
            # Distribuição por dia da semana
            df_filtrado['Dia da Semana'] = df_filtrado['Data Início'].dt.day_name()
            dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dias_portugues = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            
            df_dia_semana = df_filtrado['Dia da Semana'].value_counts().reindex(dias_ordem).reset_index()
            df_dia_semana['Dia'] = dias_portugues
            
            fig6 = px.bar(
                df_dia_semana,
                x='Dia',
                y='count',
                title="Distribuição de Paradas por Dia da Semana",
                labels={'Dia': 'Dia da Semana', 'count': 'Número de Paradas'}
            )
            st.plotly_chart(fig6, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig7 = px.bar(
                    df_filtrado['Equipamento'].value_counts().reset_index(),
                    x='Equipamento',
                    y='count',
                    title="Número de Paradas por Tipo de Equipamento",
                    labels={'Equipamento': 'Tipo de Equipamento', 'count': 'Número de Paradas'}
                )
                st.plotly_chart(fig7, use_container_width=True)
                
            with col2:
                # Tempo médio de parada por equipamento
                tempo_medio_equipamento = df_filtrado.groupby('Equipamento')['Tempo Calculado (h)'].mean().reset_index()
                fig8 = px.bar(
                    tempo_medio_equipamento,
                    x='Equipamento',
                    y='Tempo Calculado (h)',
                    title="Tempo Médio de Parada por Tipo de Equipamento (h)",
                    labels={'Equipamento': 'Tipo de Equipamento', 'Tempo Calculado (h)': 'Tempo Médio (h)'}
                )
                st.plotly_chart(fig8, use_container_width=True)
                
            # Análise de MTBF (Mean Time Between Failures) por equipamento
            st.subheader("Análise de Confiabilidade (MTBF)")
            
            # Ordenar por data de início
            df_ordenado = df_filtrado.sort_values('Data Início')
            
            # Calcular tempo entre falhas para cada equipamento
            mtbf_data = []
            for equipamento in df_ordenado['Equipamento'].unique():
                df_equip = df_ordenado[df_ordenado['Equipamento'] == equipamento].sort_values('Data Início')
                
                if len(df_equip) > 1:
                    # Calcular diferença entre início de parada atual e fim da parada anterior
                    df_equip['Tempo entre Falhas'] = df_equip['Data Início'] - df_equip['Data Fim'].shift(1)
                    mtbf = df_equip['Tempo entre Falhas'].mean().total_seconds() / 3600  # em horas
                    mtbf_data.append({'Equipamento': equipamento, 'MTBF (h)': mtbf})
            
            if mtbf_data:
                df_mtbf = pd.DataFrame(mtbf_data)
                fig9 = px.bar(
                    df_mtbf,
                    x='Equipamento',
                    y='MTBF (h)',
                    title="MTBF (Tempo Médio Entre Falhas) por Equipamento (h)",
                    labels={'Equipamento': 'Equipamento', 'MTBF (h)': 'MTBF (horas)'}
                )
                st.plotly_chart(fig9, use_container_width=True)
            else:
                st.info("Dados insuficientes para calcular MTBF. São necessárias múltiplas paradas por equipamento.")

        with tab4:
            # Tabela detalhada - INCLUINDO A COLUNA Parada_dias_horas
            st.subheader("Tabela Detalhada de Paradas")
            
            # Verificar se a coluna existe no DataFrame
            colunas_detalhes = [
                'Nº', 'Local', 'Equipamento', 'Identificação', 
                'Data Início', 'Data Fim', 'Causa', 'Status', 
                'Tempo Calculado (h)', 'Parada_dias_horas'
            ]
            
            # Manter apenas colunas que existem no DataFrame
            colunas_existentes = [col for col in colunas_detalhes if col in df_filtrado.columns]
            
            st.dataframe(
                df_filtrado[colunas_existentes],
                use_container_width=True
            )
            
            # Análise de causas
            st.subheader("Análise de Causas Mais Frequentes")
            
            # Extrair causas individuais (separadas por ;)
            todas_causas = []
            for causa in df_filtrado['Causa'].dropna():
                if ';' in str(causa):
                    todas_causas.extend([c.strip() for c in str(causa).split(';')])
                else:
                    todas_causas.append(str(causa).strip())
            
            causas_df = pd.DataFrame({'Causa': todas_causas})
            causas_count = causas_df['Causa'].value_counts().reset_index().head(10)
            
            fig10 = px.bar(
                causas_count,
                x='Causa',
                y='count',
                title="Top 10 Causas de Paradas",
                labels={'Causa': 'Causa', 'count': 'Frequência'}
            )
            st.plotly_chart(fig10, use_container_width=True)

        # Análise estatística
        st.header("Análise Estatística")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Resumo Estatístico do Tempo de Parada")
            st.dataframe(df_filtrado['Tempo Calculado (h)'].describe())

        with col2:
            st.subheader("Distribuição do Tempo de Parada")
            fig11 = px.histogram(
                df_filtrado,
                x='Tempo Calculado (h)',
                nbins=20,
                title="Distribuição do Tempo de Parada (h)"
            )
            st.plotly_chart(fig11, use_container_width=True)

        # LINHA DO TEMPO DAS PARADAS - ADICIONADO AQUI
        st.header("Linha do Tempo das Paradas")
        
        # Preparar dados para a linha do tempo
        timeline_data = []
        for _, row in df_filtrado.iterrows():
            timeline_data.append({
                'Equipamento': row['Equipamento'],
                'Local': row['Local'],
                'Início': row['Data Início'],
                'Fim': row['Data Fim'] if pd.notna(row['Data Fim']) else datetime.now(),
                'Duração': row['Tempo Calculado (h)'] if pd.notna(row['Tempo Calculado (h)']) else 0,
                'Causa': row['Causa'],
                'Status': row['Status']
            })

        timeline_df = pd.DataFrame(timeline_data)

        if not timeline_df.empty:
            fig_timeline = px.timeline(
                timeline_df, 
                x_start="Início", 
                x_end="Fim", 
                y="Equipamento",
                color="Local",
                hover_data=["Causa", "Status", "Duração"],
                title="Linha do Tempo das Paradas"
            )
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.write("Nenhum dado disponível para a linha do tempo.")

        # Insights e recomendações
        st.header("Insights e Recomendações")

        # Análise automática dos dados
        local_mais_paradas = df_filtrado['Local'].value_counts().index[0]
        equipamento_mais_paradas = df_filtrado['Equipamento'].value_counts().index[0]
        causa_mais_frequente = causas_count.iloc[0]['Causa'] if not causas_count.empty else "N/A"
        tempo_medio_parada = df_filtrado['Tempo Calculado (h)'].mean()
        
        st.info(f"""
        **Principais Insights:**
        1. **{local_mais_paradas}** é o local com o maior número de paradas
        2. **{equipamento_mais_paradas}** é o equipamento com mais ocorrências
        3. **{causa_mais_frequente}** é la causa mais frequente de paradas
        4. O tempo médio de parada é de **{tempo_medio_parada:.2f} horas**
        5. Existem **{paradas_abertas}** paradas em aberto atualmente
        """)

        st.success("""
        **Recomendações:**
        - Implementar programa de manutenção preventiva mais robusto para os equipamentos mais problemáticos
        - Priorizar a resolução das paradas em aberto
        - Analisar causas raiz dos problemas mais frequentes
        - Estabelecer metas de redução de tempo médio de parada
        - Melhorar registro das causas para facilitar análises
        """)
        
        # Opção para exportar dados filtrados
        st.sidebar.header("Exportar Dados")
        if st.sidebar.button("Exportar Dados Filtrados para CSV"):
            csv = df_filtrado.to_csv(index=False)
            st.sidebar.download_button(
                label="Baixar CSV",
                data=csv,
                file_name="paradas_filtradas.csv",
                mime="text/csv"
            )

    else:
        st.error("Erro ao processar o arquivo. Verifique se o formato está correto.")
else:
    st.info("👈 Por favor, faça upload de um arquivo Excel para começar a análise.")
    st.markdown("""
    ### Instruções:
    1. Clique no botão "Browse files" no sidebar para carregar sua planilha
    2. Certifique-se de que a planilha contenha as colunas:
       - Local, Equipamento, Identificação
       - Data Início, Data Fim
       - Causa, Status
    3. Use os filtros no sidebar para refinar sua análise
    """)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard desenvolvido para análise de paradas de equipamentos. "
    "Os dados são carregados diretamente da planilha fornecida."
)