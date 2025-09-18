import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openpyxl
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="KPIs de Manutenção",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("📊 Dashboard de KPIs de Manutenção")

# Função para carregar dados via upload
def load_data():
    uploaded_file = st.file_uploader("Faça upload da sua base de dados", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Verifica e converte colunas de data
            if 'Data Início' in df.columns:
                df['Data Início'] = pd.to_datetime(df['Data Início'], errors='coerce')
            else:
                st.error("A coluna 'Data Início' não foi encontrada no arquivo.")
                return pd.DataFrame()
                
            if 'Data Fim' in df.columns:
                df['Data Fim'] = pd.to_datetime(df['Data Fim'], errors='coerce')
            
            # Calcula tempo de parada se não existir
            if 'Tempo de Parada (h)' not in df.columns:
                mask = df['Data Fim'].notna() & df['Data Início'].notna()
                df.loc[mask, 'Tempo de Parada (h)'] = (df.loc[mask, 'Data Fim'] - df.loc[mask, 'Data Início']).dt.total_seconds() / 3600
            
            return df
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            return pd.DataFrame()
    else:
        # Retorna DataFrame vazio se não houver upload
        return pd.DataFrame()

# Carregar dados
df = load_data()

# Verifica se os dados foram carregados
if df.empty:
    st.info("Faça upload de um arquivo Excel para visualizar os dados.")
    st.stop()

# Sidebar com filtros
st.sidebar.header("Filtros")

# Filtro por local
if 'Local' in df.columns:
    locais = list(df['Local'].unique())
    locais_selecionados = st.sidebar.multiselect(
        'Selecione os Locais:',
        options=locais,
        default=locais
    )
else:
    st.sidebar.warning("Coluna 'Local' não encontrada nos dados.")
    locais_selecionados = []

# Filtro por equipamento
if 'Equipamento' in df.columns:
    equipamentos = list(df['Equipamento'].unique())
    equipamentos_selecionados = st.sidebar.multiselect(
        'Selecione os Equipamentos:',
        options=equipamentos,
        default=equipamentos
    )
else:
    st.sidebar.warning("Coluna 'Equipamento' não encontrada nos dados.")
    equipamentos_selecionados = []

# Filtro por status
if 'Status' in df.columns:
    status = list(df['Status'].unique())
    status_selecionados = st.sidebar.multiselect(
        'Selecione os Status:',
        options=status,
        default=status
    )
else:
    st.sidebar.warning("Coluna 'Status' não encontrada nos dados.")
    status_selecionados = []

# Filtro por período
if 'Data Início' in df.columns:
    min_date = df['Data Início'].min()
    max_date = df['Data Início'].max()

    periodo = st.sidebar.date_input(
        'Selecione o Período:',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    st.sidebar.warning("Coluna 'Data Início' não encontrada nos dados.")
    periodo = []

# Aplicar filtros
df_filtrado = df.copy()

if 'Local' in df.columns and locais_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Local'].isin(locais_selecionados)]

if 'Equipamento' in df.columns and equipamentos_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Equipamento'].isin(equipamentos_selecionados)]

if 'Status' in df.columns and status_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Status'].isin(status_selecionados)]

if 'Data Início' in df.columns and len(periodo) == 2:
    data_inicio = pd.to_datetime(periodo[0])
    data_fim = pd.to_datetime(periodo[1])
    df_filtrado = df_filtrado[
        (df_filtrado['Data Início'] >= data_inicio) &
        (df_filtrado['Data Início'] <= data_fim)
    ]

# Restante do código (cálculo de KPIs, gráficos, etc.) permanece igual...
# [O restante do código anterior continua aqui]

# Cálculo dos KPIs
paradas_fechadas = df_filtrado[df_filtrado['Status'] == 'Fechado']
paradas_abertas = df_filtrado[df_filtrado['Status'] == 'Aberto']

# MTTR (Mean Time To Repair)
if len(paradas_fechadas) > 0:
    mttr = paradas_fechadas['Tempo de Parada (h)'].mean()
else:
    mttr = 0

# MTBF (Mean Time Between Failures) - simplificado
if len(paradas_fechadas) > 1:
    paradas_ordenadas = paradas_fechadas.sort_values('Data Início')
    tempos_entre_falhas = []
    
    for i in range(1, len(paradas_ordenadas)):
        tempo_entre_falhas = (paradas_ordenadas.iloc[i]['Data Início'] - 
                              paradas_ordenadas.iloc[i-1]['Data Fim']).total_seconds() / 3600
        if tempo_entre_falhas > 0:
            tempos_entre_falhas.append(tempo_entre_falhas)
    
    if tempos_entre_falhas:
        mtbf = sum(tempos_entre_falhas) / len(tempos_entre_falhas)
    else:
        mtbf = 0
else:
    mtbf = 0

# Disponibilidade (simplificada)
if len(paradas_fechadas) > 0:
    tempo_total_parada = paradas_fechadas['Tempo de Parada (h)'].sum()
    # Supondo um período de operação de 30 dias (720 horas) para simplificação
    tempo_total_periodo = 720
    disponibilidade = ((tempo_total_periodo - tempo_total_parada) / tempo_total_periodo) * 100
else:
    disponibilidade = 100

# Número total de paradas
total_paradas = len(df_filtrado)

# Exibir KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MTTR (Horas)", f"{mttr:.2f}")
with col2:
    st.metric("MTBF (Horas)", f"{mtbf:.2f}")
with col3:
    st.metric("Disponibilidade (%)", f"{disponibilidade:.2f}")
with col4:
    st.metric("Total de Paradas", total_paradas)

# Gráficos e visualizações
st.subheader("Análise de Paradas por Local")
paradas_por_local = df_filtrado['Local'].value_counts()
fig_local = px.bar(
    x=paradas_por_local.index,
    y=paradas_por_local.values,
    labels={'x': 'Local', 'y': 'Número de Paradas'},
    title="Paradas por Local"
)
st.plotly_chart(fig_local, use_container_width=True)

st.subheader("Análise de Paradas por Equipamento")
paradas_por_equipamento = df_filtrado['Equipamento'].value_counts()
fig_equipamento = px.bar(
    x=paradas_por_equipamento.index,
    y=paradas_por_equipamento.values,
    labels={'x': 'Equipamento', 'y': 'Número de Paradas'},
    title="Paradas por Equipamento"
)
st.plotly_chart(fig_equipamento, use_container_width=True)

st.subheader("Tempo de Parada por Mês")
df_filtrado['Mês'] = df_filtrado['Data Início'].dt.to_period('M').astype(str)
tempo_por_mes = df_filtrado.groupby('Mês')['Tempo de Parada (h)'].sum().reset_index()
fig_tempo_mes = px.line(
    tempo_por_mes,
    x='Mês',
    y='Tempo de Parada (h)',
    title="Tempo Total de Parada por Mês"
)
st.plotly_chart(fig_tempo_mes, use_container_width=True)

st.subheader("Tipo de Manutenção (Preventiva vs Corretiva)")
# Classificação simplificada baseada nas causas
def classificar_manutencao(causa):
    if pd.isna(causa):
        return "Não Especificada"
    causa_lower = str(causa).lower()
    if 'preventiv' in causa_lower:
        return "Preventiva"
    elif 'lavagem' in causa_lower:
        return "Preventiva"
    else:
        return "Corretiva"

df_filtrado['Tipo Manutenção'] = df_filtrado['Causa'].apply(classificar_manutencao)
manutencao_por_tipo = df_filtrado['Tipo Manutenção'].value_counts()
fig_tipo = px.pie(
    values=manutencao_por_tipo.values,
    names=manutencao_por_tipo.index,
    title="Distribuição por Tipo de Manutenção"
)
st.plotly_chart(fig_tipo, use_container_width=True)

st.subheader("Linha do Tempo das Paradas")
# Preparar dados para a linha do tempo
timeline_data = []
for _, row in df_filtrado.iterrows():
    timeline_data.append({
        'Equipamento': row['Equipamento'],
        'Local': row['Local'],
        'Início': row['Data Início'],
        'Fim': row['Data Fim'] if pd.notna(row['Data Fim']) else datetime.now(),
        'Duração': row['Tempo de Parada (h)'] if pd.notna(row['Tempo de Parada (h)']) else 0,
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

# Tabela com dados detalhados
st.subheader("Dados Detalhados das Paradas")
st.dataframe(df_filtrado)

# Download dos dados filtrados
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtrado)

st.download_button(
    label="Baixar dados filtrados como CSV",
    data=csv,
    file_name="paradas_manutencao.csv",
    mime="text/csv",
)

# Notas e informações adicionais
st.sidebar.info("""
**Notas:**
- MTTR: Tempo Médio para Reparo
- MTBF: Tempo Médio Entre Falhas
- Disponibilidade: Percentual de tempo operacional
""")