import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

# Configurações gerais
plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

st.set_page_config(
    page_title="Análise de Paradas", 
    layout="wide",
    page_icon="📊"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4}
    .card { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 5px;
    }
    .highlight {background-color: #fff7e6; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Cabeçalho
st.markdown('<p class="main-header">📊 Análise de Paradas de Equipamentos</p>', unsafe_allow_html=True)
st.markdown("Este aplicativo permite explorar os dados de **paradas de equipamentos** de forma interativa.")

# =====================
# Upload ou leitura fixa
# =====================

with st.expander("📁 Carregar Dados", expanded=True):
    uploaded_file = st.file_uploader("Carregue a base de dados (Excel)", type=["xlsx"], label_visibility="collapsed")
    
    if uploaded_file:
        dados = pd.read_excel(uploaded_file)
        st.success(f"Base carregada com sucesso! {dados.shape[0]} registros e {dados.shape[1]} colunas.")
    else:
        st.warning("Por favor, carregue a base 'base_normalizada.xlsx' para iniciar a análise.")
        st.stop()

# =====================
# Filtros na sidebar
# =====================

st.sidebar.header("🔍 Filtros")

# Filtro de data se disponível
date_cols = [col for col in dados.columns if 'data' in col.lower() or 'date' in col.lower()]
if date_cols and pd.api.types.is_datetime64_any_dtype(dados[date_cols[0]]):
    min_date = dados[date_cols[0]].min()
    max_date = dados[date_cols[0]].max()
    date_range = st.sidebar.date_input(
        "Período", 
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

locais = st.sidebar.multiselect(
    "Selecione os Locais", 
    options=dados["Local"].unique(), 
    default=dados["Local"].unique()
)

equipamentos = st.sidebar.multiselect(
    "Selecione os Equipamentos", 
    options=dados["Equipamento"].unique(), 
    default=dados["Equipamento"].unique()
)

# Aplicar filtros
filtro = dados[(dados["Local"].isin(locais)) & (dados["Equipamento"].isin(equipamentos))]

# Mostrar estatísticas de filtragem
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Registros filtrados:** {len(filtro)} de {len(dados)}")
st.sidebar.markdown(f"**Percentual:** {100*len(filtro)/len(dados):.1f}%")

# =====================
# KPIs no topo
# =====================

st.markdown("## 📈 Métricas Principais")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if "Tempo de Parada (h)" in filtro.columns:
        total_horas = filtro["Tempo de Parada (h)"].sum()
        st.metric("Total Horas Parada", f"{total_horas:.1f}h")
        
with col2:
    media_paradas = filtro["Tempo de Parada (h)"].mean() if "Tempo de Parada (h)" in filtro.columns else 0
    st.metric("Tempo Médio de Parada", f"{media_paradas:.1f}h")

with col3:
    total_ocorrencias = len(filtro)
    st.metric("Total Ocorrências", total_ocorrencias)

with col4:
    equipamentos_unicos = filtro["Equipamento"].nunique()
    st.metric("Equipamentos com Paradas", equipamentos_unicos)

# =====================
# Exploração inicial
# =====================

tab1, tab2, tab3 = st.tabs(["📋 Dados", "ℹ️ Informações", "📊 Estatísticas"])

with tab1:
    st.subheader("Visualização dos Dados Filtrados")
    st.dataframe(filtro.head(10), use_container_width=True)
    
    # Opção para visualizar dados completos
    if st.checkbox("Mostrar todos os dados filtrados"):
        st.dataframe(filtro, use_container_width=True)

with tab2:
    st.subheader("Informações da Base de Dados")
    buffer = io.StringIO()
    filtro.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # Mostrar valores missing
    st.subheader("Valores Missing")
    missing = filtro.isnull().sum()
    st.dataframe(missing[missing > 0].rename("Quantidade"), use_container_width=True)

with tab3:
    st.subheader("Estatísticas Descritivas")
    st.dataframe(filtro.describe(include="all"), use_container_width=True)

# =====================
# Frequências
# =====================

st.markdown("## 📊 Análise de Frequência")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Ocorrências por Local")
    locais_freq = filtro["Local"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=locais_freq.index, x=locais_freq.values, ax=ax, orient='h')
    ax.set_title("Ocorrências por Local", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("#### Ocorrências por Equipamento")
    equip_freq = filtro["Equipamento"].value_counts().head(10)  # Top 10 apenas
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=equip_freq.index, x=equip_freq.values, ax=ax, orient='h')
    ax.set_title("Top 10 Equipamentos", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col3:
    st.markdown("#### Ocorrências por Identificação")
    id_freq = filtro["Identificação"].value_counts().head(10)  # Top 10 apenas
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=id_freq.index, x=id_freq.values, ax=ax, orient='h')
    ax.set_title("Top 10 Identificações", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# =====================
# Métricas de Posição
# =====================

if "Tempo de Parada (h)" in filtro.columns:
    st.markdown("## ⏱️ Métricas de Tempo de Parada")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tempo = filtro["Tempo de Parada (h)"]
        media = tempo.mean()
        mediana = tempo.median()
        moda = tempo.mode()[0] if not tempo.mode().empty else 0
        q1 = tempo.quantile(0.25)
        q3 = tempo.quantile(0.75)
        
        st.markdown("""
        <div class="card">
            <h4>Estatísticas de Tempo de Parada</h4>
        """, unsafe_allow_html=True)
        
        st.metric("Média", f"{media:.2f}h")
        st.metric("Mediana", f"{mediana:.2f}h")
        st.metric("Moda", f"{moda:.2f}h")
        st.metric("Q1 (25%)", f"{q1:.2f}h")
        st.metric("Q3 (75%)", f"{q3:.2f}h")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=tempo, ax=ax)
        ax.set_title("Distribuição do Tempo de Parada", fontweight='bold')
        st.pyplot(fig)
        
        # Histograma
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(tempo, kde=True, ax=ax)
        ax.set_title("Distribuição do Tempo de Parada", fontweight='bold')
        ax.set_xlabel("Tempo de Parada (h)")
        st.pyplot(fig)

# =====================
# Análises Cruzadas
# =====================

st.markdown("## 🔄 Análises Cruzadas")

tab1, tab2 = st.tabs(["Local x Status", "Equipamento x Causa"])

with tab1:
    st.markdown("#### Tabela Cruzada: Local x Status")
    local_status = pd.crosstab(filtro['Local'], filtro['Status'])
    st.dataframe(local_status, use_container_width=True)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(local_status, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title("Relação Local x Status", fontweight='bold')
    st.pyplot(fig)

with tab2:
    if "Causa" in filtro.columns:
        st.markdown("#### Tabela Cruzada: Equipamento x Causa")
        equip_causa = pd.crosstab(filtro['Equipamento'], filtro['Causa'])
        st.dataframe(equip_causa, use_container_width=True)
        
        # Heatmap para as principais causas
        top_causas = filtro['Causa'].value_counts().head(5).index
        filtro_top_causas = filtro[filtro['Causa'].isin(top_causas)]
        equip_causa_top = pd.crosstab(filtro_top_causas['Equipamento'], filtro_top_causas['Causa'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(equip_causa_top, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title("Principais Causas por Equipamento", fontweight='bold')
        st.pyplot(fig)

# =====================
# Séries Temporais
# =====================

if "Data Início" in filtro.columns:
    st.markdown("## 📅 Análise Temporal")
    
    try:
        filtro['Data Início'] = pd.to_datetime(filtro['Data Início'])
        filtro['AnoMes'] = filtro['Data Início'].dt.to_period('M').astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ocorrencias_mes = filtro.groupby('AnoMes').size()
            fig, ax = plt.subplots(figsize=(10, 6))
            ocorrencias_mes.plot(marker='o', ax=ax, linewidth=2)
            ax.set_title("Ocorrências de Paradas por Mês", fontweight='bold')
            ax.set_ylabel("Número de Ocorrências")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            tempo_medio_mes = filtro.groupby('AnoMes')["Tempo de Parada (h)"].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            tempo_medio_mes.plot(marker='s', color="orange", ax=ax, linewidth=2)
            ax.set_title("Tempo Médio de Reparos por Mês", fontweight='bold')
            ax.set_ylabel("Tempo Médio (h)")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Erro ao processar dados temporais: {e}")

# =====================
# Conclusão
# =====================

st.markdown("## 🎯 Conclusões e Insights")

with st.expander("Ver análises automáticas", expanded=True):
    st.markdown("""
    <div class="highlight">
    - O **Pátio de Valorização de Resíduos** concentra a maior parte das ocorrências.  
    - As **Manipuladoras** são os equipamentos que mais apresentam paradas.  
    - Algumas identificações específicas, como **MMG2404**, têm maior recorrência.  
    - O tempo de parada apresenta **assimetria**, com algumas ocorrências muito longas puxando a média para cima.  
    - A análise cruzada entre **equipamento e causa** revela padrões que podem direcionar ações de manutenção.  
    - A análise temporal ajuda a identificar **tendências e sazonalidade** nas paradas.  
    </div>
    """, unsafe_allow_html=True)
    
    # Análise automática dos dados
    if "Tempo de Parada (h)" in filtro.columns:
        top_equip = filtro["Equipamento"].value_counts().index[0]
        top_local = filtro["Local"].value_counts().index[0]
        max_tempo = filtro["Tempo de Parada (h)"].max()
        equip_max_tempo = filtro.loc[filtro["Tempo de Parada (h)"].idxmax(), "Equipamento"]
        
        st.markdown(f"""
        **Insights automáticos:**
        - O equipamento com mais ocorrências é: **{top_equip}**
        - O local com mais ocorrências é: **{top_local}**
        - A parada mais longa foi de **{max_tempo:.1f}h** no equipamento **{equip_max_tempo}**
        """)

# =====================
# Download dos dados filtrados
# =====================

st.markdown("---")
st.markdown("### 📥 Exportar Dados Filtrados")

# Converter DataFrame para CSV
csv = filtro.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Baixar dados filtrados como CSV",
    data=csv,
    file_name=f"dados_filtrados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    icon="📥"
)