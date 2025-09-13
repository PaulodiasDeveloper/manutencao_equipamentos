import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configurações gerais
plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

st.set_page_config(
    page_title="Análise Preditiva de Paradas", 
    layout="wide",
    page_icon="🔮"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {font-size: 2.5rem !important; justify-self: center; color: #008542;}
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
    .pred-positive {color: #ff4b4b; font-weight: bold;}
    .pred-negative {color: #0068c9; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Cabeçalho
st.markdown('<p class="main-header">🔮 Análise Preditiva de Paradas de Equipamentos</p>', unsafe_allow_html=True)
st.markdown("Este relatório apresenta análise estatística e **preditiva** das paradas de equipamentos, com insights para gestão proativa da manutenção.")

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
# Análise Preditiva
# =====================

st.markdown("## 🔮 Análise Preditiva")

# Preparar dados para modelagem preditiva
def preparar_dados_para_modelo(df):
    # Criar uma cópia para não modificar o original
    df_model = df.copy()
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    categorical_cols = ['Local', 'Equipamento', 'Identificação', 'Status', 'Causa']
    
    for col in categorical_cols:
        if col in df_model.columns:
            df_model[col] = le.fit_transform(df_model[col].astype(str))
    
    # Criar variável alvo: paradas longas (acima da média)
    if "Tempo de Parada (h)" in df_model.columns:
        tempo_medio = df_model["Tempo de Parada (h)"].mean()
        df_model['Parada_Longa'] = (df_model["Tempo de Parada (h)"] > tempo_medio).astype(int)
    
    return df_model

# Treinar modelo preditivo
def treinar_modelo(df):
    if "Parada_Longa" not in df.columns:
        st.error("Não foi possível criar a variável alvo para o modelo.")
        return None, None, None
    
    # Selecionar features e target
    features = ['Local', 'Equipamento', 'Identificação', 'Status']
    X = df[[f for f in features if f in df.columns]]
    y = df['Parada_Longa']
    
    # Verificar se temos dados suficientes
    if len(X) < 10:
        st.warning("Dados insuficientes para treinar o modelo preditivo.")
        return None, None, None
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X, accuracy

# Aplicar preparação e modelagem
df_modelo = preparar_dados_para_modelo(filtro)
modelo, X, acuracia = treinar_modelo(df_modelo)

# Exibir resultados do modelo
if modelo is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Desempenho do Modelo Preditivo")
        st.metric("Acurácia do Modelo", f"{acuracia*100:.1f}%")
        st.caption("Previsão de paradas longas (acima da média)")
        
        # Importância das features
        if hasattr(modelo, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_title("Importância das Variáveis na Previsão", fontweight='bold')
            st.pyplot(fig)
    
    with col2:
        st.markdown("### 🔍 Previsões para Novos Dados")
        
        # Simular previsão para novos dados
        st.info("O modelo pode prever se uma nova parada tende a ser longa com base nas características do equipamento e local.")
        
        # Formulário para simular previsão
        with st.form("form_previsao"):
            local_selecionado = st.selectbox("Local", options=dados["Local"].unique())
            equipamento_selecionado = st.selectbox("Equipamento", options=dados["Equipamento"].unique())
            identificacao_selecionada = st.selectbox("Identificação", options=dados["Identificação"].unique())
            status_selecionado = st.selectbox("Status", options=dados["Status"].unique())
            
            submitted = st.form_submit_button("Fazer Previsão")
            
            if submitted:
                # Codificar entradas como o modelo espera
                le = LabelEncoder()
                
                # Ajustar os label encoders com os dados originais
                for col in ['Local', 'Equipamento', 'Identificação', 'Status']:
                    le.fit(dados[col].astype(str))
                    
                    if col == 'Local':
                        local_encoded = le.transform([local_selecionado])[0]
                    elif col == 'Equipamento':
                        equipamento_encoded = le.transform([equipamento_selecionado])[0]
                    elif col == 'Identificação':
                        identificacao_encoded = le.transform([identificacao_selecionada])[0]
                    elif col == 'Status':
                        status_encoded = le.transform([status_selecionado])[0]
                
                # Fazer previsão
                nova_entrada = pd.DataFrame({
                    'Local': [local_encoded],
                    'Equipamento': [equipamento_encoded],
                    'Identificação': [identificacao_encoded],
                    'Status': [status_encoded]
                })
                
                # Garantir que as colunas estão na ordem correta
                nova_entrada = nova_entrada[X.columns]
                
                previsao = modelo.predict(nova_entrada)[0]
                probabilidade = modelo.predict_proba(nova_entrada)[0][1]
                
                if previsao == 1:
                    st.error(f"⚠️ Previsão: Parada LONGA (probabilidade: {probabilidade*100:.1f}%)")
                    st.write("Recomenda-se alocar mais recursos para minimizar o tempo de inatividade.")
                else:
                    st.success(f"✅ Previsão: Parada CURTA (probabilidade: {(1-probabilidade)*100:.1f}%)")
                    st.write("Parada esperada dentro do tempo médio histórico.")

# =====================
# Previsão de Série Temporal
# =====================

if "Data Início" in filtro.columns:
    st.markdown("## 📅 Previsão de Série Temporal")
    
    try:
        # Preparar dados temporais
        filtro['Data Início'] = pd.to_datetime(filtro['Data Início'])
        series_temporal = filtro.groupby(filtro['Data Início'].dt.to_period('M')).agg({
            'Tempo de Parada (h)': 'sum',
            'Identificação': 'count'
        }).rename(columns={'Tempo de Parada (h)': 'Horas_Parada', 'Identificação': 'Ocorrencias'})
        series_temporal.index = series_temporal.index.to_timestamp()
        
        # Previsão simples usando média móvel
        window = 3
        if len(series_temporal) > window:
            series_temporal['Previsao_Horas'] = series_temporal['Horas_Parada'].rolling(window=window).mean().shift(1)
            series_temporal['Previsao_Ocorrencias'] = series_temporal['Ocorrencias'].rolling(window=window).mean().shift(1)
            
            # Gráfico de previsão
            fig, ax = plt.subplots(2, 1, figsize=(12, 10))
            
            # Previsão de horas de parada
            ax[0].plot(series_temporal.index, series_temporal['Horas_Parada'], marker='o', label='Real', linewidth=2)
            ax[0].plot(series_temporal.index, series_temporal['Previsao_Horas'], marker='s', label='Previsão', linewidth=2)
            ax[0].set_title("Previsão de Horas de Parada (Média Móvel)", fontweight='bold')
            ax[0].set_ylabel("Horas de Parada")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Previsão de ocorrências
            ax[1].plot(series_temporal.index, series_temporal['Ocorrencias'], marker='o', label='Real', linewidth=2)
            ax[1].plot(series_temporal.index, series_temporal['Previsao_Ocorrencias'], marker='s', label='Previsão', linewidth=2)
            ax[1].set_title("Previsão de Número de Ocorrências (Média Móvel)", fontweight='bold')
            ax[1].set_ylabel("Número de Ocorrências")
            ax[1].set_xlabel("Data")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Estatísticas de previsão
            ultima_data = series_temporal.index[-1]
            ultima_previsao_horas = series_temporal['Previsao_Horas'].iloc[-1]
            ultima_previsao_ocorrencias = series_temporal['Previsao_Ocorrencias'].iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Previsão próximo mês - Horas de Parada", 
                         f"{ultima_previsao_horas:.1f}h" if not pd.isna(ultima_previsao_horas) else "N/A")
            
            with col2:
                st.metric("Previsão próximo mês - Ocorrências", 
                         f"{ultima_previsao_ocorrencias:.0f}" if not pd.isna(ultima_previsao_ocorrencias) else "N/A")
            
    except Exception as e:
        st.error(f"Erro ao processar previsão de série temporal: {e}")

# =====================
# Análise de Tendências
# =====================

st.markdown("## 📈 Análise de Tendências e Padrões")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 Padrões de Paradas por Dia da Semana")
    
    if "Data Início" in filtro.columns:
        filtro['Dia_Semana'] = filtro['Data Início'].dt.day_name()
        
        # Traduzir dias da semana se necessário
        dias_pt = {
            'Monday': 'Segunda',
            'Tuesday': 'Terça',
            'Wednesday': 'Quarta',
            'Thursday': 'Quinta',
            'Friday': 'Sexta',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
        filtro['Dia_Semana'] = filtro['Dia_Semana'].map(dias_pt)
        
        paradas_dia = filtro['Dia_Semana'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=paradas_dia.values, y=paradas_dia.index, ax=ax, orient='h')
        ax.set_title("Paradas por Dia da Semana", fontweight='bold')
        ax.set_xlabel("Número de Paradas")
        st.pyplot(fig)
        
        # Identificar dia com mais paradas
        dia_mais_paradas = paradas_dia.idxmax()
        st.info(f"**Dia com mais paradas:** {dia_mais_paradas} ({paradas_dia.max()} ocorrências)")

with col2:
    st.markdown("### 📊 Análise de Correlação")
    
    # Calcular matriz de correlação para dados numéricos
    numeric_cols = filtro.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = filtro[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Matriz de Correlação entre Variáveis Numéricas", fontweight='bold')
        st.pyplot(fig)
        
        # Encontrar correlações fortes
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_corr.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if strong_corr:
            st.write("**Correlações fortes identificadas:**")
            for corr in strong_corr:
                st.write(f"- {corr[0]} e {corr[1]}: {corr[2]:.2f}")




# =====================
# Cálculo dos Limites de Controle
# =====================

def calcular_limites_imr(serie_temporal):
    # Gráfico de individuais (I)
    media_i = serie_temporal.mean()
    mr = np.abs(serie_temporal.diff()) # Amplitude Móvel
    mr_media = mr.mean()
    # Limites de Controle para o gráfico I
    lcl_i = media_i - (2.66 * mr_media)
    ucl_i = media_i + (2.66 * mr_media)
    # Limites para o gráfico MR (usando constantes D3 e D4 para n=2)
    lcl_mr = 0 # Para n=2, D3=0
    ucl_mr = 3.267 * mr_media
    return media_i, lcl_i, ucl_i, mr_media, lcl_mr, ucl_mr
    

st.markdown("## 📈 Controle Estatístico de Processo (CEP)")

# Selecionar o que monitorar: Tempo de Parada ou Contagem de Ocorrências
variavel_cep = st.selectbox("Selecione a variável para análise CEP", options=["Tempo de Parada (h)", "Número de Ocorrências"])

if variavel_cep == "Tempo de Parada (h)":
    # Preparar dados temporais para I-MR
    dados_cep = filtro[['Data Início', 'Tempo de Parada (h)']].copy()
    dados_cep.set_index('Data Início', inplace=True)
    dados_cep.sort_index(inplace=True)

    # Calcular limites
    media_i, lcl_i, ucl_i, mr_media, lcl_mr, ucl_mr = calcular_limites_imr(dados_cep['Tempo de Parada (h)'])

    # Plotar Gráfico I
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(dados_cep.index, dados_cep['Tempo de Parada (h)'], marker='o', linestyle='-')
    ax1.axhline(media_i, color='g', linestyle='--', label='Linha Central (CL)')
    ax1.axhline(ucl_i, color='r', linestyle='--', label='Limite Superior de Controle (UCL)')
    ax1.axhline(lcl_i, color='r', linestyle='--', label='Limite Inferior de Controle (LCL)')
    ax1.set_title('Gráfico de Controle I - Tempo de Parada Individual')
    ax1.set_ylabel('Tempo de Parada (h)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plotar Gráfico MR
    mr_data = np.abs(dados_cep['Tempo de Parada (h)'].diff())
    ax2.plot(dados_cep.index[1:], mr_data[1:], marker='s', linestyle='-', color='orange')
    ax2.axhline(mr_media, color='g', linestyle='--', label='Linha Central (CL)')
    ax2.axhline(ucl_mr, color='r', linestyle='--', label='Limite Superior de Controle (UCL)')
    ax2.axhline(lcl_mr, color='r', linestyle='--', label='Limite Inferior de Controle (LCL)')
    ax2.set_title('Gráfico de Controle MR - Amplitude Móvel')
    ax2.set_ylabel('Amplitude Móvel (h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Análise de Sinal: Verificar se pontos estão fora dos limites de controle
    pontos_fora_limite_i = (dados_cep['Tempo de Parada (h)'] > ucl_i) | (dados_cep['Tempo de Parada (h)'] < lcl_i)
    if pontos_fora_limite_i.any():
        st.warning(f"**Sinal de causa especial detectado:** {pontos_fora_limite_i.sum()} ponto(s) no gráfico I estão fora dos limites de controle. Investigação recomendada.")
        st.dataframe(dados_cep[pontos_fora_limite_i])



# =====================
# Recomendações Baseadas em Dados
# =====================

st.markdown("## 🎯 Recomendações Preditivas")

if "Tempo de Parada (h)" in filtro.columns and "Equipamento" in filtro.columns:
    # Análise de equipamentos críticos
    equipamento_analysis = filtro.groupby('Equipamento').agg({
        'Tempo de Parada (h)': ['sum', 'mean', 'count']
    }).round(1)
    equipamento_analysis.columns = ['Total_Horas', 'Media_Horas', 'Numero_Ocorrencias']
    equipamento_analysis = equipamento_analysis.sort_values('Total_Horas', ascending=False)
    
    # Top 3 equipamentos com mais horas de parada
    top_equipamentos = equipamento_analysis.head(3)
    
    st.markdown("### ⚠️ Equipamentos Críticos")
    st.write("Estes equipamentos demandam atenção prioritária devido ao alto tempo total de parada:")
    
    for i, (equipamento, row) in enumerate(top_equipamentos.iterrows(), 1):
        st.markdown(f"""
        <div class="card">
            <h4>{i}. {equipamento}</h4>
            <p>Total de horas paradas: <span class="pred-positive">{row['Total_Horas']}h</span></p>
            <p>Número de ocorrências: {row['Numero_Ocorrencias']}</p>
            <p>Tempo médio de parada: {row['Media_Horas']:.1f}h</p>
        </div>
        """, unsafe_allow_html=True)





    
    # Recomendações específicas
    st.markdown("### 📋 Recomendações de Ação")
    
    recomendacoes = [
        "Implementar manutenção preventiva nos equipamentos críticos identificados",
        "Estabelecer plano de revisão para equipamentos com maior tempo médio de parada",
        "Criar estoque de peças de reposição para os equipamentos mais problemáticos",
        "Treinar equipe em procedimentos específicos para os equipamentos com mais ocorrências",
        "Monitorar continuamente os equipamentos preditos como de alto risco"
    ]
    
    for rec in recomendacoes:
        st.markdown(f"- {rec}")

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