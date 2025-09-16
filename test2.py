import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    .main-header {
        font-size: 2.5rem !important; 
        text-align: center; 
        color: #008542;
        margin-bottom: 1rem;
    }
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
# Upload de dados
# =====================
with st.expander("📁 Carregar Dados", expanded=True):
    uploaded_file = st.file_uploader("Carregue a base de dados (Excel)", type=["xlsx"], label_visibility="collapsed")

    if uploaded_file:
        # Leitura do arquivo
        try:
            dados = pd.read_excel(uploaded_file)
            st.success(f"Base carregada com sucesso! {dados.shape[0]} registros e {dados.shape[1]} colunas.")
            
            # Exibir informações do dataset
            with st.expander("Visualizar primeiras linhas"):
                st.dataframe(dados)
                
            with st.expander("Informações do dataset"):
                buffer = io.StringIO()
                dados.info(buf=buffer)
                st.text(buffer.getvalue())
                
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {str(e)}")
            st.stop()
    else:
        st.warning("Por favor, carregue a base 'base_normalizada.xlsx' para iniciar a análise.")
        st.stop()

# =====================
# Filtros na sidebar
# =====================
st.sidebar.header("🔍 Filtros")

# Verificar se as colunas necessárias existem antes de criar os filtros
colunas_disponiveis = dados.columns.tolist()

# Filtro de data se disponível
date_cols = [col for col in colunas_disponiveis if 'data' in col.lower() or 'date' in col.lower() or 'início' in col.lower()]
if date_cols:
    date_col = date_cols[0]  # Usar a primeira coluna de data encontrada
    
    # Converter para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(dados[date_col]):
        dados[date_col] = pd.to_datetime(dados[date_col], errors='coerce')
    
    # Remover valores NaT antes de calcular min e max
    datas_validas = dados[date_col].dropna()
    
    if len(datas_validas) > 0:
        min_date = datas_validas.min()
        max_date = datas_validas.max()
        
        date_range = st.sidebar.date_input(
            "Período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        st.sidebar.warning("Coluna de data não contém valores válidos.")

# Filtros para colunas categóricas
filtros_aplicados = {}

if "Local" in colunas_disponiveis:
    locais_options = dados["Local"].dropna().unique().tolist()
    locais_selecionados = st.sidebar.multiselect(
        "Selecione os Locais",
        options=locais_options,
        default=locais_options
    )
    filtros_aplicados['Local'] = locais_selecionados

if "Equipamento" in colunas_disponiveis:
    equipamentos_options = dados["Equipamento"].dropna().unique().tolist()
    equipamentos_selecionados = st.sidebar.multiselect(
        "Selecione os Equipamentos",
        options=equipamentos_options,
        default=equipamentos_options
    )
    filtros_aplicados['Equipamento'] = equipamentos_selecionados


# Aplicar filtros
filtro = dados.copy()

# Aplicar filtro de data se disponível
if date_cols and 'date_range' in locals() and len(date_range) == 2:
    data_inicio = pd.to_datetime(date_range[0])
    data_fim = pd.to_datetime(date_range[1])
    filtro = filtro[(filtro[date_col] >= data_inicio) & (filtro[date_col] <= data_fim)]

# Aplicar filtros categóricos
for coluna, valores in filtros_aplicados.items():
    if valores:  # Aplicar apenas se algum valor foi selecionado
        filtro = filtro[filtro[coluna].isin(valores)]

# Mostrar estatísticas de filtragem
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Registros totais:** {len(dados):,}")
st.sidebar.markdown(f"**Registros filtrados:** {len(filtro):,}")
st.sidebar.markdown(f"**Percentual:** {100*len(filtro)/len(dados):.1f}%")

# Botão para resetar filtros
if st.sidebar.button("🔄 Resetar Filtros"):
    st.experimental_rerun()

# =====================
# KPIs no topo - COM DEBUG
# =====================
st.markdown("## 📈 Métricas Principais")

# Debug: mostrar informações sobre o filtro
with st.expander("🔍 Debug - Informações do Filtro"):
    st.write(f"Tamanho do dataframe original: {len(dados)}")
    st.write(f"Tamanho do dataframe filtrado: {len(filtro)}")
    if len(filtro) > 0:
        st.write("Primeiras linhas do filtro:")
        st.dataframe(filtro)
        
        # Verificar valores únicos nas colunas de filtro
        if "Local" in filtro.columns:
            st.write("Valores únicos em 'Local':", filtro["Local"].unique())
        if "Equipamento" in filtro.columns:
            st.write("Valores únicos em 'Equipamento':", filtro["Equipamento"].unique())
    else:
        st.warning("O filtro está vazio!")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if "Tempo de Parada (h)" in filtro.columns:
        total_horas = filtro["Tempo de Parada (h)"].sum()
        st.metric("Total Horas Parada", f"{total_horas:.1f}h")
    else:
        st.metric("Total Horas Parada", "N/A")

with col2:
    if "Tempo de Parada (h)" in filtro.columns:
        media_paradas = filtro["Tempo de Parada (h)"].mean()
        st.metric("Tempo Médio de Parada", f"{media_paradas:.1f}h")
    else:
        st.metric("Tempo Médio de Parada", "N/A")

with col3:
    total_ocorrencias = len(filtro)
    st.metric("Total Ocorrências", total_ocorrencias)

with col4:
    if "Equipamento" in filtro.columns:
        equipamentos_unicos = filtro["Equipamento"].nunique()
        st.metric("Equipamentos com Paradas", equipamentos_unicos)
    else:
        st.metric("Equipamentos com Paradas", "N/A")

# =====================
# Análise Preditiva
# =====================
st.markdown("## 🔮 Análise Preditiva")

def preparar_dados_para_modelo(df):
    df_model = df.copy()
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    categorical_cols = ['Local', 'Equipamento', 'Identificação', 'Status', 'Causa']
    
    for col in categorical_cols:
        if col in df_model.columns:
            # Preencher valores NaN com string vazia antes de codificar
            df_model[col] = df_model[col].fillna('Missing')
            df_model[col] = le.fit_transform(df_model[col].astype(str))
    
    # Criar variável alvo: paradas longas (acima da média)
    if "Tempo de Parada (h)" in df_model.columns:
        tempo_medio = df_model["Tempo de Parada (h)"].mean()
        df_model['Parada_Longa'] = (df_model["Tempo de Parada (h)"] > tempo_medio).astype(int)
    
    return df_model

def treinar_modelo(df):
    if "Parada_Longa" not in df.columns:
        st.error("Não foi possível criar a variável alvo para o modelo.")
        return None, None, None
    
    # Selecionar features e target
    features = ['Local', 'Equipamento', 'Identificação', 'Status']
    # Usar apenas features que existem no dataframe
    features = [f for f in features if f in df.columns]
    
    if not features:
        st.error("Nenhuma feature disponível para treinar o modelo.")
        return None, None, None
        
    X = df[features]
    y = df['Parada_Longa']
    
    # Verificar se temos dados suficientes
    if len(X) < 10:
        st.warning("Dados insuficientes para treinar o modelo preditivo.")
        return None, None, None
    
    # Verificar se há variabilidade na variável alvo
    if y.nunique() < 2:
        st.warning("Variável alvo não tem variabilidade suficiente para treinamento.")
        return None, None, None
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X, accuracy

# Aplicar preparação e modelagem
if len(filtro) >= 10:  # Apenas se tivermos dados suficientes
    df_modelo = preparar_dados_para_modelo(filtro)
    modelo, X, acuracia = treinar_modelo(df_modelo)
else:
    st.warning("Dados insuficientes após filtragem para análise preditiva.")
    modelo, X, acuracia = None, None, None

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
        st.info("O modelo pode prever se uma nova parada tende a ser longa com base nas características do equipamento e local.")
        
        # Formulário para simular previsão
        with st.form("form_previsao"):
            # Coletar inputs do usuário apenas para colunas disponíveis
            input_data = {}
            
            if "Local" in dados.columns:
                local_selecionado = st.selectbox("Local", options=dados["Local"].dropna().unique())
                input_data['Local'] = local_selecionado
                
            if "Equipamento" in dados.columns:
                equipamento_selecionado = st.selectbox("Equipamento", options=dados["Equipamento"].dropna().unique())
                input_data['Equipamento'] = equipamento_selecionado
                
            if "Identificação" in dados.columns:
                identificacao_selecionada = st.selectbox("Identificação", options=dados["Identificação"].dropna().unique())
                input_data['Identificação'] = identificacao_selecionada
                
            if "Status" in dados.columns:
                status_selecionado = st.selectbox("Status", options=dados["Status"].dropna().unique())
                input_data['Status'] = status_selecionado
            
            submitted = st.form_submit_button("Fazer Previsão")
            
            if submitted:
                # Codificar entradas como o modelo espera
                le = LabelEncoder()
                encoded_data = {}
                
                for col, value in input_data.items():
                    if col in dados.columns:
                        # Ajustar os label encoders com os dados originais
                        le.fit(dados[col].astype(str).fillna('Missing'))
                        encoded_data[col] = le.transform([value])[0]
                
                # Criar DataFrame com os dados codificados
                nova_entrada = pd.DataFrame([encoded_data])
                
                # Garantir que as colunas estão na ordem correta e preencher missing com 0
                for col in X.columns:
                    if col not in nova_entrada.columns:
                        nova_entrada[col] = 0
                
                nova_entrada = nova_entrada[X.columns]
                
                # Fazer previsão
                try:
                    previsao = modelo.predict(nova_entrada)[0]
                    probabilidade = modelo.predict_proba(nova_entrada)[0][1]
                    
                    if previsao == 1:
                        st.error(f"⚠️ Previsão: Parada LONGA (probabilidade: {probabilidade*100:.1f}%)")
                        st.write("Recomenda-se alocar mais recursos para minimizar o tempo de inatividade.")
                    else:
                        st.success(f"✅ Previsão: Parada CURTA (probabilidade: {(1-probabilidade)*100:.1f}%)")
                        st.write("Parada esperada dentro do tempo médio histórico.")
                except Exception as e:
                    st.error(f"Erro ao fazer previsão: {str(e)}")

# =====================
# Previsão de Série Temporal
# =====================
date_cols = [col for col in colunas_disponiveis if 'data' in col.lower() or 'date' in col.lower() or 'início' in col.lower()]
if date_cols and "Tempo de Parada (h)" in colunas_disponiveis and len(filtro) > 0:
    st.markdown("## 📅 Previsão de Série Temporal")
    
    try:
        # Preparar dados temporais
        date_col = date_cols[0]
        filtro[date_col] = pd.to_datetime(filtro[date_col], errors='coerce')
        # Remover datas inválidas
        filtro_temporal = filtro.dropna(subset=[date_col])
        
        if len(filtro_temporal) > 0:
            series_temporal = filtro_temporal.groupby(filtro_temporal[date_col].dt.to_period('M')).agg({
                'Tempo de Parada (h)': 'sum',
                'Identificação': 'count'
            }).rename(columns={'Tempo de Parada (h)': 'Horas_Parada', 'Identificação': 'Ocorrencias'})
            series_temporal.index = series_temporal.index.to_timestamp()
            
            # Previsão simples usando média móvel
            window = min(3, len(series_temporal) - 1)  # Ajustar window se não houver dados suficientes
            
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
                if not pd.isna(series_temporal['Previsao_Horas'].iloc[-1]):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Previsão próximo mês - Horas de Parada", 
                                 f"{series_temporal['Previsao_Horas'].iloc[-1]:.1f}h")
                    
                    with col2:
                        st.metric("Previsão próximo mês - Ocorrências", 
                                 f"{series_temporal['Previsao_Ocorrencias'].iloc[-1]:.0f}")
                else:
                    st.info("Não há previsão disponível para o próximo período.")
            else:
                st.info("Dados insuficientes para previsão de série temporal.")
        else:
            st.info("Não há dados válidos para análise de série temporal.")
            
    except Exception as e:
        st.error(f"Erro ao processar previsão de série temporal: {e}")

# =====================
# Análise de Tendências e Padrões
# =====================
st.markdown("## 📈 Análise de Tendências e Padrões")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 Padrões de Paradas por Dia da Semana")
    
    date_cols = [col for col in colunas_disponiveis if 'data' in col.lower() or 'date' in col.lower() or 'início' in col.lower()]
    if date_cols and len(filtro) > 0:
        date_col = date_cols[0]
        filtro[date_col] = pd.to_datetime(filtro[date_col], errors='coerce')
        filtro_dias = filtro.dropna(subset=[date_col])
        
        if len(filtro_dias) > 0:
            filtro_dias['Dia_Semana'] = filtro_dias[date_col].dt.day_name()
            
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
            filtro_dias['Dia_Semana'] = filtro_dias['Dia_Semana'].map(dias_pt)
            
            paradas_dia = filtro_dias['Dia_Semana'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=paradas_dia.values, y=paradas_dia.index, ax=ax, orient='h')
            ax.set_title("Paradas por Dia da Semana", fontweight='bold')
            ax.set_xlabel("Número de Paradas")
            st.pyplot(fig)
            
            # Identificar dia com mais paradas
            if len(paradas_dia) > 0:
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
# Controle Estatístico de Processo (CEP)
# =====================
st.markdown("## 📈 Controle Estatístico de Processo (CEP)")

date_cols = [col for col in colunas_disponiveis if 'data' in col.lower() or 'date' in col.lower() or 'início' in col.lower()]
if date_cols and "Tempo de Parada (h)" in colunas_disponiveis and len(filtro) > 0:
    variavel_cep = st.selectbox("Selecione a variável para análise CEP", options=["Tempo de Parada (h)", "Número de Ocorrências"])
    
    try:
        date_col = date_cols[0]
        dados_cep = filtro[[date_col, 'Tempo de Parada (h)']].copy()
        dados_cep[date_col] = pd.to_datetime(dados_cep[date_col], errors='coerce')
        dados_cep = dados_cep.dropna(subset=[date_col])
        dados_cep = dados_cep.set_index(date_col).sort_index()
        
        if variavel_cep == "Número de Ocorrências":
            # Agrupar por dia para contar ocorrências
            dados_cep = dados_cep.resample('D').count()
            dados_cep.columns = ['Contagem']
            serie_i = dados_cep['Contagem']
        else:
            # Usar tempo de parada individual
            serie_i = dados_cep['Tempo de Parada (h)']
        
        if len(serie_i) >= 10:
            # Calcular limites I-MR
            media_i = serie_i.mean()
            mr = np.abs(serie_i.diff())
            mr_media = mr.mean()
            
            # Limites para o gráfico I
            lcl_i = max(0, media_i - (2.66 * mr_media))
            ucl_i = media_i + (2.66 * mr_media)
            
            # Limites para o gráfico MR
            lcl_mr = 0
            ucl_mr = 3.267 * mr_media
            
            # Criar gráficos
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfico I
            ax1.plot(serie_i.index, serie_i.values, marker='o', linestyle='-')
            ax1.axhline(media_i, color='g', linestyle='--', label='Linha Central (CL)')
            ax1.axhline(ucl_i, color='r', linestyle='--', label='Limite Superior de Controle (UCL)')
            ax1.axhline(lcl_i, color='r', linestyle='--', label='Limite Inferior de Controle (LCL)')
            ax1.set_title(f'Gráfico de Controle I - {variavel_cep}', fontweight='bold')
            ax1.set_ylabel(variavel_cep)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico MR
            mr_data = np.abs(serie_i.diff())
            ax2.plot(serie_i.index[1:], mr_data[1:], marker='s', linestyle='-', color='orange')
            ax2.axhline(mr_media, color='g', linestyle='--', label='Linha Central (CL)')
            ax2.axhline(ucl_mr, color='r', linestyle='--', label='Limite Superior de Controle (UCL)')
            ax2.axhline(lcl_mr, color='r', linestyle='--', label='Limite Inferior de Controle (LCL)')
            ax2.set_title('Gráfico de Controle MR - Amplitude Móvel', fontweight='bold')
            ax2.set_ylabel('Amplitude Móvel')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Identificar pontos fora de controle
            pontos_fora_i = (serie_i > ucl_i) | (serie_i < lcl_i)
            if pontos_fora_i.any():
                st.warning(f"**Sinal de causa especial detectado:** {pontos_fora_i.sum()} ponto(s) fora dos limites de controle.")
                
        else:
            st.info("Dados insuficientes para análise CEP (mínimo de 10 pontos necessários).")
            
    except Exception as e:
        st.error(f"Erro na análise CEP: {str(e)}")

# =====================
# Recomendações Baseadas em Dados
# =====================
st.markdown("## 🎯 Recomendações Preditivas")

if "Tempo de Parada (h)" in colunas_disponiveis and "Equipamento" in colunas_disponiveis and len(filtro) > 0:
    try:
        equipamento_analysis = filtro.groupby('Equipamento').agg({
            'Tempo de Parada (h)': ['sum', 'mean', 'count']
        }).round(1)
        
        # Simplificar colunas
        equipamento_analysis.columns = ['Total_Horas', 'Media_Horas', 'Numero_Ocorrencias']
        equipamento_analysis = equipamento_analysis.sort_values('Total_Horas', ascending=False)
        
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
            
    except Exception as e:
        st.error(f"Erro ao analisar equipamentos críticos: {str(e)}")

# =====================
# Download dos dados filtrados
# =====================
st.markdown("---")
st.markdown("### 📥 Exportar Dados Filtrados")

if len(filtro) > 0:
    # Converter DataFrame para CSV
    csv = filtro.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Baixar dados filtrados como CSV",
        data=csv,
        file_name=f"dados_filtrados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        icon="📥"
    )
else:
    st.warning("Não há dados para exportar após a filtragem.")