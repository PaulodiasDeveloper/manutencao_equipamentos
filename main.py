import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openpyxl
from io import BytesIO
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(
    page_title="KPIs de Manutenção - Análise Completa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("📊 Análise de KPIs de Manutenção")

# Função para carregar dados via upload
def load_data():
    uploaded_file = st.file_uploader("📤 Faça upload da sua base de dados Excel", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Ler o arquivo Excel
            df = pd.read_excel(uploaded_file)
            
            # Verificar colunas obrigatórias
            colunas_obrigatorias = ['Data Início', 'Status']
            colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]
            
            if colunas_faltantes:
                st.error(f"❌ Colunas obrigatórias não encontradas: {', '.join(colunas_faltantes)}")
                st.info("ℹ️ As colunas necessárias são: 'Data Início' e 'Status'")
                return pd.DataFrame()
            
            # Converter colunas de data
            df['Data Início'] = pd.to_datetime(df['Data Início'], errors='coerce')
            
            if 'Data Fim' in df.columns:
                df['Data Fim'] = pd.to_datetime(df['Data Fim'], errors='coerce')
            
            # Calcular tempo de parada se não existir
            if 'Tempo de Parada (h)' not in df.columns:
                if 'Data Fim' in df.columns:
                    mask = df['Data Fim'].notna() & df['Data Início'].notna()
                    df.loc[mask, 'Tempo de Parada (h)'] = (df.loc[mask, 'Data Fim'] - df.loc[mask, 'Data Início']).dt.total_seconds() / 3600
                else:
                    st.warning("⚠️ Coluna 'Data Fim' não encontrada. Não foi possível calcular tempo de parada.")
            
            # Mostrar preview dos dados com toggle
            st.success("✅ Arquivo carregado com sucesso!")
            
            # # Checkbox para mostrar/ocultar preview
            # show_preview = st.checkbox("👁️ Mostrar preview dos dados (primeiras 5 linhas)", value=True)
            
            # if show_preview:
            #     st.write("📋 **Preview dos dados:**")
            #     st.dataframe(df.head())

            # Checkbox para mostrar/ocultar preview
            show_preview = st.checkbox("📋 Resumo da Análise de KPIs de Manutenção)", value=True)
            
            if show_preview: st.write(""" 
            
            No período de maio a agosto de 2025, a análise dos dados de manutenção revelou uma disponibilidade operacional crítica de 42,89%, com 20 paradas registradas e tempos médios de reparo (MTTR) elevados (108,10h), superando o tempo entre falhas (MTBF de 1945,8h). 
            
            A maioria das paradas (95,2%) concentrou-se no horário administrativo, com pico às 08h. 
                                      
            A Manipuladora foi o equipamento mais problemático, responsável por 47,1% das paradas. As principais causas incluem substituição de mangueiras hidráulicas e falhas mecânicas. A Pirâmide de Bird apontou uma base significativa de atos inseguros, indicando oportunidades de prevenção." "Recomenda-se revisão da manutenção preventiva, otimização do estoque de peças e atenção ao horário de pico para elevar a confiabilidade e a segurança operacional.""")


            # Mostrar informações do dataset
            st.markdown("## 📊 **Informações do dataset:**")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.write(f"**Total de registros:** {len(df)}")
            with col_info2:
                min_date = df['Data Início'].min()
                max_date = df['Data Início'].max()
                date_range = f"{min_date.strftime('%d/%m/%Y') if pd.notna(min_date) else 'N/A'} a {max_date.strftime('%d/%m/%Y') if pd.notna(max_date) else 'N/A'}"
                st.write(f"**Período:** {date_range}")
            with col_info3:
                st.write(f"**Colunas disponíveis:** {len(df.columns)}")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar o arquivo: {e}")
            return pd.DataFrame()
    else:
        # Instruções para o usuário
        st.info("""
        📝 **Instruções para upload:**
        1. Clique em "Browse files" ou arraste seu arquivo Excel
        2. O arquivo deve conter pelo menos as colunas:
           - `Data Início` (obrigatório)
           - `Status` (obrigatório)
           - `Data Fim` (opcional, mas recomendado)
        3. Formatos suportados: .xlsx, .xls
        """)
        
        # Exemplo de estrutura esperada
        st.write("📋 **Exemplo de estrutura esperada:**")
        exemplo_data = {
            'Data Início': ['2025-05-05 09:00:00', '2025-05-12 08:30:00'],
            'Data Fim': ['2025-05-05 15:00:00', '2025-05-13 09:50:00'],
            'Local': ['AGR Cabiúnas', 'AGR Cabiúnas'],
            'Equipamento': ['Empilhadeira 2.5 ton', 'Empilhadeira 4 ton'],
            'Causa': ['Freio de mão travado', 'Cabo de bateria com folga'],
            'Status': ['Fechado', 'Fechado']
        }
        st.dataframe(pd.DataFrame(exemplo_data))
        
        return pd.DataFrame()

# Função para análise de horários de pico (apenas horário administrativo)
def analise_horarios_pico(df):
    """Analisa os horários de pico de paradas apenas no horário administrativo"""
    
    # Extrair hora do dia
    df['Hora'] = df['Data Início'].dt.hour
    df['Minuto'] = df['Data Início'].dt.minute
    df['Hora_Completa'] = df['Data Início'].dt.floor('H')  # Arredonda para hora completa
    
    # Definir turnos apenas para horário administrativo
    def classificar_turno(hora):
        if 6 <= hora < 12:
            return 'Manhã (06:00-11:59)'
        elif 12 <= hora < 14:
            return 'Almoço (12:00-13:59)'
        elif 14 <= hora < 18:
            return 'Tarde (14:00-17:59)'
        elif 18 <= hora < 22:
            return 'Final de Expediente (18:00-21:59)'
        else:
            return 'Fora do Expediente'
    
    df['Turno'] = df['Hora'].apply(classificar_turno)
    
    # Filtrar apenas horário administrativo para análise
    df_admin = df[df['Turno'] != 'Fora do Expediente']
    
    return df, df_admin

# Função para criar gráfico de Pareto
def create_pareto_chart(data, category_column, value_column, title, height=500):
    """
    Cria um gráfico de Pareto
    """
    # Agrupar dados
    grouped_data = data.groupby(category_column)[value_column].sum().reset_index()
    grouped_data = grouped_data.sort_values(value_column, ascending=False)
    
    # Calcular percentual acumulado
    grouped_data['Cumulative Percentage'] = (grouped_data[value_column].cumsum() / grouped_data[value_column].sum() * 100)
    
    # Criar gráfico de barras
    fig = go.Figure()
    
    # Adicionar barras
    fig.add_trace(go.Bar(
        x=grouped_data[category_column],
        y=grouped_data[value_column],
        name='Quantidade',
        marker_color='blue'
    ))
    
    # Adicionar linha de Pareto
    fig.add_trace(go.Scatter(
        x=grouped_data[category_column],
        y=grouped_data['Cumulative Percentage'],
        name='Percentual Acumulado',
        yaxis='y2',
        mode='lines+markers',
        marker=dict(color='red', size=8),
        line=dict(color='red', width=2)
    ))
    
    # Configurar layout
    fig.update_layout(
        title=title,
        xaxis_title=category_column,
        yaxis_title=value_column,
        yaxis2=dict(
            title='Percentual Acumulado (%)',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig, grouped_data

# Carregar dados
df = load_data()

# Verificar se os dados foram carregados
if df.empty:
    st.warning("⏳ Aguardando upload do arquivo para análise...")
    st.stop()

# Aplicar análise de horários de pico
df, df_admin = analise_horarios_pico(df)

# Sidebar com filtros
st.sidebar.header("🔧 Filtros")

# Checkbox para mostrar/ocultar filtros avançados
show_advanced_filters = st.sidebar.checkbox("🎛️ Mostrar filtros avançados", value=True)

if show_advanced_filters:
    # Filtro por local (se a coluna existir)
    if 'Local' in df.columns:
        locais = list(df['Local'].unique())
        locais_selecionados = st.sidebar.multiselect(
            'Selecione os Locais:',
            options=locais,
            default=locais
        )
    else:
        st.sidebar.warning("⚠️ Coluna 'Local' não encontrada nos dados.")
        locais_selecionados = []

    # Filtro por equipamento (se a coluna existir)
    if 'Equipamento' in df.columns:
        equipamentos = list(df['Equipamento'].unique())
        equipamentos_selecionados = st.sidebar.multiselect(
            'Selecione os Equipamentos:',
            options=equipamentos,
            default=equipamentos
        )
    else:
        st.sidebar.warning("⚠️ Coluna 'Equipamento' não encontrada nos dados.")
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
        st.sidebar.warning("⚠️ Coluna 'Status' não encontrada nos dados.")
        status_selecionados = []

    # Filtro por turno (apenas horário administrativo)
    turnos_admin = [t for t in df['Turno'].unique() if t != 'Fora do Expediente']
    turnos_selecionados = st.sidebar.multiselect(
        'Selecione os Turnos:',
        options=turnos_admin,
        default=turnos_admin
    )

    # Filtro por período
    if 'Data Início' in df.columns:
        min_date = df['Data Início'].min()
        max_date = df['Data Início'].max()

        if pd.notna(min_date) and pd.notna(max_date):
            periodo = st.sidebar.date_input(
                'Selecione o Período:',
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            st.sidebar.warning("⚠️ Datas inválidas para filtro de período.")
            periodo = []
    else:
        st.sidebar.warning("⚠️ Coluna 'Data Início' não encontrada nos dados.")
        periodo = []
else:
    # Se filtros avançados estiverem ocultos, usar todos os dados
    locais_selecionados = list(df['Local'].unique()) if 'Local' in df.columns else []
    equipamentos_selecionados = list(df['Equipamento'].unique()) if 'Equipamento' in df.columns else []
    status_selecionados = list(df['Status'].unique()) if 'Status' in df.columns else []
    turnos_selecionados = [t for t in df['Turno'].unique() if t != 'Fora do Expediente']
    periodo = []

# Aplicar filtros CORRETAMENTE
df_filtrado = df.copy()

# Aplicar filtros apenas se as colunas existirem e se houver seleção
if 'Local' in df.columns and locais_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Local'].isin(locais_selecionados)]

if 'Equipamento' in df.columns and equipamentos_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Equipamento'].isin(equipamentos_selecionados)]

if 'Status' in df.columns and status_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Status'].isin(status_selecionados)]

if 'Turno' in df.columns and turnos_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Turno'].isin(turnos_selecionados)]

# Filtro de período - CORREÇÃO IMPORTANTE
if 'Data Início' in df.columns and len(periodo) == 2:
    try:
        data_inicio = pd.to_datetime(periodo[0])
        data_fim = pd.to_datetime(periodo[1])
        # Incluir todo o dia final (até 23:59:59)
        data_fim = data_fim + timedelta(hours=23, minutes=59, seconds=59)
        
        df_filtrado = df_filtrado[
            (df_filtrado['Data Início'] >= data_inicio) & 
            (df_filtrado['Data Início'] <= data_fim)
        ]
    except Exception as e:
        st.error(f"Erro ao aplicar filtro de período: {e}")

# MOSTRAR INFORMAÇÕES SOBRE FILTROS - para debug
st.sidebar.info(f"📊 **Registros após filtros:** {len(df_filtrado)}/{len(df)}")

# Verificar se há dados após filtragem
if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhum registro encontrado com os filtros aplicados!")
    st.info("💡 Tente ajustar os filtros ou verificar se os dados possuem as colunas necessárias")

# Cálculo dos KPIs CORRETOS
paradas_fechadas = df_filtrado[df_filtrado['Status'] == 'Fechado']
paradas_abertas = df_filtrado[df_filtrado['Status'] == 'Aberto']

# Verificar se temos dados suficientes para cálculos
dados_suficientes = len(paradas_fechadas) > 0 and 'Tempo de Parada (h)' in df_filtrado.columns

if dados_suficientes:
    # MTTR (CORRETO)
    mttr = paradas_fechadas['Tempo de Parada (h)'].mean()

    # MTBF e Disponibilidade (CORRIGIDOS)
    if len(paradas_fechadas) > 0:
        # Usar todo o dataset filtrado para calcular o período total
        # Isso inclui o último registro, mesmo que seja aberto
        data_minima = df_filtrado['Data Início'].min()
        data_maxima = df_filtrado['Data Início'].max()
        
        # Tempo total do período analisado (considera TODOS os registros)
        tempo_total_periodo = (data_maxima - data_minima).total_seconds() / 3600
        
        # MTBF = Tempo operacional / Número de falhas
        tempo_operacional = tempo_total_periodo - paradas_fechadas['Tempo de Parada (h)'].sum()
        mtbf = tempo_operacional / len(paradas_fechadas) if len(paradas_fechadas) > 0 else 0
        
        # Disponibilidade = Tempo operacional / Tempo total
        disponibilidade = (tempo_operacional / tempo_total_periodo) * 100 if tempo_total_periodo > 0 else 100
        
    else:
        # Caso sem paradas fechadas
        mtbf = 0
        disponibilidade = 100

    # Outros cálculos
    tempo_total_parada = paradas_fechadas['Tempo de Parada (h)'].sum()
    tempo_operacional_calc = tempo_operacional
    
else:
    # Valores padrão quando não há dados suficientes
    mttr = 0
    mtbf = 0
    disponibilidade = 0
    tempo_total_parada = 0
    tempo_operacional_calc = 0

# Total de paradas
total_paradas = len(df_filtrado)
paradas_abertas_count = len(paradas_abertas)

# Exibir KPIs
st.markdown("### 🎯 Visão Geral dos Principais KPIs Manutenção")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MTTR (Horas)", f"{mttr:.2f}", "Tempo Médio para Reparo")
with col2:
    st.metric("MTBF (Horas)", f"{mtbf:.2f}", "Tempo Médio Entre Falhas")
with col3:
    st.metric("Disponibilidade (%)", f"{disponibilidade:.2f}%", "Percentual Operacional")
with col4:
    st.metric("Total de Paradas", total_paradas, f"{paradas_abertas_count} em aberto")

if dados_suficientes:
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Tempo Total Parada (h)", f"{tempo_total_parada:.1f}", "Horas de indisponibilidade")
    with col6:
        eficiencia_manutencao = (1 - (mttr/mtbf)) * 100 if mtbf > 0 else 0
        st.metric("Eficiência Manutenção", f"{eficiencia_manutencao:.1f}%", "MTTR/MTBF")
    with col7:
        taxa_falhas = 1/mtbf if mtbf > 0 else 0
        st.metric("Taxa de Falhas", f"{taxa_falhas:.4f}", "Falhas por hora")
    with col8:
        confiabilidade = np.exp(-tempo_operacional_calc/mtbf) * 100 if mtbf > 0 else 100
        st.metric("Confiabilidade", f"{confiabilidade:.1f}%", "Probabilidade de operação")
else:
    st.warning("⚠️ Dados insuficientes para calcular todos os KPIs. Verifique se existe a coluna 'Tempo de Parada (h)' e paradas fechadas.")

# ANÁLISE DE HORÁRIOS DE PICO - APENAS HORÁRIO ADMINISTRATIVO
st.markdown("---")
st.markdown("### 🕐 Análise de Horários de Pico - Horário Administrativo")



# Estatísticas sobre horário administrativo
total_paradas_admin = len(df_admin)
percentual_admin = (total_paradas_admin / len(df)) * 100 if len(df) > 0 else 0

st.info(f"📊 **{total_paradas_admin} ocorrências ({percentual_admin:.1f}%) em horário administrativo**")

col_pico1, col_pico2 = st.columns(2)

with col_pico1:
    # Distribuição por turno (apenas administrativo)
    paradas_por_turno = df_filtrado[df_filtrado['Turno'] != 'Fora do Expediente']['Turno'].value_counts()
    fig_turno = px.bar(
        x=paradas_por_turno.index,
        y=paradas_por_turno.values,
        title="📊 Paradas por Período do Dia",
        labels={'x': 'Período', 'y': 'Número de Paradas'},
        color=paradas_por_turno.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_turno, use_container_width=True)
    
    # Equipamentos mais problemáticos por turno
    if 'Equipamento' in df_filtrado.columns:
        st.markdown("**🔧 Top 3 Equipamentos Mais Problemáticos por Período:**")
        for turno in paradas_por_turno.index:
            equip_turno = df_filtrado[df_filtrado['Turno'] == turno]['Equipamento'].value_counts().head(3)
            if len(equip_turno) > 0:
                st.write(f"**{turno}:**")
                for i, (equip, count) in enumerate(equip_turno.items(), 1):
                    st.write(f"{i}. {equip} - {count} parada(s)")

with col_pico2:
    # Distribuição por hora do dia (apenas horário administrativo)
    df_admin_filtrado = df_filtrado[df_filtrado['Turno'] != 'Fora do Expediente']
    paradas_por_hora = df_admin_filtrado['Hora'].value_counts().sort_index()
    
    fig_hora = px.bar(
        x=paradas_por_hora.index.astype(str) + ':00',
        y=paradas_por_hora.values,
        title="⏰ Paradas por Hora do Dia (Expediente)",
        labels={'x': 'Hora do Dia', 'y': 'Número de Paradas'},
        color=paradas_por_hora.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_hora, use_container_width=True)
    
    # Horário de pico
    if len(paradas_por_hora) > 0:
        hora_pico = paradas_por_hora.idxmax()
        total_pico = paradas_por_hora.max()
        st.metric("🕐 Horário de Pico", f"{hora_pico:02d}:00", f"{total_pico} paradas")
    
    # Análise de tendência por dia da semana
    df_filtrado['Dia_Semana'] = df_filtrado['Data Início'].dt.day_name()
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dias_portugues = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    paradas_por_dia = df_filtrado['Dia_Semana'].value_counts().reindex(dias_ordem, fill_value=0)
    paradas_por_dia.index = dias_portugues
    
    fig_dia = px.bar(
        x=paradas_por_dia.index,
        y=paradas_por_dia.values,
        title="📅 Paradas por Dia da Semana",
        labels={'x': 'Dia da Semana', 'y': 'Número de Paradas'},
        color=paradas_por_dia.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_dia, use_container_width=True)

# Análise detalhada do horário de pico
st.markdown("#### 🔍 Análise Detalhada do Horário de Pico")

if len(paradas_por_hora) > 0:
    hora_pico = paradas_por_hora.idxmax()
    paradas_pico = df_filtrado[df_filtrado['Hora'] == hora_pico]
    
    col_pico3, col_pico4, col_pico5 = st.columns(3)
    
    with col_pico3:
        # Equipamentos no horário de pico
        if 'Equipamento' in paradas_pico.columns:
            equip_pico = paradas_pico['Equipamento'].value_counts().head(5)
            st.write("**🔧 Equipamentos no Pico:**")
            for equip, count in equip_pico.items():
                st.write(f"- {equip}: {count} parada(s)")
    
    with col_pico4:
        # Locais no horário de pico
        if 'Local' in paradas_pico.columns:
            local_pico = paradas_pico['Local'].value_counts().head(3)
            st.write("**🏭 Locais no Pico:**")
            for local, count in local_pico.items():
                st.write(f"- {local}: {count} parada(s)")
    
    with col_pico5:
        # Causas no horário de pico
        if 'Causa' in paradas_pico.columns:
            causa_pico = paradas_pico['Causa'].value_counts().head(3)
            st.write("**⚡ Causas no Pico:**")
            for causa, count in causa_pico.items():
                st.write(f"- {causa}: {count} parada(s)")

# Análise de padrões temporais
st.markdown("#### 📈 Padrões Temporais das Paradas")

col_temp1, col_temp2 = st.columns(2)

with col_temp1:
    # Paradas por mês
    df_filtrado['Mês'] = df_filtrado['Data Início'].dt.strftime('%Y-%m')
    paradas_por_mes = df_filtrado['Mês'].value_counts().sort_index()
    
    fig_mes = px.line(
        x=paradas_por_mes.index,
        y=paradas_por_mes.values,
        title="📅 Tendência de Paradas por Mês",
        labels={'x': 'Mês', 'y': 'Número de Paradas'},
        markers=True
    )
    st.plotly_chart(fig_mes, use_container_width=True)

with col_temp2:
    # Distribuição por tipo de dia (útil vs final de semana)
    def classificar_tipo_dia(dia):
        if dia in ['Saturday', 'Sunday']:
            return 'Final de Semana'
        else:
            return 'Dia Útil'
    
    df_filtrado['Tipo_Dia'] = df_filtrado['Dia_Semana'].apply(classificar_tipo_dia)
    paradas_por_tipo_dia = df_filtrado['Tipo_Dia'].value_counts()
    
    fig_tipo_dia = px.pie(
        values=paradas_por_tipo_dia.values,
        names=paradas_por_tipo_dia.index,
        title="📊 Paradas: Dia Útil vs Final de Semana"
    )
    st.plotly_chart(fig_tipo_dia, use_container_width=True)

# PIRÂMIDE DE BIRD
st.markdown("---")
st.markdown("### 🏗️ Pirâmide de Bird - Análise de Segurança")

# Dados para a pirâmide (valores baseados na relação clássica 1-3-8-20-600)
piramide_data = {
    'Nível': ['Acidente com Afastamento', 'Acidente sem Afastamento', 
              'Incidente com Danos', 'Quase Acidentes', 'Atos Inseguros'],
    'Quantidade': [1, 3, 8, 20, 600],
    'Cor': ['#FF6B6B', '#FF8E53', '#FFB142', '#FFDA79', '#FFF8E1'],
    'Descrição': [
        'Lesões graves com afastamento',
        'Lesões leves sem afastamento',
        'Danos materiais significativos',
        'Situações que quase resultaram em acidentes',
        'Comportamentos ou condições inseguras'
    ]
}

fig_piramide = go.Figure()

fig_piramide.add_trace(go.Bar(
    y=piramide_data['Nível'],
    x=piramide_data['Quantidade'],
    orientation='h',
    marker_color=piramide_data['Cor'],
    text=piramide_data['Quantidade'],
    textposition='auto',
    hovertemplate='<b>%{y}</b><br>Quantidade: %{x}<br>%{customdata}<extra></extra>',
    customdata=piramide_data['Descrição']
))

fig_piramide.update_layout(
    title="Pirâmide de Bird - Relação de Eventos de Segurança",
    xaxis_title="Quantidade de Ocorrências (escala logarítmica)",
    yaxis_title="Nível de Gravidade",
    showlegend=False,
    height=500,
    xaxis_type="log"
)

st.plotly_chart(fig_piramide, use_container_width=True)

# ANÁLISE DE PARETO - NOVA SEÇÃO ADICIONADA
st.markdown("---")
st.markdown("### 📊 Principais Análise Causas de Parada")

if len(df_filtrado) > 0:
    # Selecionar a coluna para análise de Pareto
    pareto_options = []
    if 'Causa' in df_filtrado.columns:
        pareto_options.append('Causa')
    if 'Equipamento' in df_filtrado.columns:
        pareto_options.append('Equipamento')
    if 'Local' in df_filtrado.columns:
        pareto_options.append('Local')
    
    if pareto_options:
        pareto_category = st.selectbox(
            "Selecione a categoria para análise de Pareto:",
            options=pareto_options,
            index=0
        )
        
        # Criar gráfico de Pareto
        if 'Tempo de Parada (h)' in df_filtrado.columns:
            # Usar tempo de parada como valor
            fig_pareto, pareto_data = create_pareto_chart(
                df_filtrado, 
                pareto_category, 
                'Tempo de Parada (h)', 
                f'Pareto - Tempo de Parada por {pareto_category}',
                height=600
            )
        else:
            # Usar contagem de ocorrências como valor
            fig_pareto, pareto_data = create_pareto_chart(
                df_filtrado, 
                pareto_category, 
                'Status',  # Usaremos qualquer coluna só para contar
                f'Pareto - Número de Paradas por {pareto_category}',
                height=600
            )
            # Ajustar para usar contagem em vez de soma
            pareto_count = df_filtrado[pareto_category].value_counts().reset_index()
            pareto_count.columns = [pareto_category, 'Count']
            pareto_count = pareto_count.sort_values('Count', ascending=False)
            pareto_count['Cumulative Percentage'] = (pareto_count['Count'].cumsum() / pareto_count['Count'].sum() * 100)
            
            fig_pareto.data[0].y = pareto_count['Count']
            fig_pareto.data[1].y = pareto_count['Cumulative Percentage']
        
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Mostrar tabela com dados do Pareto
        with st.expander("📋 Ver dados detalhados do Pareto"):
            st.dataframe(pareto_data)
            
            # Análise 80/20
            if len(pareto_data) > 0:
                eighty_percent_index = pareto_data[pareto_data['Cumulative Percentage'] >= 80].index.min()
                if not pd.isna(eighty_percent_index):
                    top_categories = pareto_data.head(eighty_percent_index + 1)
                    st.write(f"**Princípio 80/20:** {len(top_categories)} categorias representam 80% do total")
                    for i, row in top_categories.iterrows():
                        st.write(f"- {row[pareto_category]}: {row['Cumulative Percentage']:.1f}%")
    else:
        st.warning("ℹ️ Não há colunas adequadas para análise de Pareto (Causa, Equipamento ou Local).")
else:
    st.warning("⚠️ Não há dados para análise de Pareto.")

# Gráficos de análise
st.markdown("---")
st.markdown("### 📊 Análise Detalhada das Paradas")

# Checkbox para mostrar/ocultar gráficos
show_charts = st.checkbox("📈 Mostrar gráficos de análise", value=True)

if show_charts:
    # Gráficos condicionais baseados nas colunas disponíveis
    colunas_disponiveis = df_filtrado.columns

    if 'Local' in colunas_disponiveis:
        col11, col12 = st.columns(2)
        
        with col11:
            # Paradas por Local
            paradas_por_local = df_filtrado['Local'].value_counts()
            fig_local = px.bar(
                x=paradas_por_local.index,
                y=paradas_por_local.values,
                labels={'x': 'Local', 'y': 'Número de Paradas'},
                title="Paradas por Local",
                color=paradas_por_local.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_local, use_container_width=True)

        with col12:
            if 'Equipamento' in colunas_disponiveis:
                # Paradas por Equipamento
                paradas_por_equipamento = df_filtrado['Equipamento'].value_counts()
                fig_equipamento = px.pie(
                    values=paradas_por_equipamento.values,
                    names=paradas_por_equipamento.index,
                    title="Distribuição de Paradas por Equipamento"
                )
                st.plotly_chart(fig_equipamento, use_container_width=True)

    if 'Data Início' in colunas_disponiveis:
        col13, col14 = st.columns(2)
        
        with col13:
            # Tempo de Parada por Mês
            df_filtrado['Mês'] = df_filtrado['Data Início'].dt.to_period('M').astype(str)
            tempo_por_mes = df_filtrado.groupby('Mês')['Tempo de Parada (h)'].sum().reset_index() if 'Tempo de Parada (h)' in colunas_disponiveis else df_filtrado.groupby('Mês').size().reset_index(name='Count')
            fig_tempo_mes = px.line(
                tempo_por_mes,
                x='Mês',
                y='Tempo de Parada (h)' if 'Tempo de Parada (h)' in colunas_disponiveis else 'Count',
                title="Tendência de Paradas por Mês",
                markers=True
            )
            st.plotly_chart(fig_tempo_mes, use_container_width=True)

        with col14:
            if 'Causa' in colunas_disponiveis:
                # Tipo de Manutenção
                def classificar_manutencao(causa):
                    if pd.isna(causa):
                        return "Não Especificada"
                    causa_lower = str(causa).lower()
                    if any(word in causa_lower for word in ['preventiv', 'lavagem', 'programada', 'manutenção preventiva', 'preventiva']):
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

    # Análise de causas
    if 'Causa' in colunas_disponiveis:
        st.markdown("### 🔍 Análise de Causas")
        
        causas_texto = ' '.join(df_filtrado['Causa'].dropna().astype(str))
        palavras_chave = [word for word in causas_texto.lower().split() if len(word) > 4]
        if palavras_chave:
            palavras_frequentes = pd.Series(palavras_chave).value_counts().head(10)
            
            fig_causas = px.bar(
                x=palavras_frequentes.values, 
                y=palavras_frequentes.index,
                orientation='h',
                title="Palavras-chave Mais Frequentes nas Causas",
                labels={'x': 'Frequência', 'y': 'Palavra-chave'},
                color=palavras_frequentes.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_causas, use_container_width=True)



# Análise de Regressão Linear para Previsão de Produtividade

# Análise de Regressão Linear para Previsão de Produtividade

st.markdown("## 🔬 Análise de Regressão Linear para Previsão de Produtividade")

col_treino1, col_treino2 = st.columns(2)
with col_treino1:
    var_alvo = st.selectbox('Selecione a variável de produtividade/alvo:', [col for col in df_filtrado.columns if df_filtrado[col].dtype in [np.float64, np.int64]], key='reg_target')
with col_treino2:
    vars_exp = st.multiselect('Selecione variáveis explicativas:', [col for col in df_filtrado.columns if (df_filtrado[col].dtype in [np.float64, np.int64] and col != var_alvo)], key='reg_features')

if var_alvo and vars_exp:
    df_model = df_filtrado.dropna(subset=[var_alvo]+vars_exp)
    X = df_model[vars_exp].values
    y = df_model[var_alvo].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    st.write(f"**Coeficientes:** {modelo.coef_}")
    st.write(f"**Intercepto:** {modelo.intercept_:.2f}")
    st.write(f"**R² (explicação da regressão):** {modelo.score(X,y):.3f}")

    novo = st.text_input(f'Digite valores para {vars_exp} separados por vírgula para prever produtividade:', key='reg_input')
    if novo:
        try:
            valores = np.array([float(val) for val in novo.split(',')]).reshape(1,-1)
            pred = modelo.predict(valores)
            st.success(f"Produtividade prevista: {pred[0]:.2f}")
        except Exception:
            st.error("Formato inválido! Insira valores numéricos separados por vírgula.")
else:
    st.info("Selecione o alvo e ao menos uma variável explicativa para rodar a regressão.")




# Recomendações finais baseadas na análise de horário administrativo
st.markdown("---")
st.markdown("### 🎯 Recomendações Estratégicas - Horário Administrativo")

# Encontrar turno e hora de pico para recomendações específicas
turno_pico = paradas_por_turno.index[0] if len(paradas_por_turno) > 0 else "Tarde"
hora_pico_val = paradas_por_hora.idxmax() if len(paradas_por_hora) > 0 else 14

recomendacoes = {
    "Prioridade Alta": [
        f"Reforçar manutenção preventiva no período da {turno_pico}",
        f"Atenção especial às {hora_pico_val:02d}:00h (horário de pico)",
        "Implementar checklist matinal antes do início das operações",
        "Treinamento específico para operadores nos horários de maior movimento",
        "Estoque estratégico de peças para falhas mais frequentes"
    ],
    "Prioridade Média": [
        "Rotação de equipamentos durante os horários de pico",
        "Sistema de alerta para manutenção preventiva baseado em horários",
        "Análise mensal de indicadores por turno",
        "Programa de inspeção visual diária"
    ],
    "Prioridade Baixa": [
        "Estudo de redistribuição de carga horária",
        "Benchmark com outras unidades com perfis similares",
        "Implementação de sistema preditivo de manutenção"
    ]
}

for prioridade, itens in recomendacoes.items():
    with st.expander(f"{prioridade}"):
        for item in itens:
            st.write(f"• {item}")

# Tabela com dados detalhados
st.markdown("---")
st.markdown("### 📋 Dados Detalhados das Paradas")

# Mostrar estatísticas dos filtros
st.info(f"**Total de registros carregados:** {len(df)}")
st.success(f"**Registros após filtros:** {len(df_filtrado)}")

# Checkbox para mostrar/ocultar tabela completa
show_full_table = st.checkbox("📊 Mostrar tabela completa de dados", value=False)

if show_full_table:
    # Mostrar todas as colunas e linhas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    st.dataframe(df_filtrado, height=400, use_container_width=True)
    
    # Botão para mostrar informações técnicas
    if st.button("🔍 Mostrar informações técnicas da tabela"):
        st.write(f"**Forma da tabela:** {df_filtrado.shape}")
        st.write(f"**Colunas:** {list(df_filtrado.columns)}")
        st.write(f"**Intervalo de datas:** {df_filtrado['Data Início'].min()} a {df_filtrado['Data Início'].max()}")
else:
    st.write(f"**Total de registros filtrados:** {len(df_filtrado)}")
    st.write("💡 Marque a caixa acima para visualizar a tabela completa")

# Download dos dados filtrados - CORRIGIDO
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, encoding='utf-8', sep=';')

if len(df_filtrado) > 0:
    csv = convert_df_to_csv(df_filtrado)
    
    st.download_button(
        label=f"📥 Baixar dados filtrados ({len(df_filtrado)} registros)",
        data=csv,
        file_name="paradas_manutencao_analise.csv",
        mime="text/csv",
    )
else:
    st.warning("Não há dados para download")

# Informações finais na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 Sobre os KPIs:**

**MTTR (Mean Time To Repair):**
- Tempo médio para reparar uma falha
- Fórmula: Σ(Tempo de reparo) / Nº de reparos
- Meta: Quanto menor, melhor
- Tempo médio de reparo elevado – indica lentidão na resolução 108.10 h

**MTBF (Mean Time Between Failures):**
- Tempo médio entre falhas
- Fórmula: Tempo operacional / Nº de falhas
- Meta: Quanto maior, melhor
- Baixo tempo entre falhas – equipamentos falham com frequência

**Disponibilidade:**
- Percentual de tempo operacional
- Fórmula: (Tempo operacional / Tempo total) × 100
- Meta: >95%
- Muito abaixo do ideal (>95%) – impacto direto na operação

**Eficiência Manutenção:**                
- Eficiência da Manutenção -33.2%
- Processo ineficaz – possíveis gargalos ou falta de recursos

**Taxa de Falhas:**
- Alta frequência de quebras 0.0123 h

**Confiabilidade:**
- Nenhum equipamento operou sem falhas no período 0.0%


**🏗️ Pirâmide de Bird:**
Relação 1-3-8-20-600 mostra que para cada acidente grave há:
- 3 acidentes leves
- 8 incidentes com danos
- 20 quase acidentes
- 600 atos inseguros

**📊 Análise de Pareto:**
Princípio 80/20 onde 20% das causas geram 80% dos problemas
""")

st.sidebar.markdown("---")
st.sidebar.info("📊 **Dashboard desenvolvido para análise de KPIs de manutenção**")