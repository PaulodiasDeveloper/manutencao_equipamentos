# 📊 Dashboard de KPIs de Manutenção com Pirâmide de Bird

Um dashboard interativo desenvolvido em **Streamlit** para análise completa de indicadores de manutenção, incluindo a clássica **Pirâmide de Bird** para análise de segurança.

---

## ✨ Funcionalidades

- 📈 **Análise de KPIs**: MTTR, MTBF, Disponibilidade, Eficiência e outros indicadores  
- 🕐 **Análise de Horários de Pico**: Identificação de períodos críticos no horário administrativo  
- 🏗️ **Pirâmide de Bird**: Visualização da relação de eventos de segurança (1-3-8-20-600)  
- 📊 **Análise de Pareto**: Identificação das principais causas usando o princípio 80/20  
- 🔧 **Filtros Avançados**: Filtragem por local, equipamento, status, turno e período  
- 📤 **Upload de Dados**: Suporte a arquivos Excel (.xlsx, .xls)  
- 📥 **Exportação de Resultados**: Download dos dados filtrados em CSV  

---

## 🚀 Como Usar

### Pré-requisitos
- Python 3.7+  
- pip (gerenciador de pacotes Python)  

### Instalação
Clone o repositório:

```bash
git clone <url-do-repositorio>
cd dashboard-manutencao
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Executando a Aplicação

```bash
streamlit run app.py
```

O dashboard estará disponível em **http://localhost:8501**

---

## 📋 Estrutura do Arquivo de Dados

Seu arquivo Excel deve conter pelo menos as seguintes colunas:

| Coluna             | Obrigatório | Descrição                                |
|--------------------|-------------|------------------------------------------|
| Data Início        | ✅           | Data e hora do início da parada          |
| Status             | ✅           | Status da manutenção (ex: "Aberto")      |
| Data Fim           | ❌           | Data e hora do fim da parada             |
| Tempo de Parada (h)| ❌           | Duração da parada em horas               |
| Local              | ❌           | Localização do equipamento               |
| Equipamento        | ❌           | Nome/identificação do equipamento        |
| Causa              | ❌           | Causa da parada                          |

### Exemplo de Estrutura

| Data Início         | Data Fim           | Local        | Equipamento          | Causa                     | Status   |
|---------------------|--------------------|--------------|----------------------|---------------------------|----------|
| 2025-05-05 09:00:00 | 2025-05-05 15:00:00| AGR Cabiúnas | Empilhadeira 2.5 ton | Freio de mão travado      | Fechado  |
| 2025-05-12 08:30:00 | 2025-05-13 09:50:00| AGR Cabiúnas | Empilhadeira 4 ton   | Cabo de bateria com folga | Fechado  |

---

## 📊 KPIs Calculados

- **MTTR (Mean Time To Repair)**: Tempo médio para reparo  
- **MTBF (Mean Time Between Failures)**: Tempo médio entre falhas  
- **Disponibilidade**: Percentual de tempo operacional  
- **Eficiência da Manutenção**: Relação MTTR/MTBF  
- **Taxa de Falhas**: Falhas por hora de operação  
- **Confiabilidade**: Probabilidade de operação sem falhas  

---

## 🏗️ Pirâmide de Bird

A aplicação inclui a visualização da **Pirâmide de Bird**, mostrando a relação clássica:

- 1 Acidente com Afastamento  
- 3 Acidentes sem Afastamento  
- 8 Incidentes com Danos  
- 20 Quase Acidentes  
- 600 Atos Inseguros  

---

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web em Python  
- **Pandas**: Manipulação e análise de dados  
- **Plotly**: Visualizações interativas  
- **NumPy**: Computação numérica  
- **OpenPyXL**: Leitura de arquivos Excel  

---

## 📦 Instalação das Dependências

```bash
pip install streamlit pandas numpy plotly openpyxl
```

Ou use o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🎯 Funcionalidades de Análise

- **Análise Temporal**: Tendências por mês, dia da semana e hora  
- **Distribuição por Turno**: Manhã, Almoço, Tarde e Final de Expediente  
- **Análise por Equipamento**: Identificação dos equipamentos mais problemáticos  
- **Análise por Local**: Distribuição geográfica das paradas  
- **Análise de Causas**: Palavras-chave mais frequentes nas descrições  

---

## 📝 Recomendações Estratégicas

O dashboard gera recomendações automáticas baseadas na análise dos dados, categorizadas por prioridade:

- **Alta**: Ações críticas para horários de pico  
- **Média**: Melhorias processuais e de gestão  
- **Baixa**: Otimizações e estudos futuros  

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto  
2. Criar uma branch para sua feature  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commitar suas mudanças  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Fazer push para a branch  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Abrir um Pull Request  

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 📞 Suporte

Para dúvidas ou problemas:

- Verifique se o arquivo de dados segue a estrutura exigida  
- Confirme que todas as dependências estão instaladas  
- Abra uma **issue** no repositório com detalhes do problema  

---

💡 Desenvolvido para otimizar a gestão de manutenção e melhorar a disponibilidade de equipamentos 🚀
