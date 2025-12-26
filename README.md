# Churn Predict

Este projeto é uma aplicação de Machine Learning para prever a probabilidade de **Churn** (cancelamento) de clientes. Ele inclui um dashboard interativo desenvolvido com [Streamlit](https://streamlit.io/) e um script de modelagem para treinar e avaliar o modelo.

## 📋 Funcionalidades

- **Previsão em Tempo Real:** Insira os dados de um cliente (idade, gênero, contrato, etc.) e receba a probabilidade de churn instantaneamente.
- **Insights do Dataset:** Visualize distribuições de dados, correlações e analise o comportamento dos clientes através de filtros interativos.
- **Relatório do Modelo:** Detalhes sobre o desempenho do modelo, detecção de *data leakage* e testes de robustez.
- **Treinamento Automatizado:** Script para pré-processamento, treinamento (Random Forest) e avaliação do modelo.

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Streamlit** (Dashboard Web)
- **Scikit-learn** (Modelagem e Pré-processamento)
- **Pandas & NumPy** (Manipulação de Dados)
- **Plotly, Matplotlib & Seaborn** (Visualização de Dados)
- **Joblib** (Persistência do Modelo)

## 🚀 Como Executar

### 1. Instalação das Dependências

Certifique-se de ter o Python instalado. É recomendado usar um ambiente virtual. Instale as bibliotecas necessárias:

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn joblib
```

### 2. Executar o Dashboard (App)

Para iniciar a interface web interativa:

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente no seu navegador.

### 3. Treinar o Modelo (Opcional)

Se desejar retreinar o modelo com os dados atuais em `data/`:

```bash
python churn_model.py
```

Isso irá gerar novos arquivos de modelo (`.pkl`), métricas e previsões na pasta `data/`.

## 📂 Estrutura do Projeto

- `app.py`: Código principal da aplicação Streamlit.
- `churn_model.py`: Script responsável pelo treinamento do modelo Random Forest, pré-processamento e geração de arquivos auxiliares.
- `data/`:
    - `customer_churn_dataset-training-master.csv`: Dados de treino.
    - `customer_churn_dataset-testing-master.csv`: Dados de teste.
    - `churn_model.pkl`: Modelo treinado salvo.
    - `encoders.pkl` & `scaler.pkl`: Objetos de pré-processamento salvos.
    - `metrics.json`: Métricas de desempenho do treino.
    - `churn_predictions.csv`: Previsões geradas pelo script de modelagem.

## 📊 Sobre o Modelo

O modelo utiliza um **Random Forest Classifier**. Durante o desenvolvimento, foram identificados e tratados problemas de *Data Leakage* relacionados às variáveis "Support Calls" e "Total Spend", resultando em um modelo final robusto com cerca de **90% de acurácia**, focado na generalização para novos clientes.
