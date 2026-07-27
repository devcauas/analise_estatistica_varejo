# Análise Estatística e Preditiva do Setor de Varejo

Este repositório contém um estudo estatístico e de Machine Learning voltado para o setor de varejo (incluindo dados do mercado brasileiro e transações). O objetivo do projeto é extrair *insights* de vendas, analisar tendências do setor e construir modelos preditivos para tomada de decisão.

---

## Funcionalidades e Estrutura do Código

* **`analise_exploratoria.py`**: Análise inicial de dados (EDA), distribuição de vendas, identificação de outliers e estatísticas descritivas das transações.
* **`analise_setor.py`**: Avaliação de desempenho por segmento/categoria do varejo e comportamento ao longo do tempo.
* **`modelo_ML.py`**: Algoritmo de Machine Learning voltado para predição (ex: previsão de demanda, classificação de clientes ou estimativa de vendas).
* **`modelo_varejo_brasil.py`**: Modelagem e análise focada nas especificidades e dados do mercado varejista brasileiro.
* Exemplo:
transacao_id,item
1,atum enlatado
1,cerveja lata
1,geleia
(...)

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.13.5
* **Análise de Dados:** Pandas, NumPy
* **Visualização de Dados:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn

---

## Como Obter os Dados

Devido ao limite de tamanho do GitHub, os conjuntos de dados não estão versionados no repositório. Para rodar o projeto localmente, baixe os arquivos e salve-os dentro da pasta `data/`:

1. **`Retail_Transactions_Dataset.csv`**: https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset
2. **`varejo_brasil_long.csv`**: Dataset sintético/simulado para análise e modelagem do mercado brasileiro.
---

## Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/devcauas/analise_estatistica_varejo.git](https://github.com/devcauas/analise_estatistica_varejo.git)
   cd analise_estatistica_varejo
