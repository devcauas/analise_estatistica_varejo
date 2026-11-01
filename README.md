# Análise Estatística e Market Basket do Setor de Varejo

Este repositório contém um estudo estatístico e de Machine Learning voltado para o setor de varejo (incluindo dados do mercado brasileiro e transações). O objetivo do projeto é extrair *insights* de vendas, analisar tendências do setor e encontrar regras de associação entre produtos (FP-Growth), que viram recomendações a partir do carrinho.

---

## Funcionalidades e Estrutura do Código

* **`analise_exploratoria.py`**: Análise inicial de dados (EDA), distribuição de vendas, identificação de outliers e estatísticas descritivas das transações.
* **`analise_setor.py`**: Avaliação de desempenho por segmento/categoria do varejo e comportamento ao longo do tempo.
* **`modelo_ML.py`**: Market basket analysis com FP-Growth no setor Pharmacy: itemsets frequentes, regras de associação avaliadas por lift e função de recomendação por produto.
* **`modelo_varejo_brasil.py`**: Regras de associação (FP-Growth) e grafo de produtos (NetworkX) com os dados do varejo brasileiro.
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
* **Machine Learning:** mlxtend, NetworkX

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
