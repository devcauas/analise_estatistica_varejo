"""
- [Algoritmo FP-Growth]  Você pode pesquisar sobre **Algoritmo Apriori, Algoritmo FP-Growth (Frequent Pattern Growth) e Algoritmo Eclat.**
- [ ]  É importante que você crie métricas para avaliar o modelo e que você explique o porquê escolheu determinada técnica.
- [ ]  Lembre-se de criar elementos visuais para a suas análises, como gráficos e tabelas.
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from itertools import combinations
from collections import Counter

# %%
# ==============================================================
# 1. CARREGAR E FILTRAR DADOS
# ==============================================================
df_original = pd.read_csv("data/Retail_Transactions_Dataset.csv")
setor_escolhido = 'Pharmacy'
df = df_original[df_original['Store_Type'] == setor_escolhido].copy()

print(f"Total de transações no setor {setor_escolhido}: {len(df)}")
df.head()

# %%
# ==============================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================

# Transformar a string de produtos em lista limpa
transactions = df['Product'].str.split(',').apply(
    lambda items: [item.strip().strip("[]'\" ") for item in items]
).tolist()

# Verificar resultado da limpeza
print("Exemplo de 3 transações após limpeza:")
for t in transactions[:3]:
    print(f"  {t}")

# Distribuição de itens por transação
tamanhos = [len(t) for t in transactions]
print(f"\nMédia de itens por transação: {sum(tamanhos)/len(tamanhos):.2f}")
print(f"Mínimo: {min(tamanhos)} | Máximo: {max(tamanhos)}")

# %%
# ==============================================================
# 3. ANÁLISE EXPLORATÓRIA DOS PRODUTOS
# ==============================================================

# Produtos mais frequentes individualmente
todos_itens = [item for transacao in transactions for item in transacao]
print(f"Total de produtos únicos: {len(set(todos_itens))}")
print(f"Total de transações: {len(transactions)}")

contagem = Counter(todos_itens)
print("\nTop 10 produtos mais frequentes:")
for produto, count in contagem.most_common(10):
    print(f"  {produto}: {count} ({count/len(transactions)*100:.2f}%)")

# %%
# Gráfico — Top 15 produtos mais frequentes
top_produtos = pd.DataFrame(contagem.most_common(15), columns=['Produto', 'Frequência'])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_produtos, x='Frequência', y='Produto', palette='Blues_r')
plt.title(f'Top 15 Produtos Mais Frequentes — {setor_escolhido}', fontsize=14)
plt.xlabel('Frequência')
plt.ylabel('Produto')
plt.tight_layout()
plt.show()

# %%
# ==============================================================
# 4. ANÁLISE DE PARES — DIAGNÓSTICO
# ==============================================================

pares = []
for transacao in transactions:
    if len(transacao) >= 2:
        for par in combinations(sorted(transacao), 2):
            pares.append(par)

contagem_pares = Counter(pares)
print(f"Total de pares únicos encontrados: {len(contagem_pares)}")

suporte_maximo = contagem_pares.most_common(1)[0][1] / len(transactions)
print(f"Suporte máximo real dos pares: {suporte_maximo:.4f} ({suporte_maximo*100:.3f}%)")

print(f"\nTop 10 pares mais frequentes:")
for par, count in contagem_pares.most_common(10):
    suporte = count / len(transactions)
    print(f"  {par}: {count} vezes | suporte: {suporte:.4f} ({suporte*100:.3f}%)")

# %%
# ==============================================================
# 5. MODELO FP-GROWTH
# ==============================================================

# One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Dimensões da matriz: {df_transformed.shape}")
print(f"  → {df_transformed.shape[0]} transações × {df_transformed.shape[1]} produtos")

# %%
# Aplicar FP-Growth
# min_support=0.002 foi definido após análise dos pares:
# o suporte máximo real dos pares é ~0.26%, então 0.2% captura
# as associações mais relevantes sem gerar ruído.
frequent_itemsets = fpgrowth(df_transformed, min_support=0.002, use_colnames=True)
frequent_itemsets['tamanho'] = frequent_itemsets['itemsets'].apply(len)

print(f"Total de itemsets frequentes: {len(frequent_itemsets)}")
print(f"  → Individuais: {len(frequent_itemsets[frequent_itemsets['tamanho'] == 1])}")
print(f"  → Pares:       {len(frequent_itemsets[frequent_itemsets['tamanho'] == 2])}")

# %%
# ==============================================================
# 6. GERAR REGRAS DE ASSOCIAÇÃO
# ==============================================================

# Usando confidence como métrica de entrada e avaliando por lift
# Motivo: com dados sintéticos, o lift fica próximo de 1.0 e
# filtrar diretamente por lift rejeitaria todas as regras válidas.
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.01
)

print(f"Total de regras geradas: {len(rules)}")
print(f"\nTop 10 regras por lift:")
print(
    rules.sort_values('lift', ascending=False)
    .head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .to_string(index=False)
)

# %%
# ==============================================================
# 7. VISUALIZAÇÕES DAS REGRAS
# ==============================================================

# Gráfico — Support vs Confidence colorido por Lift
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    rules['support'],
    rules['confidence'],
    c=rules['lift'],
    cmap='RdYlGn',
    alpha=0.7,
    edgecolors='k',
    linewidths=0.3
)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Suporte')
plt.ylabel('Confiança')
plt.title(f'Regras de Associação — {setor_escolhido}\nSupport vs Confidence (cor = Lift)', fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Top 10 regras por lift em gráfico de barras
top_rules = rules.sort_values('lift', ascending=False).head(10).copy()
top_rules['regra'] = (
    top_rules['antecedents'].apply(lambda x: ', '.join(list(x))) +
    ' → ' +
    top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_rules, x='lift', y='regra', palette='Greens_r')
plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (independente)')
plt.title(f'Top 10 Regras por Lift — {setor_escolhido}', fontsize=13)
plt.xlabel('Lift')
plt.ylabel('Regra')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ==============================================================
# 8. FUNÇÃO DE RECOMENDAÇÃO
# ==============================================================

def recomendar_produto(produto, rules, top_n=3):
    """
    Dado um produto, retorna os top_n produtos recomendados
    com base nas regras de associação geradas pelo FP-Growth.
    """
    recomendacoes = rules[
        rules['antecedents'].apply(lambda x: produto in x)
    ].sort_values('lift', ascending=False).head(top_n)

    if recomendacoes.empty:
        return f"Nenhuma recomendação encontrada para '{produto}'"

    print(f"\nQuem compra '{produto}' também tende a comprar:")
    for _, row in recomendacoes.iterrows():
        consequente = ', '.join(list(row['consequents']))
        print(f"  → {consequente} | confiança: {row['confidence']:.2%} | lift: {row['lift']:.3f}")

# Teste com os produtos mais frequentes
for produto in ['Toothpaste', 'Soap', 'Deodorant']:
    recomendar_produto(produto, rules)
#%%

"""
A MATEMÁTICA POR TRÁS DAS MÉTRICAS

- support(X) = transações com X / total de transações

Exemplo:
  Toothpaste aparece em 12.282 de 166.915 transações
  support(Toothpaste) = 12.282 / 166.915 = 0.0736 (7.36%)


  
- confidence(X → Y) = support(X U Y) / support(X)

Exemplo:
  support(Toothpaste U Soap) = 0.0026
  support(Toothpaste)        = 0.0736
  confidence = 0.0026 / 0.0736 = 0.035 (3.5%)

  → De quem compra Toothpaste, 3.5% também compra Soap



- lift(X → Y) = confidence(X → Y) / support(Y)

lift = 1.0 → independentes (coincidência)
lift > 1.0 → associação real positiva ✅
lift < 1.0 → associação negativa ❌

Exemplo:
  confidence(Toothpaste → Soap) = 0.035
  support(Soap)                 = 0.0379
  lift = 0.035 / 0.0379 = 0.923 → levemente negativo
"""