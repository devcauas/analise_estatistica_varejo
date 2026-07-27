# %%
# ==============================================================
# MARKET BASKET ANALYSIS — FP-GROWTH
# Varejo Brasil | Versão Refatorada
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# ==============================================================
# 0. CONFIGURAÇÕES GLOBAIS
# ==============================================================

CONFIG = {
    "min_support":    0.002,   # ~0.2% das transações
    "min_confidence": 0.05,    # Filtro pós-geração mais rigoroso
    "min_lift":       1.1,     # Apenas associações acima do aleatório
    "min_leverage":   0.0001,  # Associação real (não espúria)
    "top_n_rules":    10,      # Regras exibidas nos prints
    "top_n_produtos": 15,      # Produtos no gráfico de frequência
}

PALETTE = "Blues_r"

# %%
# ==============================================================
# 1. CARGA E INSPEÇÃO DOS DADOS
# ==============================================================

df = pd.read_csv("data/varejo_brasil_long.csv")

print("=== Informações Gerais ===")
print(df.info())
print("\n=== Estatísticas Descritivas ===")
print(df.describe())

print("\n=== Itens por Transação ===")
print(df.groupby("transacao_id")["item"].count().describe())

# %%
# ==============================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================

df_agrupado = (
    df.groupby("transacao_id")["item"]
    .apply(list)
    .reset_index()
)

todos_itens = [item for transacao in df_agrupado["item"] for item in transacao]
contagem = Counter(todos_itens)

n_transacoes = len(df_agrupado)
n_produtos   = len(set(todos_itens))

print(f"Total de transações:     {n_transacoes:,}")
print(f"Total de produtos únicos: {n_produtos:,}")

# %%
# ==============================================================
# 3. ANÁLISE EXPLORATÓRIA — FREQUÊNCIA DE PRODUTOS
# ==============================================================

top_produtos = pd.DataFrame(
    contagem.most_common(CONFIG["top_n_produtos"]),
    columns=["Produto", "Frequência"]
)
top_produtos["% Transações"] = top_produtos["Frequência"] / n_transacoes * 100

print(f"\nTop {CONFIG['top_n_produtos']} produtos mais frequentes:")
print(top_produtos.to_string(index=False))

# Gráfico
fig, ax = plt.subplots(figsize=(12, 6))
bars = sns.barplot(
    data=top_produtos, x="Frequência", y="Produto",
    palette=PALETTE, ax=ax
)
for bar, (_, row) in zip(ax.patches, top_produtos.iterrows()):
    ax.text(
        bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
        f"{row['% Transações']:.1f}%", va="center", fontsize=9
    )

ax.set_title(
    f"Top {CONFIG['top_n_produtos']} Produtos Mais Frequentes — {n_transacoes:,} transações",
    fontsize=14
)
ax.set_xlabel("Frequência")
ax.set_ylabel("Produto")
plt.tight_layout()
plt.show()

# %%
# ==============================================================
# 4. ENCODING E FP-GROWTH
# ==============================================================

te = TransactionEncoder()
te_ary = te.fit(df_agrupado["item"]).transform(df_agrupado["item"])
df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Matriz one-hot: {df_transformed.shape[0]:,} transações × {df_transformed.shape[1]} produtos")

# FP-Growth
frequent_itemsets = fpgrowth(
    df_transformed,
    min_support=CONFIG["min_support"],
    use_colnames=True
)
frequent_itemsets["tamanho"] = frequent_itemsets["itemsets"].apply(len)

print(f"\nItemsets frequentes encontrados: {len(frequent_itemsets):,}")
for t in sorted(frequent_itemsets["tamanho"].unique()):
    n = len(frequent_itemsets[frequent_itemsets["tamanho"] == t])
    label = {1: "Individuais", 2: "Pares", 3: "Triplas"}.get(t, f"Tamanho {t}")
    print(f"  → {label}: {n:,}")

# %%
# ==============================================================
# 5. GERAÇÃO E FILTRAGEM DAS REGRAS
# ==============================================================

# Gerar todas as regras com threshold mínimo de confiança
rules_raw = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=CONFIG["min_confidence"]
)

print(f"Regras brutas geradas: {len(rules_raw):,}")

# Filtro combinado: lift + confidence + leverage
# - lift > 1.1 → associação acima do esperado pelo acaso
# - leverage > 0 → co-ocorrência real, não explicada pelas frequências individuais
rules = rules_raw[
    (rules_raw["lift"]      > CONFIG["min_lift"]) &
    (rules_raw["leverage"]  > CONFIG["min_leverage"])
].copy()

print(f"Regras após filtros (lift > {CONFIG['min_lift']}, leverage > {CONFIG['min_leverage']}): {len(rules):,}")

# Colunas legíveis
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

# %%
# ==============================================================
# 6. TOP REGRAS POR DIFERENTES MÉTRICAS
# ==============================================================

metricas = ["lift", "confidence", "leverage", "conviction"]

for metrica in metricas:
    if metrica not in rules.columns:
        continue
    print(f"\n{'='*60}")
    print(f"Top {CONFIG['top_n_rules']} regras por {metrica.upper()}")
    print(f"{'='*60}")
    print(
        rules.sort_values(metrica, ascending=False)
        .head(CONFIG["top_n_rules"])[
            ["antecedents_str", "consequents_str",
             "support", "confidence", "lift", "leverage", "conviction"]
        ]
        .rename(columns={
            "antecedents_str": "SE comprou",
            "consequents_str": "ENTÃO comprou",
        })
        .to_string(index=False)
    )

# %%
# ==============================================================
# 7. VISUALIZAÇÕES
# ==============================================================

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# --- 7a. Support vs Confidence (cor = Lift) ---
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(
    rules["support"], rules["confidence"],
    c=rules["lift"], cmap="RdYlGn",
    alpha=0.7, edgecolors="k", linewidths=0.3, s=60
)
plt.colorbar(sc, ax=ax1, label="Lift")
ax1.set_xlabel("Suporte")
ax1.set_ylabel("Confiança")
ax1.set_title("Support vs Confidence\n(cor = Lift)")

# --- 7b. Distribuição de Lift ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(rules["lift"], bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
ax2.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Lift = 1 (aleatório)")
ax2.axvline(rules["lift"].mean(), color="orange", linestyle="-", linewidth=1.2,
            label=f"Média = {rules['lift'].mean():.3f}")
ax2.set_xlabel("Lift")
ax2.set_ylabel("Contagem de regras")
ax2.set_title("Distribuição do Lift")
ax2.legend(fontsize=9)

# --- 7c. Distribuição de Conviction ---
ax3 = fig.add_subplot(gs[1, 0])
conviction_vals = rules["conviction"].replace([float("inf")], rules["conviction"][
    rules["conviction"] != float("inf")].max() * 1.2)
ax3.hist(conviction_vals, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
ax3.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="Conviction = 1")
ax3.set_xlabel("Conviction (valores inf. substituídos)")
ax3.set_ylabel("Contagem de regras")
ax3.set_title("Distribuição da Conviction")
ax3.legend(fontsize=9)

# --- 7d. Top 10 regras por Lift (barras horizontais) ---
ax4 = fig.add_subplot(gs[1, 1])
top_lift = rules.sort_values("lift", ascending=False).head(10).copy()
top_lift["label"] = top_lift["antecedents_str"] + " → " + top_lift["consequents_str"]
top_lift = top_lift.sort_values("lift")
ax4.barh(top_lift["label"], top_lift["lift"], color="#FF5722", edgecolor="white")
ax4.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax4.set_xlabel("Lift")
ax4.set_title("Top 10 Regras por Lift")
ax4.tick_params(axis="y", labelsize=8)

plt.suptitle(
    f"Análise de Regras de Associação — {n_transacoes:,} transações | {len(rules):,} regras",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.show()

# %%
# ==============================================================
# 8. GRAFO DE ASSOCIAÇÕES (Top regras por lift)
# ==============================================================

def plot_grafo_associacoes(rules, top_n=15, min_lift=1.0):
    """Plota um grafo direcionado das top regras de associação."""
    top = rules.sort_values("lift", ascending=False).head(top_n)

    G = nx.DiGraph()
    for _, row in top.iterrows():
        for ant in row["antecedents"]:
            for cons in row["consequents"]:
                G.add_edge(ant, cons, weight=row["lift"], confidence=row["confidence"])

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    edges     = G.edges(data=True)
    lifts     = [d["weight"] for _, _, d in edges]
    lift_norm = [(l - min(lifts)) / (max(lifts) - min(lifts) + 1e-9) for l in lifts]

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#1565C0",
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=[plt.cm.YlOrRd(v) for v in lift_norm],
        width=[1.5 + 2.5 * v for v in lift_norm],
        arrows=True, arrowsize=20,
        connectionstyle="arc3,rad=0.1", ax=ax
    )

    sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                                norm=plt.Normalize(vmin=min(lifts), vmax=max(lifts)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Lift", shrink=0.7)

    ax.set_title(f"Grafo de Associações — Top {top_n} regras por Lift", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

plot_grafo_associacoes(rules, top_n=20)

# %%
# ==============================================================
# 9. FUNÇÃO DE RECOMENDAÇÃO — CARRINHO MÚLTIPLO
# ==============================================================

def recomendar_produtos(
    carrinho: list[str],
    rules: pd.DataFrame,
    top_n: int = 5,
    metrica: str = "lift"
) -> pd.DataFrame:
    """
    Recomenda produtos com base em um carrinho com 1 ou mais itens.

    Parâmetros
    ----------
    carrinho : list[str]
        Lista de produtos já no carrinho do cliente.
    rules : pd.DataFrame
        DataFrame de regras de associação gerado pelo mlxtend.
    top_n : int
        Número de recomendações a retornar.
    metrica : str
        Métrica de ordenação: 'lift', 'confidence', 'leverage' ou 'conviction'.

    Retorna
    -------
    pd.DataFrame com as top_n recomendações e suas métricas.
    """
    carrinho_set = set(carrinho)

    # Filtra regras cujo antecedente tem interseção com o carrinho
    # e cujo consequente não repete itens já no carrinho
    mask = rules.apply(
        lambda row: (
            len(carrinho_set & row["antecedents"]) > 0
            and len(carrinho_set & row["consequents"]) == 0
        ),
        axis=1
    )

    candidatas = rules[mask].copy()

    if candidatas.empty:
        print(f"⚠️  Nenhuma recomendação encontrada para o carrinho: {carrinho}")
        return pd.DataFrame()

    resultado = (
        candidatas
        .sort_values(metrica, ascending=False)
        .drop_duplicates(subset="consequents_str")
        .head(top_n)[["antecedents_str", "consequents_str",
                      "support", "confidence", "lift", "leverage"]]
        .rename(columns={
            "antecedents_str": "Base",
            "consequents_str": "Recomendação",
        })
        .reset_index(drop=True)
    )

    print(f"\n🛒 Carrinho: {carrinho}")
    print(f"📦 Top {top_n} recomendações (ordenadas por {metrica}):\n")
    print(resultado.to_string(index=False))
    return resultado


# %%
# ==============================================================
# 10. TESTES DA FUNÇÃO DE RECOMENDAÇÃO
# ==============================================================

# Item único
recomendar_produtos(["leite integral"], rules, top_n=5)
recomendar_produtos(["pão"], rules, top_n=5)

# Carrinho com múltiplos itens
top3 = [p for p, _ in contagem.most_common(3)]
recomendar_produtos(top3, rules, top_n=5, metrica="confidence")

# %%
# ==============================================================
# 11. RESUMO FINAL DAS MÉTRICAS
# ==============================================================

print("\n" + "="*55)
print("RESUMO FINAL")
print("="*55)
print(f"  Transações analisadas : {n_transacoes:,}")
print(f"  Produtos únicos       : {n_produtos:,}")
print(f"  Itemsets frequentes   : {len(frequent_itemsets):,}")
print(f"  Regras brutas         : {len(rules_raw):,}")
print(f"  Regras após filtros   : {len(rules):,}")
print(f"\n  Métricas das regras filtradas:")
for col in ["support", "confidence", "lift", "leverage"]:
    print(f"    {col:<12} | min={rules[col].min():.4f} | "
          f"média={rules[col].mean():.4f} | max={rules[col].max():.4f}")
# %%
