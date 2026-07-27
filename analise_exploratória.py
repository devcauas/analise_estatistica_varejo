"""
**Análise Exploratória**

Vamos responder algumas perguntas de negócios; essas respostas poderão ajudar os gerentes e diretores a tomar ações que aumentem as vendas ou resolvam algum gargalo no negócio.

- [X]  Qual o total de itens vendidos (somar a quantidade de todos os itens).
- [X]  Qual o valor total de vendas.
- [X]  Quantos itens cada **Store_type** vendeu e quantos % representa do total de itens vendidos
- [X]  Qual o total de custo por **Store_type** e qual o % do total de custos.
- [X]  Qual método de pagamento é mais usado e qual é menos usado.
- [X]  Qual o método escolhido para as compras mais caras.
- [X]  Qual a quantidade de vendas por hora.
- [X]  Qual a quantidade de vendas por dia da semana.
- [X]  O dia e hora com mais venda por cidade.
"""

#%%

import pandas as pd

df = pd.read_csv("data/Retail_Transactions_Dataset.csv")

df.head()

#%%

df["Total_Items"].sum()
#%%

df["Total_Cost"].sum()
#%%

vendas_por_tipo = df.groupby('Store_Type')['Total_Items'].sum().reset_index()

total_absoluto = vendas_por_tipo['Total_Items'].sum()
vendas_por_tipo['%_Representacao'] = (vendas_por_tipo['Total_Items'] / total_absoluto) * 100

print(vendas_por_tipo)
print(vendas_por_tipo['%_Representacao'].sum())
# %%

vendas_por_custo = df.groupby('Store_Type')['Total_Cost'].sum().reset_index()

total_absoluto = vendas_por_custo['Total_Cost'].sum()
vendas_por_custo['%_Representacao'] = (vendas_por_custo['Total_Cost'] / total_absoluto) * 100

print(vendas_por_custo)
print(vendas_por_custo['%_Representacao'].sum())

# %%

pagamento_mais_usado = df['Payment_Method'].value_counts().idxmax()
print(f"Pagamento mais usado: {pagamento_mais_usado}")

pagamento_menos_usado = df['Payment_Method'].value_counts().idxmin()
print(f"Pagamento menos usado: {pagamento_menos_usado}")
# %%

limiar = df['Total_Cost'].quantile(0.75)
compras_caras = df[df['Total_Cost'] >= limiar]

metodo_escolhido = compras_caras['Payment_Method'].value_counts().idxmax()

print(f"O método mais usado para compras acima de R${limiar} é: {metodo_escolhido}")

# %%

df['Date'] = pd.to_datetime(df['Date'])

df['hora'] = df['Date'].dt.hour

vendas_por_hora = df.groupby('hora')['Total_Cost'].sum().mean()

print(f"{vendas_por_hora:,.2f} vendas por hora")

# %%

df['Date'] = pd.to_datetime(df['Date'])

df['hora'] = df['Date'].dt.day_name()

vendas_por_dia_semanal = df.groupby('hora')['Total_Cost'].sum()

vendas_por_dia_semanal

# %%

df['Date'] = pd.to_datetime(df['Date'])

df['Week_Day'] = df['Date'].dt.day_name() 
df['Hour'] = df['Date'].dt.hour

vendas_agrupadas = df.groupby(['City', 'Week_Day', 'Hour'])['Total_Items'].sum().reset_index()

indices_maximos = vendas_agrupadas.groupby('City')['Total_Items'].idxmax()
melhores_momentos = vendas_agrupadas.loc[indices_maximos]

print(melhores_momentos)

# %%
