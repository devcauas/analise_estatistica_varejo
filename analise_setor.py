"""
- [X]  Você vai escolher um **Store_type** e vai fazer as próximas análises no setor escolhido
- [X]  Filtre ou recorte o dataset com o setor da sua escolha

Na coluna **Product** temos mais de um item vendido por transação; você vai separar esses itens e agrupar por item. Queremos saber:
    -[X]  Qual item no seu setor vendeu mais (apareceu com mais frequência)
    -[X]  Qual item no seu setor vendeu menos (apareceu com menos frequência)
    -[X]  Qual item cada perfil de cliente compra mais
"""

#%%
import pandas as pd

df_original = pd.read_csv("data/Retail_Transactions_Dataset.csv")

df_original.head()
#%%

setor_escolhido = 'Pharmacy'

df = df_original[df_original['Store_Type'] == setor_escolhido].copy()

df.head()

# %%

df.tail()
# %%

df.value_counts()
# %%

item_mais_vendido = df['Product'].value_counts().idxmax()
item_mais_vendido

# %%

item_menos_vendido = df['Product'].value_counts().idxmin()
item_menos_vendido
# %%

perfil_cliente = df.groupby('Customer_Category')['Product'].agg(lambda x: x.value_counts().idxmax())
perfil_cliente
# %%
