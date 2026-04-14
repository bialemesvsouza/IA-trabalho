import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

url = "https://docs.google.com/spreadsheets/d/1g1aQ61vijh6uHJuc8sijeBEMsoIQ2a5yLwUK04Wptlg/export?format=csv"
df = pd.read_csv(url)

mapping = {
    "Você ficou gripado no ano passado ?": "gripe",
    "Você tomou vacina da gripe no ano passado?": "vacina",
    "  Você frequentou no ano passado,  semanalmente ambientes com muitas pessoas? (salas cheias, ônibus, eventos, etc.)  ": "ambientes_cheios",
    "  Você viajou no ano passado mais de 100 km de distância?  ": "viajou",
    "  Você tem alergia nas vias aéreas (rinite, sinusite, etc.)?  ": "alergia",
    "Quantas horas você dormiu em média por noite no ano passado?": "horas_sono",
    "Você praticou atividade física no ano passado?": "exercicio",
    "Você se alimentou de forma balanceada no ano passado?": "alimentacao",
    "Em média, quantas vezes você lavou as mãos por dia no ano passado?": "lavagem_maos",
    "Na sua percepção, o seu nível de estresse no ano passado foi:": "estresse"
}
df_cleaned = df.rename(columns=mapping).drop(columns=["Carimbo de data/hora"], errors='ignore').dropna()

print("Executando Algoritmo de Regras de Associação...")

df_regras = pd.get_dummies(df_cleaned.astype(str)).astype(bool)

frequent_itemsets = apriori(df_regras, min_support=0.2, use_colnames=True)

if not frequent_itemsets.empty:
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    
    regras_gripe = regras[regras['consequents'].apply(lambda x: any('gripe' in item for item in x))]
    
    display_cols = ['antecedents', 'consequents', 'support', 'confidence']
    print("\nRegras de Associação (Focadas na Gripe):")
    print(regras_gripe[display_cols].head())
else:
    print("Nenhum padrão frequente encontrado com este suporte.")