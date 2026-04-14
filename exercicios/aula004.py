import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

data = [
    ['infantil', 'miopia', 'não', 'reduzida', 'nenhuma'],
    ['infantil', 'miopia', 'sim', 'normal', 'gelatinosa'],
    ['infantil', 'hipermetropia', 'não', 'normal', 'gelatinosa'],
    ['infantil', 'hipermetropia', 'sim', 'normal', 'dura'],
    ['adolescente', 'miopia', 'não', 'reduzida', 'gelatinosa'],
    ['adolescente', 'miopia', 'sim', 'reduzida', 'nenhuma'],
    ['adolescente', 'miopia', 'não', 'normal', 'dura'],
    ['adolescente', 'hipermetropia', 'não', 'reduzida', 'gelatinosa'],
    ['adolescente', 'hipermetropia', 'sim', 'normal', 'dura'],
    ['adulto', 'miopia', 'não', 'normal', 'gelatinosa'],
    ['adulto', 'miopia', 'sim', 'normal', 'dura'],
    ['adulto', 'miopia', 'sim', 'normal', 'gelatinosa'],
    ['adulto', 'hipermetropia', 'não', 'reduzida', 'nenhuma'],
    ['adulto', 'hipermetropia', 'sim', 'normal', 'gelatinosa'],
    ['adulto', 'hipermetropia', 'não', 'normal', 'gelatinosa']
]

columns = ['Idade', 'Diagnostico', 'Astigmatismo', 'Taxa_Lacrimal', 'Classe']
df = pd.DataFrame(data, columns=columns)

print("--- EXERCÍCIO AULA 004: REDES BAYESIANAS ---")
print("Previsão para paciente: Idade=infantil, Diagnostico=hipermetropia, Astigmatismo=não, Taxa=reduzida\n")

paciente_teste = {
    'Idade': 'infantil',
    'Diagnostico': 'hipermetropia',
    'Astigmatismo': 'não',
    'Taxa_Lacrimal': 'reduzida'
}

nb_model = BayesianNetwork([
    ('Classe', 'Idade'),
    ('Classe', 'Diagnostico'),
    ('Classe', 'Astigmatismo'),
    ('Classe', 'Taxa_Lacrimal')
])

nb_model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=df.shape[0])

nb_infer = VariableElimination(nb_model)
pred_nb = nb_infer.query(variables=['Classe'], evidence=paciente_teste)

print("=== RESULTADO NAIVE BAYES ===")
print(pred_nb)


rb_model = BayesianNetwork([
    ('Classe', 'Idade'),
    ('Classe', 'Diagnostico'),
    ('Classe', 'Astigmatismo'),
    ('Classe', 'Taxa_Lacrimal'),
    ('Idade', 'Taxa_Lacrimal') 
])

rb_model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=df.shape[0])

rb_infer = VariableElimination(rb_model)
pred_rb = rb_infer.query(variables=['Classe'], evidence=paciente_teste)

print("\n=== RESULTADO REDE BAYESIANA ===")
print(pred_rb)