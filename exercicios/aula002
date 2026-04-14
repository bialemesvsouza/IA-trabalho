## **1. Conceitual – Importância da Preparação**

**Por que a etapa de preparação de dados pode impactar mais o desempenho de um modelo do que a escolha do algoritmo?**

A preparação de dados envolve transformar dados brutos em um formato apropriado para o aprendizado. Como os algoritmos dependem da representação dos objetos por meio de vetores de atributos, e não dos objetos em si, a qualidade dessa representação torna-se um fator decisivo. Dessa forma, o pré-processamento melhora significativamente a qualidade dos dados. Quando os dados estão mal preparados, contendo ruídos, valores ausentes, desbalanceamento ou escalas inadequadas, até mesmo algoritmos avançados tendem a apresentar resultados insatisfatórios, reforçando a ideia de que “dados ruins geram resultados ruins”.

---

## **2. Valores Ausentes**

**a) Quais são as possíveis estratégias para tratar esses valores?**

As principais estratégias para lidar com valores ausentes incluem a remoção das instâncias que apresentam dados faltantes, o preenchimento manual com auxílio de especialistas e o preenchimento automático. Este último pode ser realizado por meio da atribuição de valores constantes, como “desconhecido”, pelo uso de medidas estatísticas como média, mediana ou moda, ou ainda por meio de modelos preditivos capazes de estimar os valores ausentes com base nos demais dados disponíveis.

**b) Em quais situações remover registros é inadequado?**

A remoção de registros torna-se inadequada quando a quantidade de dados ausentes é significativa, como em cenários onde uma grande porcentagem do conjunto de dados seria descartada. Além disso, essa prática pode eliminar informações relevantes e introduzir vieses no modelo, comprometendo a qualidade do aprendizado.

---

## **3. Normalização vs Padronização**

**Explique a diferença entre Normalização (Min-Max) e Padronização (Z-score):**

A padronização, também conhecida como Z-score, baseia-se na média e no desvio padrão dos dados, transformando-os de modo que passem a ter média igual a zero e variância igual a um. Esse método é especialmente adequado para dados com distribuição normal. Já a normalização Min-Max consiste em reescalar os valores para um intervalo específico, geralmente entre zero e um, sendo mais indicada quando não se conhece a distribuição dos dados ou quando se deseja limitar os valores a uma faixa definida.

**Em quais tipos de algoritmos cada uma é mais recomendada? Justifique.**

A padronização é recomendada para algoritmos que assumem dados centrados em torno de zero ou que são sensíveis à distância, como o KNN, especialmente em presença de outliers. Por outro lado, a normalização Min-Max é amplamente utilizada em redes neurais, pois valores dentro de um intervalo fixo favorecem a estabilidade e a convergência durante o treinamento.

---

## **4. Dados Categóricos**

**a) Como transformar essa variável (clima = {sol, chuva, nublado}) para uso em um modelo de Machine Learning?**

A variável categórica pode ser transformada por meio da codificação One-Hot, que consiste na criação de novas colunas binárias para cada categoria possível. Nesse caso, seriam criadas colunas como clima_sol, clima_chuva e clima_nublado, nas quais cada instância recebe valor 1 para a categoria correspondente e 0 para as demais.

**b) Qual problema pode ocorrer se utilizarmos apenas codificação numérica simples (0, 1, 2)?**

A utilização de codificação numérica simples pode introduzir uma relação de ordem inexistente entre as categorias, sugerindo, por exemplo, que uma categoria é maior ou mais relevante que outra. Esse tipo de distorção pode prejudicar o aprendizado do modelo, sendo evitado com o uso da codificação One-Hot.

---

## **5. Overfitting e Separação de Dados**

**Explique por que dividimos o dataset em Treino, Validação e Teste:**

A divisão do conjunto de dados é fundamental para garantir uma avaliação adequada do modelo. O conjunto de treino é utilizado para o aprendizado do modelo, o conjunto de validação serve para ajustar hiperparâmetros e orientar melhorias durante o treinamento, enquanto o conjunto de teste é reservado para avaliar o desempenho final do modelo em dados nunca vistos.

**O que acontece se utilizarmos o conjunto de teste durante o treinamento?**

Caso o conjunto de teste seja utilizado durante o treinamento, ocorre o chamado vazamento de dados (data leakage). Nesse cenário, o modelo acaba memorizando os dados em vez de aprender padrões generalizáveis, o que leva ao overfitting. Como consequência, a avaliação do modelo se torna irreal e excessivamente otimista.

---

## **6. Balanceamento de Classes**

**a) Qual problema pode ocorrer (com 95% classe A e 5% classe B)?**

Em situações de desbalanceamento, o modelo tende a favorecer a classe majoritária, resultando em um desempenho aparentemente alto, porém com baixa capacidade de identificar corretamente a classe minoritária. Isso compromete a utilidade prática do modelo, especialmente em problemas críticos.

**b) Cite duas técnicas para tratar esse desbalanceamento:**

Duas abordagens comuns são o undersampling e o oversampling. O undersampling consiste na remoção de instâncias da classe majoritária, enquanto o oversampling envolve o aumento da quantidade de dados da classe minoritária, seja por replicação ou pela geração de novos exemplos sintéticos, como ocorre com técnicas como SMOTE e ADASYN.

---

## **7. Detecção de Outliers**

**a) O que são outliers?**

Outliers são valores atípicos que se diferenciam significativamente dos demais dados, estando fora do padrão esperado da distribuição.

**b) Quando removê-los pode prejudicar o modelo?**

A remoção de outliers pode ser prejudicial quando esses valores representam fenômenos reais importantes, como fraudes ou comportamentos anômalos que o modelo precisa aprender a identificar.

**c) Um método estatístico para detectá-los:**

Entre os métodos mais utilizados estão o Z-score, que identifica valores distantes da média em termos de desvio padrão, e o intervalo interquartil (IQR), que detecta valores fora dos limites considerados normais dentro da distribuição dos dados.

---

## **8. Feature Engineering**

**Explique por que criar a variável idade pode ser melhor do que usar diretamente a data de nascimento.**

A variável idade é numérica e representa diretamente uma informação relevante para a maioria dos modelos, facilitando a identificação de padrões. Em contraste, a data de nascimento é um dado mais complexo, que exige transformações adicionais para se tornar útil. Assim, a criação de novas variáveis a partir das originais contribui para melhorar o desempenho do modelo.

---

## **9. Pipeline de Pré-processamento**

**Descreva um pipeline completo de preparação para um dataset com os problemas listados:**

Um pipeline completo de preparação começa pela limpeza dos dados, tratando valores ausentes por meio de remoção ou imputação adequada. Em seguida, realiza-se a conversão de dados categóricos, geralmente por meio de codificação One-Hot ou ordinal, dependendo do caso. Posteriormente, aplica-se a transformação das variáveis numéricas, utilizando normalização ou padronização para ajustar as escalas. Por fim, são aplicadas técnicas de balanceamento de classes no conjunto de treino, como oversampling ou combinações de métodos, garantindo melhor desempenho do modelo.

---

## **10. Estudo de Caso Aplicado**

**Descreva todas as etapas de preparação antes de treinar o modelo:**

Inicialmente, deve-se realizar a limpeza dos dados, tratando valores ausentes em variáveis como renda mensal e idade, preferencialmente utilizando a mediana para evitar influência de outliers. Em seguida, realiza-se a conversão de variáveis categóricas, como estado civil, por meio de codificação One-Hot, e a transformação de variáveis como histórico de inadimplência, que pode ser convertida em formato ordinal.

Na etapa seguinte, aplica-se a normalização das variáveis numéricas, garantindo que atributos com diferentes magnitudes não prejudiquem o aprendizado do modelo. Depois disso, os dados devem ser divididos em conjuntos de treino e teste, sendo importante que essa divisão ocorra antes de qualquer técnica de balanceamento. Por fim, caso exista desbalanceamento entre as classes, devem ser aplicadas técnicas como oversampling apenas no conjunto de treino, garantindo uma avaliação mais realista do modelo.

