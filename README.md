# Previsão de Preços de Imóveis

Projeto desenvolvido para a disciplina de Machine Learning – Teoria e Aplicado da UNIMAR (1/2026).

# Integrantes

- Gabriele Martinez 1991045
- Julia Capellini 1994186
- Lucas Gimenez 1996567

# Descrição do Problema

A precificação de imóveis é uma das tarefas mais relevantes do mercado imobiliário. O valor de venda de uma casa depende de múltiplos fatores: tamanho, qualidade de acabamento, localização, número de garagens, entre outros, tornando a estimativa manual imprecisa e subjetiva. Este projeto propõe o uso de modelos de Machine Learning para automatizar e tornar mais precisa essa estimativa.

# Objetivo do Projeto

Prever o preço de venda de imóveis residenciais, com base em suas características estruturais e de localização, utilizando técnicas de regressão supervisionada.

# Dataset Utilizado 

- **Fonte:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Arquivo principal:** train.csv (1.460 registros, 81 variáveis)

# Tipo de Problema de Machine Learning

**Regressão supervisionada:** o modelo aprende a partir de exemplos rotulados (imóveis com preço conhecido) para prever o preço de imóveis novos.

# Metodologia

- **Análise Exploratória (EDA):** histograma do SalePrice, scatter plot de área vs. preço, boxplot de qualidade vs. preço, heatmap de correlação e identificação de outliers.
- **Pré-processamento:** remoção de outliers (GrLivArea > 4.000 e SalePrice < 300.000), criação da feature TotalSF, transformação logarítmica (log1p) na variável alvo, Pipeline com ColumnTransformer para imputação e encoding, divisão 60/20/20.
- **Treinamento e Validação:** três modelos treinados com validação cruzada K-Fold (k=5).
- **Tuning:** GridSearchCV aplicado ao Random Forest com busca nos hiperparâmetros n_estimators, max_depth e min_samples_split.
- **Avaliação Final:** modelo otimizado avaliado no conjunto de teste isolado, com análise de resíduos por faixa de preço.
- **Deploy:** modelo salvo em .pkl e disponibilizado via aplicação Streamlit.

# Modelos Treinados

- Linear Regression
- Ridge
- Random Forest

# Modelo Final Escolhido

**Random Forest otimizado via GridSearchCV**, salvo como pipeline completo (preprocessor + modelo) em model/modelo_final.pkl.
Apesar de a Regressão Linear ter apresentado melhores métricas na validação, o Random Forest foi escolhido para a versão final por ser mais robusto a não-linearidades e menos sensível à heterocedasticidade observada nos resíduos da Regressão Linear. Após o tuning, o modelo otimizado foi avaliado no conjunto de teste isolado, confirmando boa capacidade de generalização.

# Métricas de Avaliação

- **MAE (Mean Absolute Error):** erro absoluto médio em dólares.
- **RMSE (Root Mean Squared Error):** penaliza erros grandes, mesma unidade do preço.
- **R² (Coeficiente de Determinação):** proporção da variância do preço explicada pelo modelo.
- **MAPE (Mean Absolute Percentage Error):** erro percentual médio, facilita interpretação relativa.

# Principais Resultados

- A Regressão Linear obteve o melhor R² na validação: 0.9159 (91,59% da variância explicada).
- O modelo erra mais em imóveis de alto valor (acima de $300.000), que são menos representados no treino.
- Nas faixas entre $100.000 e $300.000, os resíduos ficam próximos de zero sem padrão sistemático.
- As variáveis de maior impacto no preço são: OverallQual, TotalSF e GrLivArea.
- Entre as variáveis de localização, bairros como StoneBr e Veenker elevam o preço estimado, enquanto IDOTRR e BrDale o reduzem.

# Estrutura dos Arquivos

preco-imoveis/

│

├── app.py                          # Aplicação Streamlit

├── requirements.txt                # Dependências do projeto

├── README.md                       # Documentação do projeto

│

├── notebooks/

│   └── Dprecos_imoveis_corrigido.ipynb  # Notebook revisado (P2)

│

├── model/

│   └── modelo_final.pkl            # Pipeline final salvo (joblib)

│

├── reports/

│   └── relatorio_atualizado.pdf    # Relatório final atualizado

│

└── data/

    └── train.csv            # Arquivo do dataset


# Tecnologias Utilizadas

- Python 3.11
- pandas — manipulação de dados
- numpy — operações numéricas e transformações
- matplotlib / seaborn — visualização de dados
- scikit-learn — modelos, pipelines, métricas e validação cruzada
- joblib — serialização do modelo
- Streamlit — interface web do aplicativo
- GitHub — versionamento e hospedagem do projeto
- Streamlit Community Cloud — deploy da aplicação

# Instruções para executar o notebook

- Clone o repositório:

git clone https://github.com/JuliaCapellini/preco_imoveis.git

cd previsao-precos-imoveis

- Instale as dependências:

pip install -r requirements.txt

- Coloque o arquivo train.csv dentro da pasta data/.

- Abra o notebook:

jupyter notebook notebooks/Desafio_ML_melhorado.ipynb

- Execute todas as células em ordem. A última célula salva automaticamente o modelo em model/modelo_final.pkl.

# Instruções para executar o app Streamlit

- Certifique-se de que o arquivo model/modelo_final.pkl existe (gerado ao rodar o notebook).

- Na raiz do projeto, execute:

streamlit run app.py

- O app abrirá automaticamente no navegador em http://localhost

# Link do App Publicado

[Previsão de Preços de Imóveis](https://precoimoveis-ml-ads-g2.streamlit.app/)

# Limitações

- O modelo foi treinado exclusivamente com dados de Ames, Iowa (EUA), não é diretamente transferível para outros mercados imobiliários.
- O app utiliza apenas 7 variáveis como entrada, enquanto o dataset original possui 81. Isso simplifica a experiência do usuário, mas reduz a precisão em relação ao modelo completo.
- Imóveis com características muito fora do intervalo de treino (ex.: acima de $500.000) tendem a apresentar erros de predição maiores.

# Conclusão

O projeto demonstrou que é possível prever o preço de imóveis com boa precisão utilizando técnicas de regressão supervisionada. A Regressão Linear apresentou o melhor desempenho na validação (R² de 0.9159), enquanto o Random Forest otimizado foi adotado como modelo final por sua maior robustez. A análise de resíduos por faixa de preço revelou que o modelo é mais preciso em imóveis de valor intermediário e que o principal desafio está na predição de imóveis de luxo. A aplicação Streamlit traduz o modelo técnico em uma ferramenta acessível para estimativa de preços.
