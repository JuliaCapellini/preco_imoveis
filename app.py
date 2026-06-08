import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Previsão de Preços de Imóveis",
    page_icon="🏠",
    layout="centered"
)

# ── Carregamento do modelo ──────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    return joblib.load("model/modelo_final.pkl")

modelo = carregar_modelo()

# ── Título e descrição ──────────────────────────────────────────────────────
st.title("🏠 Previsão de Preços de Imóveis")
st.markdown(
    "Preencha as características do imóvel abaixo e clique em **Prever Preço** "
    "para obter uma estimativa do valor de venda."
)
st.divider()

# ── Campos de entrada ───────────────────────────────────────────────────────
st.subheader("Características do Imóvel")

col1, col2 = st.columns(2)

with col1:
    GrLivArea = st.number_input(
        "Área construída acima do solo (sq ft)",
        min_value=300, max_value=6000, value=1500, step=50,
        help="Área habitável acima do nível do solo em pés quadrados."
    )
    TotalBsmtSF = st.number_input(
        "Área do porão (sq ft)",
        min_value=0, max_value=3000, value=800, step=50,
        help="Área total do porão em pés quadrados."
    )
    FirstFlrSF = st.number_input(
        "Área do 1º andar (sq ft)",
        min_value=300, max_value=4000, value=900, step=50,
        help="Área do primeiro andar em pés quadrados."
    )
    SecondFlrSF = st.number_input(
        "Área do 2º andar (sq ft)",
        min_value=0, max_value=2000, value=0, step=50,
        help="Área do segundo andar (0 se não houver)."
    )

with col2:
    OverallQual = st.slider(
        "Qualidade geral do imóvel (1–10)",
        min_value=1, max_value=10, value=6,
        help="Avaliação geral do material e acabamento do imóvel."
    )
    GarageCars = st.selectbox(
        "Capacidade da garagem (nº de carros)",
        options=[0, 1, 2, 3, 4],
        index=2,
        help="Quantidade de carros que a garagem comporta."
    )
    ExterQual = st.selectbox(
        "Qualidade do revestimento externo",
        options=["Ex", "Gd", "TA", "Fa"],
        index=2,
        format_func=lambda x: {"Ex": "Ex – Excelente", "Gd": "Gd – Bom",
                                "TA": "TA – Médio", "Fa": "Fa – Regular"}[x],
        help="Qualidade do material de revestimento exterior."
    )

st.divider()
st.subheader("Localização e Condição de Venda")

col3, col4 = st.columns(2)

with col3:
    Neighborhood = st.selectbox(
        "Bairro (Neighborhood)",
        options=[
            'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
            'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
            'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
            'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
        ],
        index=5,
        help="Localização do imóvel dentro dos limites de Ames, Iowa."
    )

with col4:
    SaleCondition = st.selectbox(
        "Condição da venda",
        options=['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'],
        index=0,
        format_func=lambda x: {
            'Normal': 'Normal – Venda comum',
            'Abnorml': 'Abnorml – Execução hipotecária / curta',
            'Partial': 'Partial – Casa não concluída',
            'AdjLand': 'AdjLand – Compra de terreno adjacente',
            'Alloca': 'Alloca – Dois imóveis com escrituras separadas',
            'Family': 'Family – Venda entre familiares'
        }[x],
        help="Condição da venda."
    )

st.divider()

# ── Botão de predição ───────────────────────────────────────────────────────
if st.button("🔍 Prever Preço", use_container_width=True, type="primary"):

    # Calculando TotalSF a partir das entradas
    TotalSF = TotalBsmtSF + FirstFlrSF + SecondFlrSF

    # Montando o DataFrame com as mesmas colunas usadas no treino
    entrada = pd.DataFrame([{
        'TotalSF': TotalSF,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'OverallQual': OverallQual,
        'ExterQual': ExterQual,
        'Neighborhood': Neighborhood,
        'SaleCondition': SaleCondition
    }])

    # Predição (o modelo já inclui o preprocessor dentro do pipeline)
    preco_log = modelo.predict(entrada)[0]
    preco_real = np.expm1(preco_log)

    # ── Exibição do resultado ───────────────────────────────────────────────
    st.success(f"### 💰 Preço estimado: **$ {preco_real:,.0f}**")

    # Interpretação simples
    if preco_real < 100_000:
        faixa = "abaixo de $100.000 — imóvel de baixo custo."
    elif preco_real < 200_000:
        faixa = "entre $100.000 e $200.000 — faixa econômica intermediária."
    elif preco_real < 300_000:
        faixa = "entre $200.000 e $300.000 — faixa intermediária a alto padrão."
    elif preco_real < 500_000:
        faixa = "entre $300.000 e $500.000 — imóvel de alto padrão."
    else:
        faixa = "acima de $500.000 — imóvel de luxo."

    st.info(
        f"📊 **Interpretação:** O modelo estima um imóvel {faixa} "
        f"Com base nas características fornecidas (área total de {TotalSF} sq ft, "
        f"qualidade {OverallQual}/10 e {GarageCars} vaga(s) de garagem)."
    )

    with st.expander("Ver dados utilizados na predição"):
        st.dataframe(entrada)

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Projeto desenvolvido para a disciplina Machine Learning – UNIMAR | Grupo 2 · "
    "Dataset: Kaggle House Prices Advanced Regression Techniques"
)
