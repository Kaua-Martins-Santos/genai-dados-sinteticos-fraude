# Dados sintéticos com IA generativa para datasets desbalanceados

Estudo de caso de um curso de **IA Generativa aplicada à área de Dados**, executado e documentado por mim.

## O problema
Detecção de fraude em cartão de crédito com o dataset público [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): **284.807 transações e só 492 fraudes (0,17%)**. Um modelo treinado nesses dados aprende pouco sobre a classe que importa.

## A abordagem
1. Treinar um **baseline** (Regressão Logística) nos dados originais.
2. Usar o **Gemini** com *few-shot prompting* (5 fraudes reais como exemplo) para gerar transações fraudulentas sintéticas.
3. Fazer o parse da resposta do LLM para um DataFrame, juntar ao treino e **retreinar**.
4. Comparar no mesmo conjunto de teste.

## Resultado (classe fraude, conjunto de teste com 147 fraudes)

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| Baseline | 0,76 | 0,63 | 0,69 |
| + 11 fraudes sintéticas (Gemini) | 0,80 | 0,67 | 0,73 |

## Limitações (leitura honesta)
- Uma única execução, sem `random_state` fixo — a diferença pode estar dentro da variação entre execuções.
- Apenas 11 exemplos sintéticos; os valores gerados têm padrões repetidos, sinal de que o LLM não captura bem a distribuição real das features (PCA).
- O baseline não convergiu (`max_iter=100`, dados sem normalização). Normalizar e usar `class_weight='balanced'` ou SMOTE seriam comparações justas antes de atribuir o ganho ao dado sintético.

## Rodando
Abra o notebook no Colab, salve sua chave do [Google AI Studio](https://aistudio.google.com/app/apikey) em **Secrets** como `GOOGLE_API_KEY` e rode as células.

> O notebook usa o SDK `google-generativeai` e o modelo `gemini-1.5-flash`, como no material original — ambos já foram descontinuados. Para rodar hoje, migre para o pacote `google-genai`.

## Stack
Python · pandas · scikit-learn · seaborn · Gemini API
