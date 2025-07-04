# Explicação Detalhada: Função Hipótese e Função Custo em Machine Learning

## Função Hipótese (h)

A função hipótese é o modelo matemático que tenta prever o valor alvo (target) com base nas features de entrada. No contexto de regressão linear, a hipótese tem a forma:

**Para uma feature:**
```
h(x) = θ₀ + θ₁x
```

**Para múltiplas features:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Onde:
- `h(x)` é o valor previsto
- `θ₀` é o bias (intercepto)
- `θ₁, θ₂, ..., θₙ` são os pesos/coeficientes para cada feature
- `x₁, x₂, ..., xₙ` são as features de entrada

No caso do dataset California Housing, a hipótese tenta prever o preço médio das casas (`target`) usando diferentes combinações de features como renda média (`MedInc`), idade da casa (`HouseAge`), etc.

## Função Custo (J)

A função custo mede o quão erradas estão as previsões do modelo comparadas aos valores reais. A mais comum para regressão linear é o **Erro Quadrático Médio (MSE)**:

```
J(θ) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Onde:
- `m` é o número de exemplos de treino
- `h(x⁽ⁱ⁾)` é a previsão para o i-ésimo exemplo
- `y⁽ⁱ⁾` é o valor real do i-ésimo exemplo
- A soma é feita sobre todos os exemplos de treino
- O símbolo × representa multiplicação

**Objetivo:** Encontrar os valores de θ que minimizam J(θ).

## Como se relacionam no projeto estudoML.ipynb

### 1. Diferentes hipóteses testadas:
- **Hipótese 1:** Usa apenas renda média (modelo mais simples)
- **Hipótese 2:** Adiciona idade da casa e número de quartos
- **Hipótese 3:** Inclui também localização (latitude/longitude)
- **Hipótese 4:** Usa clusters de vizinhança ao invés de coordenadas
- **Hipótese 5:** Usa todas as features disponíveis

### 2. Normalização aplicada:
- **MinMax:** Escala valores entre 0 e 1
- **Z-score:** Padroniza com média 0 e desvio padrão 1

Isso ajuda o algoritmo de otimização a convergir mais rapidamente.

### 3. Processo de aprendizado:
- O modelo ajusta os parâmetros θ para minimizar a função custo
- Usa os dados de treino (70%) para aprender
- Valida performance nos dados de teste (30%)

A ideia é que hipóteses com features mais relevantes terão menor custo e melhor capacidade de generalização.

---

# Processo Detalhado de Minimização da Função Custo

## 1. O Problema de Otimização

O objetivo é encontrar os valores de θ (theta) que minimizam a função custo J(θ). Isso é um problema de otimização onde queremos:

```
min J(θ) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

## 2. Algoritmo do Gradiente Descendente

O método mais comum é o **Gradiente Descendente**, que funciona como descer uma montanha sempre na direção mais íngreme.

### Conceito Intuitivo:
- Imagine que você está no topo de uma montanha (alto custo)
- Quer chegar ao vale (baixo custo)
- A cada passo, olha ao redor e caminha na direção de maior descida
- Repete até chegar ao fundo

### Implementação Matemática:

**Passo 1: Inicialização**
```
θ₀ = 0
θ₁ = 0
θ₂ = 0
...
```

**Passo 2: Calcular Gradientes (Derivadas Parciais)**

Para cada parâmetro θⱼ:
```
∂J/∂θⱼ = (1/m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
```

Onde:
- Para θ₀ (bias): xⱼ⁽ⁱ⁾ = 1
- Para outros θⱼ: xⱼ⁽ⁱ⁾ é o valor da j-ésima feature

**Passo 3: Atualizar Parâmetros**
```
θⱼ := θⱼ - α × (∂J/∂θⱼ)
```

Onde α (alpha) é a **taxa de aprendizado** (learning rate)

## 3. Processo Completo Detalhado

### Exemplo com Uma Feature (Hipótese 1)

```python
# Dados: X = renda média, y = preço da casa
# Modelo: h(x) = θ₀ + θ₁×x

# 1. Inicializar parâmetros
θ₀ = 0
θ₁ = 0
α = 0.01  # taxa de aprendizado
num_iterações = 1000

# 2. Loop de treinamento
para cada iteração:
    # 2.1 Calcular previsões para todos os exemplos
    previsões = θ₀ + θ₁×X
    
    # 2.2 Calcular erros
    erros = previsões - y
    
    # 2.3 Calcular gradientes
    gradiente_θ₀ = (1/m) × soma(erros)
    gradiente_θ₁ = (1/m) × soma(erros × X)
    
    # 2.4 Atualizar parâmetros simultaneamente
    temp_θ₀ = θ₀ - α × gradiente_θ₀
    temp_θ₁ = θ₁ - α × gradiente_θ₁
    
    θ₀ = temp_θ₀
    θ₁ = temp_θ₁
    
    # 2.5 (Opcional) Calcular custo atual
    custo = (1/2m) × soma(erros²)
```

### Exemplo com Múltiplas Features (Hipótese 2-5)

```python
# Dados: X = matriz com n features, y = preços
# Modelo: h(X) = θ₀ + θ₁×x₁ + θ₂×x₂ + ... + θₙ×xₙ

# 1. Inicializar
θ = vetor_zeros(n+1)  # n features + 1 bias
X_com_bias = adicionar_coluna_de_1s(X)  # Para θ₀

# 2. Loop de treinamento
para cada iteração:
    # 2.1 Calcular previsões (produto matricial)
    previsões = X_com_bias × θ
    
    # 2.2 Calcular erros
    erros = previsões - y
    
    # 2.3 Calcular todos os gradientes de uma vez
    gradientes = (1/m) × X_com_bias.T × erros
    
    # 2.4 Atualizar todos os parâmetros
    θ = θ - α × gradientes
```

## 4. Detalhes Importantes da Implementação

### Taxa de Aprendizado (α)
- **Muito pequena**: Convergência lenta, muitas iterações
- **Muito grande**: Pode divergir, nunca encontrar o mínimo
- **Ideal**: Depende dos dados, geralmente entre 0.001 e 0.1

### Critérios de Parada
1. **Número fixo de iterações**: Ex: 1000 iterações
2. **Convergência**: Quando J(θ) para de diminuir significativamente
3. **Gradiente pequeno**: Quando |∂J/∂θ| < ε (epsilon pequeno)

### Normalização é Crucial
Sem normalização:
- Feature 1: Renda (0-15)
- Feature 2: População (0-35000)

O gradiente da população dominaria, causando convergência ruim.

Com normalização (como implementado no notebook):
- Todas as features na mesma escala
- Convergência mais rápida e estável

## 5. Variações do Algoritmo

### Batch Gradient Descent (descrito acima)
- Usa todos os exemplos a cada iteração
- Convergência estável mas lenta para datasets grandes

### Stochastic Gradient Descent (SGD)
```python
para cada época:
    embaralhar_dados()
    para cada exemplo (x⁽ⁱ⁾, y⁽ⁱ⁾):
        gradiente = (h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × x⁽ⁱ⁾
        θ = θ - α × gradiente
```
- Atualiza após cada exemplo
- Mais rápido mas mais "barulhento"

### Mini-Batch Gradient Descent
- Meio termo: usa pequenos lotes (32, 64, 128 exemplos)
- Balanceia velocidade e estabilidade

## 6. Monitoramento do Aprendizado

```python
custos = []
para cada iteração:
    # ... atualizar θ ...
    custo_atual = calcular_J(θ)
    custos.append(custo_atual)

# Plotar curva de aprendizado
plot(custos)
```

A curva deve descer e estabilizar. Se subir, α está muito alto.

## 7. Verificação Final

Após treinar:
1. **Custo no treino**: Deve ser baixo
2. **Custo no teste**: Não deve ser muito maior que treino
3. **Análise de resíduos**: Erros devem ser aleatórios

Esse é o processo completo de como o modelo "aprende" - ajustando iterativamente os parâmetros para minimizar o erro nas previsões!