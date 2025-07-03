# 🎯 Exemplos Práticos de Machine Learning

## Objetivo
Implementar os conceitos do Módulo 2 em exemplos práticos para consolidar o aprendizado antes de avançar para o Módulo 3.

## 📋 Exemplos Recomendados

### **1. Regressão Linear Simples**
- **Dataset**: Preços de casas vs. área
- **Objetivo**: Implementar do zero (sem sklearn)
- **Praticar**: 
  - Função de custo
  - Descida do gradiente
  - Plotagem de resultados
  - Verificação de convergência

### **2. Regressão Linear Múltipla**
- **Dataset**: Preços de casas (área + quartos + banheiros)
- **Foco**: Vetorização com NumPy, feature scaling
- **Praticar**:
  - Implementação vetorizada
  - Normalização de características
  - Múltiplas variáveis
  - Comparação de performance

### **3. Regressão Polinomial**
- **Dataset**: Dados com curva não-linear
- **Objetivo**: Criar características polinomiais
- **Praticar**:
  - Engenharia de características
  - Detecção de overfitting
  - Dimensionamento crítico
  - Visualização de curvas

## 📚 Datasets Sugeridos

### **Opção 1: Datasets Prontos**
- **California Housing** (sklearn.datasets)
- **Boston Housing** (clássico para regressão)
- **Diabetes Dataset** (sklearn.datasets)

### **Opção 2: Dados Sintéticos**
- Gerar com NumPy usando funções conhecidas
- Adicionar ruído controlado
- Testar diferentes níveis de complexidade

## 🛠 Estrutura de Implementação Recomendada

### **Passo 1: Preparação dos Dados**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar dados
# 2. Explorar visualmente
# 3. Verificar dimensões
```

### **Passo 2: Preprocessamento**
```python
# 1. Normalização/Padronização
# 2. Divisão treino/teste
# 3. Criação de características (se necessário)
```

### **Passo 3: Implementação do Algoritmo**
```python
# 1. Função de custo
# 2. Cálculo do gradiente
# 3. Descida do gradiente
# 4. Função de predição
```

### **Passo 4: Treinamento**
```python
# 1. Inicialização dos parâmetros
# 2. Loop de treinamento
# 3. Monitoramento da convergência
# 4. Salvamento de métricas
```

### **Passo 5: Avaliação**
```python
# 1. Predições no conjunto de teste
# 2. Cálculo de métricas (MSE, MAE, R²)
# 3. Visualização dos resultados
# 4. Comparação com sklearn
```

## 💡 Dicas Importantes

### **Implementação**
1. **Sempre implemente do zero primeiro** - isso solidifica o entendimento
2. **Use vetorização** - evite loops explícitos
3. **Monitore convergência** - plote J(θ) vs iterações
4. **Normalize características** - especialmente para múltiplas variáveis

### **Debugging**
1. **Teste com dados simples** - comece com 1D
2. **Verifique gradientes** - compare com diferenças finitas
3. **Plot early and often** - visualize cada passo
4. **Compare com sklearn** - valide sua implementação

### **Boas Práticas**
1. **Documente seu código** - explique cada função
2. **Use nomes descritivos** - evite variáveis como `x1`, `x2`
3. **Modularize** - separe em funções reutilizáveis
4. **Teste diferentes hiperparâmetros** - taxa de aprendizado, iterações

## 🎯 Ordem de Implementação Sugerida

### **Semana 1: Fundamentos**
1. Regressão linear simples com dados sintéticos
2. Implementar função de custo e gradiente
3. Testar diferentes taxas de aprendizado

### **Semana 2: Expansão**
1. Regressão linear múltipla
2. Implementar vetorização
3. Praticar feature scaling

### **Semana 3: Complexidade**
1. Regressão polinomial
2. Engenharia de características
3. Detecção de overfitting

## 📊 Métricas para Avaliar

### **Durante o Treinamento**
- Convergência da função de custo
- Tempo de execução
- Número de iterações necessárias

### **Pós-Treinamento**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Visualização das predições vs. valores reais

## 🔗 Recursos Adicionais

### **Bibliotecas Essenciais**
- `numpy` - Operações vetorizadas
- `pandas` - Manipulação de dados
- `matplotlib` - Visualização
- `sklearn` - Comparação e datasets

### **Referências**
- Documentação oficial do NumPy
- Matplotlib tutorials
- Sklearn documentation
- Coursera Module 2 materials

## 📝 Checklist de Progresso

- [ ] Implementar regressão linear simples
- [ ] Implementar função de custo
- [ ] Implementar descida do gradiente
- [ ] Adicionar monitoramento de convergência
- [ ] Implementar regressão múltipla
- [ ] Adicionar feature scaling
- [ ] Implementar vetorização
- [ ] Testar regressão polinomial
- [ ] Comparar com sklearn
- [ ] Documentar aprendizados

---

**Lembre-se**: O objetivo é entender profundamente os conceitos antes de usar bibliotecas prontas. A implementação do zero é fundamental para o aprendizado sólido em ML!