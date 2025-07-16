# Insights da Análise de Regressão - California Housing Dataset

## Análise do Erro Atual

### Situação Atual
- **RMSE obtido**: 0.752 (melhor resultado com polinômios de grau 4)
- **Problema identificado**: O erro é muito alto considerando a escala dos valores target

### Impacto do RMSE = 0.752
- Para casa de $80k (target=0.8): erro de ±$75k (**94% do valor real**)
- Para casa de $100k (target=1.0): erro de ±$75k (**75% do valor real**)
- Para casa de $200k (target=2.0): erro de ±$75k (**37.5% do valor real**)
- Para casa de $400k (target=4.0): erro de ±$75k (**18.8% do valor real**)

**Conclusão**: O modelo tem dificuldade especialmente com casas de menor valor, onde o erro representa uma porcentagem enorme do preço real.

## Problemas Identificados no Código

### 1. Features Subutilizadas
- **Atual**: Usando apenas 3 features (MedInc, HouseAge, AveRooms)
- **Disponível**: 8 features + clusters geográficos criados
- **Correlações importantes não exploradas**:
  - MedInc: 0.69 (sendo usada ✓)
  - AveRooms: 0.15 (sendo usada ✓)
  - HouseAge: 0.11 (sendo usada ✓)
  - Latitude: -0.14 (não usada ✗)
  - Longitude: -0.046 (não usada ✗)

### 2. Hipóteses Preparadas mas Não Testadas
```python
# Estas hipóteses foram criadas mas não testadas com polinômios:
- hipotese3: inclui Latitude/Longitude
- hipotese4: inclui neighborhood_cluster
- hipotese5: todas as features
```

### 3. Instabilidade dos Modelos SGD
- SGD com polinômios apresentou RMSE explosivos (ex: 13363320776912036429824.000)
- Provável causa: `alpha=0.00001` muito baixo para features polinomiais
- Modelos regularizados (Ridge, Lasso, ElasticNet) funcionaram melhor

## Próximos Passos Recomendados

### 1. Explorar Todas as Features (Prioridade Alta)
```python
# Testar hipotese5 com todas as features normalizadas
# Usar apenas modelos regularizados que mostraram estabilidade
modelos_regularizados = {
    'Ridge': Ridge(alpha=0.001),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001)
}
```

### 2. Ajustar Hiperparâmetros
- Fazer grid search para encontrar melhor `alpha` para cada modelo
- Para SGD, testar `alpha` entre 0.01 e 0.1
- Considerar normalização mais robusta (StandardScaler do sklearn)

### 3. Engenharia de Features
- **Interações**: Criar features como `MedInc × neighborhood_cluster`
- **Transformações**: Log ou sqrt em features com distribuição assimétrica
- **Binning**: Criar categorias para features contínuas (ex: faixas de income)

### 4. Algoritmos Alternativos (Após otimizar features)
- **Random Forest**: Captura relações não-lineares sem precisar de polinômios
- **XGBoost/LightGBM**: Estado da arte para dados tabulares
- **Redes Neurais**: Para capturar interações complexas

### 5. Validação Mais Robusta
- Implementar Cross-Validation (K-fold)
- Analisar resíduos por faixa de preço
- Verificar se o erro é homoscedástico

## Ordem Sugerida de Implementação

1. **Primeiro**: Testar hipotese5 (todas features) com modelos regularizados
2. **Segundo**: Otimizar hiperparâmetros via GridSearchCV
3. **Terceiro**: Engenharia de features (interações, transformações)
4. **Quarto**: Apenas se necessário, partir para algoritmos mais complexos

## Métricas Adicionais a Considerar

- **R² Score**: Para entender % da variância explicada
- **MAE (Mean Absolute Error)**: Menos sensível a outliers
- **MAPE (Mean Absolute Percentage Error)**: Para entender erro percentual médio
- **Análise por quartis**: Verificar performance em diferentes faixas de preço

## Código de Referência para Próximos Testes

```python
# Exemplo de teste completo com hipotese5
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Preparar dados
features_zscore = [col for col in hipotese5.columns if 'Zscore' in col and col != 'target']
X = hipotese5[features_zscore]
y = hipotese5['target']

# Validação cruzada
for nome, modelo in modelos_regularizados.items():
    scores = cross_val_score(modelo, X, y, cv=5, 
                           scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"{nome}: RMSE médio = {rmse_scores.mean():.3f} (±{rmse_scores.std():.3f})")
```