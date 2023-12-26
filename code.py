import yfinance as yF
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#formatando valores com duas casas decimais
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings("ignore")
Cotacoes = yF.Ticker("ITUB3.SA")
dados = Cotacoes.history(period="5y")
dados.head(50)
dados.reset_index(inplace=True)
dados.drop('Dividends', axis=1, inplace=True)
dados.drop('Stock Splits', axis=1, inplace=True)
dados.columns = ['Data','Abertura','Maximo','Minimo','Fechamento','Volume']
dados.head()
print('Menor data: ', dados['Data'].min())
print('Maior data:', dados['Data'].max())
display(dados.loc[dados.index.max()])
plt.plot(dados["Fechamento"])
plt.title("Preço Diário de Fechamento das Ações", size = 14)
plt.show()
dados['mm5d'] = dados['Fechamento'].rolling(5).mean()
dados['mm14d'] = dados['Fechamento'].rolling(14).mean()
dados['mm21d'] = dados['Fechamento'].rolling(21).mean()
qtd_linhas = len(dados)
qtd_linhas_treino = qtd_linhas - 400
qtd_linhas_teste = qtd_linhas - 20
qtd_linhas_validacao = qtd_linhas_treino - qtd_linhas_teste
info = (
    f"linhas treino = 0:{qtd_linhas_treino}"
    f" linhas teste = 0:{qtd_linhas_treino}:{qtd_linhas_teste}"
    f" linhas validacao = 0:{qtd_linhas_teste}:{qtd_linhas}"
)
info
preditoras = dados.drop(['Data', 'Fechamento','Volume'], 1)
target = dados['Fechamento']
scaler = MinMaxScaler().fit(preditoras)
preditoras_normalizadas = scaler.transform(preditoras)
X_Train = preditoras_normalizadas[:qtd_linhas_treino]
X_test = preditoras_normalizadas[qtd_linhas_treino:qtd_linhas_teste]
Y_Train = target[:qtd_linhas_treino]
Y_test = target[qtd_linhas_treino:qtd_linhas_teste]
print(len(X_Train), len(Y_Train))
print(len(X_test), len(Y_test))
lr = linear_model.LinearRegression()
lr.fit(X_Train, Y_Train )
predicao = lr.predict(X_test)
cd = r2_score(Y_test, predicao)
f'Coeficiente de determinação:{cd * 100:.2f}'
rn = MLPRegressor(max_iter = 2000)
rn.fit(X_Train, Y_Train )
predicao = rn.predict(X_test)
cd = rn.score(X_test,Y_test)
f'Coeficiente de determinação:{cd * 100:.2f}'
previsao = preditoras_normalizadas[qtd_linhas_teste:qtd_linhas]
data_pregao_full = dados['Data']
data_pregao = data_pregao_full[qtd_linhas_teste:qtd_linhas]
res_full = dados['Fechamento']
res = res_full[qtd_linhas_teste:qtd_linhas]
pred = lr.predict(previsao)
df = pd.DataFrame({'Data_Pregão':data_pregao, 'Real': res, 'Previsão':pred})
df.set_index('Data_Pregão', inplace = True)
plt.figure(figsize = (16,8))
plt.title('Preço das Ações')
plt.plot(df['Real'], label = 'Real', color = 'blue', marker = 'o')
plt.plot(df['Previsão'], label = 'Previsão', color = 'red', marker = 'o')
plt.xlabel('Data Pregão')
plt.ylabel('Preço Fechamento')
leg = plt.legend()
