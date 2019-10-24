# coding=utf-8
"""
Created on Tue Jul 16 10:47:18 2019
Usando pandas para buscar informações da binance.com
@author: Felipe Soares
"""
import requests        #Para fazer solicitações http para binance
import json            #para analisar o que a binance envia de volta para nós
import pandas as pd    #Para armazenar e manipular os dados que recebemos de volta
import numpy as np     

import matplotlib.pyplot as plt # for charts and such

import datetime as dt  # for dealing with times




#Cria URL da API Binance
root_url = 'https://api.binance.com/api/v1/klines'

symbol = 'BTCUSDT'

interval = '1h'

url = root_url + '?symbol=' + symbol + '&interval=' + interval
print(url)


def get_bars(symbol, interval = '1h'):
   url = root_url + '?symbol=' + symbol + '&interval=' + interval
   data = json.loads(requests.get(url).text)
   df = pd.DataFrame(data)
   df.columns = ['open_time',
                 'o', 'h', 'l', 'c', 'v',
                 'close_time', 'qav', 'num_trades',
                 'taker_base_vol', 'taker_quote_vol', 'ignore']
   df.index = [dt.datetime.fromtimestamp(x/1000.0) for x in df.close_time]
   return df


btcusdt = get_bars('BTCUSDT')



btcusdt = btcusdt['c'].astype('float') 



btcusdt.plot()



