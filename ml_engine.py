import os
import time
import math
import pandas as pd
import numpy as np

# ========================================================
# FIX DO MATPLOTLIB: Força ele a trabalhar sem interface 
# gráfica (Agg) para não travar a Thread do Flask!
# ========================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import Callback

os.makedirs('logs', exist_ok=True)

class KerasWebLogger(Callback):
    def __init__(self, epochs, log_queue, log_history):
        super().__init__()
        self.total_epochs = epochs
        self.log_queue = log_queue
        self.log_history = log_history
        self.epoch_start = 0

    def emit(self, raw_msg, html_msg):
        self.log_history.append(html_msg) 
        self.log_queue.put(html_msg)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        self.emit(f"Epoch {epoch+1}/{self.total_epochs}\n", f"<span class='text-slate-100 font-bold'>Epoch {epoch+1}/{self.total_epochs}</span>\n")

    def on_epoch_end(self, epoch, logs=None):
        dur = max(time.time() - self.epoch_start, 0.001)
        steps = self.params.get('steps', 142)
        ms_step = (dur / steps) * 1000
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        raw = f"{steps}/{steps} ━━━━━━━━━━━━━━━━━━━━ {int(dur)}s {int(ms_step)}ms/step - loss: {loss:.4f} - val_loss: {val_loss:.4f}\n"
        html = f"{steps}/{steps} <span class='text-emerald-500'>━━━━━━━━━━━━━━━━━━━━</span> {int(dur)}s {int(ms_step)}ms/step - loss: <span class='text-amber-400'>{loss:.4f}</span> - val_loss: <span class='text-sky-400'>{val_loss:.4f}</span>\n"
        self.emit(raw, html)

def run_training(params, log_queue):
    log_history = []
    
    def emit_log(raw, html=None):
        if html is None: html = raw
        log_history.append(html)
        log_queue.put(html)

    try:
        emit_log("[1] Carregando os dados do dataset...\n", "<span class='text-sky-400'>[1] Carregando os dados do dataset...</span>\n")
        df_gen = pd.read_csv('dataset/Plant_1_Generation_Data.csv')
        df_weather = pd.read_csv('dataset/Plant_1_Weather_Sensor_Data.csv')

        emit_log("[2] Processando datas e agrupando...\n", "<span class='text-sky-400'>[2] Processando datas e agrupando...</span>\n")
        df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], dayfirst=True)
        df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
        df_gen_plant = df_gen.groupby('DATE_TIME').agg({'DC_POWER': 'sum'}).reset_index()
        df_merged = pd.merge(df_gen_plant, df_weather, on='DATE_TIME', how='inner')
        df_merged = df_merged.sort_values('DATE_TIME').reset_index(drop=True)
        df_merged['HOUR'] = df_merged['DATE_TIME'].dt.hour
        
        features = ['DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR']
        df_final = df_merged[features]

        emit_log("[3] Normalizando os dados (MinMaxScaler)...\n", "<span class='text-sky-400'>[3] Normalizando os dados (MinMaxScaler)...</span>\n")
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler_X.fit_transform(df_final)
        y_scaled = scaler_y.fit_transform(df_final[['DC_POWER']])

        TIME_STEPS = int(params.get('janela', 8))
        HORIZONTE = int(params.get('horizonte', 1))
        
        emit_log(f"[4] Aplicando Janela Deslizante (Passos: {TIME_STEPS}, Horizonte: {HORIZONTE})...\n", f"<span class='text-sky-400'>[4] Aplicando Janela Deslizante (Passos: {TIME_STEPS}, Horizonte: {HORIZONTE})...</span>\n")

        def create_dataset(X, y, time_steps, horizonte):
            Xs, ys = [], []
            for i in range(len(X) - time_steps - horizonte + 1):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps + horizonte - 1])
            return np.array(Xs), np.array(ys)

        X_data, y_data = create_dataset(X_scaled, y_scaled, TIME_STEPS, HORIZONTE)
        train_size = int(len(X_data) * 0.8)
        X_train, X_test = X_data[:train_size], X_data[train_size:]
        y_train, y_test = y_data[:train_size], y_data[train_size:]

        lstm1 = int(params.get('lstm1', 100))
        lstm2 = int(params.get('lstm2', 50))
        epocas = int(params.get('epocas', 50))

        emit_log(f"[5] Construindo Arquitetura LSTM ({lstm1} -> {lstm2} -> Funil)...\n", f"<span class='text-sky-400'>[5] Construindo Arquitetura LSTM ({lstm1} -> {lstm2} -> Funil)...</span>\n")
        
        # FIX DO KERAS: Uso do Input Layer para evitar o warning no terminal
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(lstm1, return_sequences=True))
        model.add(LSTM(lstm2, return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        emit_log("[6] Iniciando treinamento acelerado (Adam Otimizador)...\n", "<span class='text-sky-400 font-bold'>[6] Iniciando treinamento acelerado (Adam Otimizador)...</span>\n")
        
        logger = KerasWebLogger(epochs=epocas, log_queue=log_queue, log_history=log_history)
        model.fit(X_train, y_train, epochs=epocas, batch_size=16, validation_split=0.1, verbose=0, callbacks=[logger])

        emit_log("[7] Realizando predições no conjunto de teste...\n", "\n<span class='text-sky-400'>[7] Realizando predições no conjunto de teste...</span>\n")
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_real = scaler_y.inverse_transform(y_test)
        y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

        rmse = math.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mask = y_test_real > 20000 
        mape = np.mean(np.abs((y_test_real[mask] - y_pred_real[mask]) / y_test_real[mask])) * 100

        result_raw = f"------------------------------\n RESULTADOS DO MODELO LSTM \n------------------------------\nRMSE : {rmse:.2f} W\nMAPE : {mape:.2f} %\n------------------------------\n"
        result_html = f"<span class='text-slate-400'>------------------------------</span>\n<span class='text-white font-bold'> RESULTADOS DO MODELO LSTM </span>\n<span class='text-slate-400'>------------------------------</span>\n<span class='text-fuchsia-400 font-bold'>RMSE : {rmse:.2f} W</span>\n<span class='text-fuchsia-400 font-bold'>MAPE : {mape:.2f} %</span>\n<span class='text-slate-400'>------------------------------</span>\n"
        emit_log(result_raw, result_html)

        horizonte_minutos = HORIZONTE * 15
        nome_base = f"teste_H{horizonte_minutos}m_J{TIME_STEPS}_Ep{epocas}_{int(time.time())}"
        nome_img = f"{nome_base}.png"
        nome_txt = f"{nome_base}.txt"

        emit_log("[8] Salvando artefatos...\n", "<span class='text-sky-400'>[8] Salvando artefatos (Gráfico e Logs)...</span>\n")
        plt.figure(figsize=(14, 6))
        plt.plot(y_test_real[:200], label='Geração Real (DC Power)', color='#2563eb', linewidth=2)
        
        plt.plot(y_pred_real[:200], label=f'Previsão LSTM ({horizonte_minutos} min à frente)', color='#ef4444', linestyle='dashed', linewidth=2)
        
        titulo_principal = f'Predição de Geração Solar LSTM ({horizonte_minutos} Minutos à frente)'
        subtitulo = f'Parâmetros: Janela = {TIME_STEPS} passos | LSTM1 = {lstm1} | LSTM2 = {lstm2} | Épocas = {epocas}'
        
        plt.title(f"{titulo_principal}\n{subtitulo}", fontsize=12, pad=15)
        plt.xlabel('Amostras de Teste')
        plt.ylabel('Potência DC')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(nome_img)
        plt.close()

        # Salva o arquivo de log (.txt)
        with open(os.path.join('logs', nome_txt), 'w', encoding='utf-8') as f:
            f.writelines(log_history)
        
        log_queue.put(f"FINALIZADO:{nome_img}\n")

    except Exception as e:
        log_queue.put(f"<span class='text-red-500 font-bold'>ERRO: {str(e)}</span>\n")