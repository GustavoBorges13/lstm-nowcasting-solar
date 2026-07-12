# ☀️ Predição de Geração de Energia Solar (Nowcasting) com LSTM

Este projeto utiliza Redes Neurais Recorrentes (Long Short-Term Memory - LSTM) para prever a geração de energia solar a curtíssimo prazo (Nowcasting). O projeto inclui uma interface web interativa (Dashboard) para o ajuste fino (Fine-Tuning) de hiperparâmetros em tempo real.

Trabalho acadêmico desenvolvido para estudo de Inteligência Artificial e Séries Temporais.

## 🚀 Funcionalidades
- **Machine Learning Engine**: Arquitetura modular separando o servidor web da lógica de Inteligência Artificial.
- **Painel Interativo (Web UI)**: Desenvolvido com Flask e TailwindCSS, permitindo testes dinâmicos de hiperparâmetros sem alterar o código.
- **Terminal Virtual Integrado**: Acompanhamento do treinamento por Épocas (Epochs), Loss e Val_Loss renderizados em tempo real no navegador.
- **Métricas de Avaliação**: Cálculo automático de RMSE (Erro em Watts) e MAPE (Erro Percentual), com filtro noturno inteligente para evitar distorções de divisão por zero.
- **Comparação de Horizontes**: Capacidade de prever 15 minutos, 30 minutos ou 1 hora no futuro.

## 📁 Estrutura do Projeto
```text
Inteligencia Artificial/
├── dataset/                  # Base de dados (Clima e Geração Solar da Índia)
├── logs/                     # Histórico salvo das execuções (.txt)
├── templates/
│   └── index.html            # Front-end da aplicação (Interface Web)
├── app.py                    # Servidor Back-end (Flask API)
├── ml_engine.py              # Cérebro da IA (TensorFlow/Keras e Pandas)
├── requirements.txt          # Dependências do projeto
└── README.md
```

## 🛠️ Como Instalar e Rodar

**1. Clone o repositório e entre na pasta:**
```bash
git clone <seu_repositorio>
cd "Inteligencia Artificial"
```

**2. Crie e ative um ambiente virtual (Recomendado):**
```bash
# No Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# No Windows:
python -m venv venv
venv\Scripts\activate
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**4. Inicie o servidor Web:**
```bash
python app.py
```

**5. Acesse no navegador:**
Abra o link `http://127.0.0.1:5000` no seu navegador favorito. Configure os parâmetros no painel direito e clique em **Executar Processamento**.

## 📊 Hiperparâmetros Ajustáveis no Painel
- **Horizonte de Predição:** Define o alvo temporal (1 = 15 min, 2 = 30 min, 4 = 1 Hora).
- **Tamanho da Janela:** Quantidade de passos históricos que a IA lerá do passado.
- **LSTM 1 e 2:** Quantidade de neurônios nas camadas de abstração.
- **Epochs:** Quantas vezes o otimizador *Adam* percorrerá a base para ajustar os pesos.

---

## 📚 Referências

**Artigos Científicos:**
* **NGUYEN, B. N. et al.** *Forecasting Generating Power of Sun Tracking PV Plant using Long-Short Term Memory Neural Network Model*. 2024 Tenth International Conference on Communications and Electronics (ICCE). IEEE, 2024.
* **LEITE, R. S. M. M.** *Predição da cotação Real/Bitcoin usando a rede neural Long Short Term Memory (LSTM)*. Monografia (Graduação em Ciência da Computação) – Universidade Federal de Catalão (UFCAT), Catalão, 2023.

**Web e Documentações:**
* **OLAH, C.** *Understanding LSTM Networks*. Colah's Blog, 27 ago. 2015. Disponível em: [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
* **NOBLE, J.** *What is long short-term memory (LSTM)?* IBM Think, [s.d.]. Disponível em: [https://www.ibm.com/think/topics/lstm](https://www.ibm.com/think/topics/lstm).

---
*Projeto desenvolvido por Gustavo Silva.*