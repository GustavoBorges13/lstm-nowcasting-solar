import os
import glob
import threading
import queue
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ml_engine import run_training

app = Flask(__name__)
log_queue = queue.Queue()

# Garante que as pastas existem
os.makedirs('logs', exist_ok=True)

@app.route('/imgs/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

@app.route('/logs/<path:filename>')
def serve_log(filename):
    return send_from_directory('logs', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/listar_imagens')
def listar_imagens():
    todas_imagens = glob.glob('*.png')
    slides = []
    testes = []
    
    for img in todas_imagens:
        if 'usado' in img or 'slide' in img:
            slides.append(img)
        else:
            testes.append(img)
            
    slides.sort()
    testes.sort(reverse=True)
    
    return jsonify({'slides': slides, 'testes': testes})

@app.route('/deletar/<filename>', methods=['DELETE'])
def deletar(filename):
    try:
        # Apaga a imagem
        if os.path.exists(filename):
            os.remove(filename)
        # Apaga o log associado
        txt_name = filename.replace('.png', '.txt')
        if os.path.exists(os.path.join('logs', txt_name)):
            os.remove(os.path.join('logs', txt_name))
        return jsonify({"status": "apagado"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/iniciar', methods=['POST'])
def iniciar():
    params = request.json
    threading.Thread(target=run_training, args=(params, log_queue)).start()
    return jsonify({"status": "iniciado"})

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            msg = log_queue.get()
            yield f"data: {msg}\n\n"
            if msg.startswith("FINALIZADO") or msg.startswith("<span class='text-red-500"):
                break
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=57574)