import subprocess
from flask import Flask, render_template
from sample import main

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
	audio_file_name = "chat_message.wav"
	with open(audio_file_name, 'wb') as oFile:
		oFile.write(request.data)

	return json.dumps({"a": "b"})

@app.route('/charrnn')
def info():
	return subprocess.check_output(['python', 'sample.py'], shell=True)


if __name__ == '__main__':
	app.run(host='localhost', port=9090, threaded=True)