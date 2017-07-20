from flask import Flask, render_template

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

if __name__ == '__main__':
	app.run(host='localhost', port=9090, threaded=True)