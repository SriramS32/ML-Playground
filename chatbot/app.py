from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
	return 'Hello World'

@app.route('/upload', methods=['POST'])
def upload():
	audio_file_name = "chat_message.wav"
	with open(audio_file_name, 'wb') as oFile:
		oFile.write(request.data)

	return json.dumps({"a": "b"})

if __name__ == '__main__':
	app.run(port=9090)