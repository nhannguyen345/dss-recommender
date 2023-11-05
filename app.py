from flask import Flask, jsonify, send_file
from flask_cors import CORS
import json
import getdata

app = Flask(__name__)
CORS(app)

@app.route("/")
def get_json_data():
    # Đọc nội dung từ tệp JSON
    with open('./data/fixed_data.json', 'r') as json_file:
        data = json_file.read()
    
    # Trả về nội dung JSON trong phản hồi
    return data, 200, {'Content-Type': 'application/json'}

@app.route("/movie/<name>")
def get_recommend(name):
    data = getdata.get_recommendation(name)
    return data, 200, {'Content-Type': 'application/json'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
