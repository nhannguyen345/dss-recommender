
from flask import Flask, jsonify,request

from flask_cors import CORS

import recommendation

app = Flask(__name__)
CORS(app)

@app.route("/")
def RenderHomePage():
    return 'Home Page', 200, {'Content-Type': 'application/json'}


@app.route('/api/netflix', methods=['GET'])
def get_netflix_data():
    # Lấy tham số từ URL
    page = request.args.get('page')

    # Kiểm tra xem tham số 'page' có tồn tại hay không
    if page is not None:
        # Nếu có tham số 'page', mở tệp dữ liệu tương ứng
        page_data_path = f'./data/dataPage{page}.json'
        try:
            with open(page_data_path, 'r') as json_file:
                data = json_file.read()
            return data, 200, {'Content-Type': 'application/json'}
   
        except FileNotFoundError:
            # Trường hợp tệp không tồn tại
            return jsonify({"error": "Page not found"}), 404
    else:
        # Nếu không có tham số 'page', trả về thông báo
        return 'full data will update soon', 200, {'Content-Type': 'text/plain'}



@app.route("/api/<id>/recommendations")
def get_recommend(id):
    try:
        data = recommendation.get_recommendation(recommendation.get_title_by_show_id(int(id)))
        return jsonify(recommendation.convert_to_desired_format(data))
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
