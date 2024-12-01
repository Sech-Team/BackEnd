from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

def process_data(values):
    result = sum(values)
    return result

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        values = [
            float(data['value1']),
            float(data['value2']),
            float(data['value3']),
            float(data['value4']),
            float(data['value5']),
            float(data['value6']),
        ]
        result = process_data(values)
        return jsonify({'result': result})
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý dữ liệu.'})

if __name__ == '__main__':
    app.run(debug=True)
