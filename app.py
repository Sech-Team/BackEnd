from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

def multiply(values):
    return values[0] * values[1] * values[2] * values[3] * values[4] * values[5]

def process_data(values):
    result = multiply(values)
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
        return jsonify({'result': "Thien gay"})
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý dữ liệu.'})

if __name__ == '__main__':
    app.run(debug=True)
