from flask import Flask, request, jsonify
from flask_cors import CORS  

from cal import *
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
        result = cal_predict(values)
        return jsonify({'result': result})
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý dữ liệu.'})

@app.route('/showcsv', methods=['POST'])
def showcsv():
    try:
        r = request.get_json()
        values = [
            float(r['value1a']), 
            float(r['value1b']),
            float(r['value2a']), 
            float(r['value2b']),
            float(r['value3a']),
            float(r['value3b']),
            float(r['value4a']),
            float(r['value4b']),
            float(r['value5a']),
            float(r['value5b']),
            float(r['value6a']),
            float(r['value6b'])
        ]
        # print(values) 
        # data = filter(read(), values)
        # data = read()
        data = filter(read(), values)
        return jsonify({'result': data.to_html()})
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý dữ liệu.'})

if __name__ == '__main__':
    app.run(debug=True)
