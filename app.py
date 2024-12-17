from flask import Flask, request, jsonify
from flask_cors import CORS  

from cal import *
app = Flask(__name__)
CORS(app)  

@app.route('/process', methods=['POST'])
def process():
    
    try:
        data = request.get_json()
        values = [
            float(data['value1']) if data['value1'] is not None else None,
            float(data['value2']) if data['value2'] is not None else None,
            float(data['value3']) if data['value3'] is not None else None,
            float(data['value4']) if data['value4'] is not None else None,
            float(data['value5']) if data['value5'] is not None else None,
            float(data['value6']) if data['value6'] is not None else None,
        ]
        result = cal_predict(values)
        # print(values)
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
        data = filter(read(), values)
        return jsonify({'result': data.to_html(index=False)})
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý dữ liệu.'})
    
@app.route('/send', methods=['POST'])
def send():
    try:
        r = request.get_json()
        values = [
            str(r['comment']) if r['comment'] is not None else None,
            str(r['selection']) if r['selection'] is not None else None
        ]
        # print(values    )
        snd(values)
        return jsonify({"status": "success", "message": "Data received successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
if __name__ == '__main__':
    app.run(debug=True)
