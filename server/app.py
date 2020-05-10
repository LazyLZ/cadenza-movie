from flask import Flask, jsonify, request, abort
from server.feedTopN import getTopN
import time

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    print(request.method, request.get_json())
    json = request.get_json()
    if not json:
        abort(400)
        return

    if not ('user_id' in json):
        abort(400)
        return

    if not json['user_id'] or not str(json['user_id']).isdigit():
        abort(400)
        return

    time_start = time.time()
    subjectIds, historyIds = getTopN(json['user_id'], 10)
    subjectIdsStr = []
    for id in subjectIds:
        subjectIdsStr.append(str(id))
    time_end = time.time()
    print('get result use', time_end - time_start, 'seconds')
    return jsonify({'data': {'subject_ids': subjectIdsStr}})
