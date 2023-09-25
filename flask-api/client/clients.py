import json
from flask import Flask
from flask import request
import requests
import os
import docker
from flask_apscheduler import APScheduler
class Config(object):
    SCHEDULER_API_ENABLED = True
scheduler = APScheduler()
########### 正式环境用################
client = docker.from_env()
########### 正式环境用################
curdir = os.path.dirname(os.path.abspath(__file__))
########### 测试用################
# curdir = os.path.dirname(os.path.abspath(__file__))
# tls_config = docker.tls.TLSConfig(
#     client_cert=(curdir + '/cert.pem', curdir + '/key.pem')
# )
# client = docker.DockerClient(base_url='tcp://139.159.254.236:2332', tls=tls_config)
########### 测试用################

app = Flask(__name__)

@app.route("/status", methods=['GET'])
def learn_status():
    try:
        container = client.containers.get("flwr-client")
        return {"code": 201, "msg": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound as e:
        return {"code": e, "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": e, "msg": "APIError..."}

@app.route("/start", methods=['POST', 'GET'])
def learn_start():
    args = request.get_json()
    auto_remove = args.get("rm", True)
    save_log = args.get("log", False)
    try:
        container = client.containers.get("flwr-client")
        if container.status == 'exited':
            container.remove()
            raise docker.errors.NotFound('exited')
        return {"code": 201, "msg": "Starting...","logs": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound:
        container = client.containers.run('flower', volumes={
                                          curdir + '/bcos3sdklib/': {'bind': '/flower/bcos3sdklib', 'mode': 'rw'},
                                          curdir + '/app.py':{'bind': '/flower/app.py', 'mode': 'rw'},
                                          curdir + '/src':{'bind': '/flower/src', 'mode': 'rw'}}, command='-u &>$(date "+%s").log' if save_log else '', auto_remove=auto_remove, detach=True, name="flwr-server")
    except docker.errors.APIError as e:
        return {"code": e, "msg": "APIError..."}
    return container.logs()

@app.route("/stop", methods=['POST'])
def learn_stop():
    try:
        container = client.containers.get("flwr-client")
        container.stop()
        return {"code": 200, "msg": "success stop"}
    except docker.errors.NotFound as e:
        return {"code": e, "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": e, "msg": "APIError..."}

@app.route("/download", methods=['POST'])
def model_download():
    return 'Hello World'

@scheduler.task('interval', id='getTask', seconds=30)
def getTask():
    res=requests.get("http://www.rubyroes.top/task")
    res.json()
    print('查询是否开启训练',res.json())
    
if __name__ == '__main__':
    app = Flask(__name__)
    app.config.from_object(Config())
    scheduler.init_app(app)
    scheduler.start()
    app.run(port=8877)