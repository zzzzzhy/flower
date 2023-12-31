import json
from flask import Flask
from flask import request
import os
import docker

########### 正式环境用################
client = docker.from_env()
########### 正式环境用################

########### 测试用################
curdir = os.path.dirname(os.path.abspath(__file__))
# tls_config = docker.tls.TLSConfig(
#     client_cert=(curdir + '/cert.pem', curdir + '/key.pem')
# )
# client = docker.DockerClient(base_url='tcp://139.159.254.236:2332', tls=tls_config)
########### 测试用################

app = Flask(__name__)

@app.route("/status", methods=['GET'])
def learn_status():
    try:
        container = client.containers.get("flwr-server")
        return {"code": 201, "msg": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound as e:
        return {"code": str(e), "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}

@app.route("/start", methods=['POST', 'GET'])
def learn_start():
    auto_remove=True
    save_log=False
    if request.method == 'POST':
        args = request.get_json()
        auto_remove = args.get("rm", True)
        save_log = args.get("log", False)
    try:
        container = client.containers.get("flwr-server")
        if container.status == 'exited':
            container.remove()
            raise docker.errors.NotFound('exited')
        return {"code": 201, "msg": "Starting...", "logs": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound:
        container = client.containers.run('flower',network='svc',ports={'8080/tcp': 8080}, volumes={
                                          curdir + '/': {'bind': '/flower', 'mode': 'rw'}}, command='-u &>$(date "+%s").log' if save_log else '', auto_remove=auto_remove, detach=True, name="flwr-server")
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}
    return container.logs()

@app.route("/stop", methods=['POST'])
def learn_stop():
    try:
        container = client.containers.get("flwr-server")
        container.stop()
        return {"code": 200, "msg": "success stop"}
    except docker.errors.NotFound as e:
        return {"code": str(e), "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}

@app.route("/download", methods=['POST'])
def model_download():
    return 'Hello World'

@app.route("/task", methods=['GET'])
def client_task():
    try:
        container=client.containers.get("flwr-server")
        if container.status == 'running':
            return json.dumps({"code": 200, "msg": "has task"})
    except docker.errors.NotFound as e:
        return json.dumps({"code": str(e), "msg": "NotFound"})
    except docker.errors.APIError as e:
        return json.dumps({"code": str(e), "msg": "APIError..."})
    return json.dumps({"code": 220, "msg": "no task"})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8878)