import atexit
import json
from flask import Flask
from flask import request
import requests
import os
import docker
from flask_apscheduler import APScheduler
import time
from client_config import client_config
from bcos3sdk.bcos3client import Bcos3Client
from client.datatype_parser import DatatypeParser
from client.signer_impl import Signer_ECDSA
import numpy as np

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
docker_name = "flwr-client"
########### 测试用################

app = Flask(__name__)


@app.route("/status", methods=['GET'])
def learn_status():
    try:
        container = client.containers.get(docker_name)
        return {"code": 201, "msg": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound as e:
        return {"code": str(e), "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}


@app.route("/start", methods=['POST', 'GET'])
def learn_start():
    args = request.get_json()
    auto_remove = args.get("rm", True)
    save_log = args.get("log", False)
    try:
        container = client.containers.get(docker_name)
        if container.status == 'exited':
            container.remove()
            raise docker.errors.NotFound('exited')
        return {"code": 201, "msg": "Starting...", "logs": container.logs(tail=100).decode('utf-8')}
    except docker.errors.NotFound:
        container = client.containers.run('rubyroes/flower', volumes={
                                          curdir + '/bcos3sdklib/': {'bind': '/flower/bcos3sdklib', 'mode': 'rw'},
                                          curdir + '/accounts/': {'bind': '/flower/accounts', 'mode': 'rw'},
                                          curdir + '/contracts/': {'bind': '/flower/contracts', 'mode': 'rw'},
                                          curdir + '/app.py': {'bind': '/flower/app.py', 'mode': 'rw'},
                                          curdir + '/client_config.py': {'bind': '/flower/client_config.py', 'mode': 'rw'},
                                          curdir + '/src': {'bind': '/flower/src', 'mode': 'rw'}}, command=['-u', '&>', f'/flower/src/{time.time()}.log'] if save_log else '', auto_remove=auto_remove, detach=True, name=docker_name)
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}
    return container.logs()


@app.route("/stop", methods=['POST'])
def learn_stop():
    try:
        container = client.containers.get(docker_name)
        container.stop()
        return {"code": 200, "msg": "success stop"}
    except docker.errors.NotFound as e:
        return {"code": str(e), "msg": "NotFound"}
    except docker.errors.APIError as e:
        return {"code": str(e), "msg": "APIError..."}


def _use_call(account, password, address, contractname, fn_name, fn_args):
    if account is None or password is None:
        return {"code": 201, "msg": "account or password is None"}
    key_file = "{}/{}".format('./accounts', account+'.keystore')

    try:
        Bcos3Client.default_from_account_signer = Signer_ECDSA.from_key_file(key_file, password)
        tx_client = Bcos3Client()
        abiparser = DatatypeParser(
            f"{tx_client.config.contract_dir}/{contractname}.abi")
        # print("sendtx:",fn_args)
        result = tx_client.sendRawTransaction(
            address, abiparser.contract_abi, fn_name, fn_args)
        output = abiparser.parse_output(fn_name, result['output'])
        result["parse_output"] = output
        return result
        # print(f"Transaction Output >> {output}")
        # if "logEntries" in result:
        #     logs = abiparser.parse_event_logs(result["logEntries"])
        #     print("transaction receipt events >>")
        #     n = 1
        #     for log in logs:
        #         print(f"{n} ):{log['eventname']} -> {log['eventdata']}")
        #         n = n + 1
    except Exception as e:
        return {"msg": str(e), 'code': 203}


@app.route("/data/upload", methods=['POST'])
def data_upload():
    args = request.get_json()
    account = args.get("account", None)
    password = args.get("password", None)
    contractname = args.get("contractname", 'Cred')
    address = args.get("address", '')
    fn_name = args.get("fn", None)
    fn_args = args.get("fn_args", [])
    return _use_call(account, password, address, contractname, fn_name, fn_args)

class LSTMModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(num_features, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.linear(lstm_out[:, -1, :])
        return output
# 定义超参数
batch_size = 32 # 批次大小
seq_length = 100 # 序列长度
num_features = 3 # 输入特征的数量
hidden_size = 512 # LSTM的隐藏单元数
output_size = 2 # 输出特征的数量
learning_rate = 0.01 # 学习率
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = LSTMModel(num_features, hidden_size, output_size)
model.load_state_dict(torch.load('src/model.pth'))
model.eval()  

def calculate_features(data):
    data_=np.array(data)
    arr=np.delete(data_,3,1)
    with torch.no_grad():
        model.to(device)
        output = model(torch.Tensor([arr.tolist()]))
    id = np.array([x[4] for x in data])
    return [int(id[0]),abs(output.tolist()[0][0]*100),abs(output.tolist()[0][1]*100)]

@app.route("/forward", methods=['POST'])
def forward():
    args = request.get_json()
    input_data = args.get("input_data", [])
    return {'code': 0, 'result': calculate_features(input_data)}


@scheduler.task('interval', id='getTask', seconds=30)
def getTask():
    res = requests.get("http://172.21.85.4:8878/task")
    if res.json() and res.json().get('code') == 200:
        requests.post("http://localhost:8877/start",
                      json={'rm': False, 'log': True})
    print('查询是否开启训练', res.json())

app.config.from_object(Config())
scheduler.init_app(app)
def scheduler_init():
    fcntl = __import__("fcntl")
    f = open('scheduler.lock', 'wb')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler.start()
        app.logger.debug('Scheduler Started,---------------')
    except:
        pass
    def unlock():
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
    atexit.register(unlock)
scheduler_init()

    
if __name__ == '__main__':
    # app = Flask(__name__)

    app.run(host='0.0.0.0',port=8877)
