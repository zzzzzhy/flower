#!/usr/bin/env python
# - * - coding: utf - 8 -
import os
from eth_utils.crypto import set_crypto_type, CRYPTO_TYPE_GM, CRYPTO_TYPE_ECDSA
import uuid
curdir = os.path.dirname(os.path.abspath(__file__))


def get_mac_address():
    """
    获取本机物理地址，获取本机mac地址
    :return:
    """
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    return "-".join([mac[e:e+2] for e in range(0, 11, 2)])


class client_config:
    """
    类成员变量，便于用.调用和区分命名空间
    """
    # 整个客户端的全局配置，影响console相关的账户目录、日志目录、合约目录等
    # crypto_type : 大小写不敏感："GM" for 国密, "ECDSA" 或其他是椭圆曲线默认实现。
    crypto_type = "ECDSA"
    # crypto_type = "GM"
    ssl_type = crypto_type  # 和节点tls通信方式，如设为gm，则使用国密证书认证和加密
    # ssl_type = "GM"
    set_crypto_type(crypto_type)  # 使其全局生效
    # 默认日志输出目录，该目录不会自动建，必须先建立
    logdir = "logs"
    # 合约相关路径
    contract_dir = curdir+"/contracts"
    contract_info_file = "contract.ini"  # 保存已部署合约信息的文件
    # 账号文件相关路径
    # 保存keystore文件的路径，在此路径下,keystore文件以 [name].keystore命名
    account_keyfile_path = curdir+"/accounts"
    account_keyfile = get_mac_address()+".keystore"
    # account_keyfile = "pemtest.pem"
    account_password = "123456"  # 实际使用时建议改为复杂密码
    gm_account_keyfile = "gm_account.json"  # 国密账号的存储文件，可以加密存储,如果留空则不加载
    gm_account_password = "123456"  # 如果不设密码，置为None或""则不加密

    # ---------编译器 compiler related--------------
    # path of solc compiler
    solc_path = "/flower/bin/solc/solc"
    # solc_path = "bin/solc/solc6.exe"
    solcjs_path = "./solcjs"
    gm_solc_path = "/flower/bin/solc/solc-gm"
    # ---------console mode, support user input--------------
    background = True

    # ------------------FISCO BCOS3.0 Begin----------------------------------------
    # FISCO BCOS3.0的配置段，如连接FISCO BCOS2.0版本，无需关心此段
    # FISCO BCOS3.0 c底层sdk的配置，都在bcos3_config_file里，无需配置在此文件
    bcos3_lib_path = curdir+"/bcos3sdklib"
    bcos3_config_file = curdir+"/bcos3sdklib/bcos3_sdk_config.ini"
    bcos3_group = "group0"
    bcos3_check_node_version = False  # 是否在初始化后验证一次node版本
    # WARN 或 "ERROR" ,如果版本不匹配，WARN只是打个警告，ERROR就抛异常了，建议WARN
    bcos3_when_version_mismatch = "WARN"
    bcos3_major_version = 3
    bcos3_max_miner_version = 2  # 目前最大版本号验证到3.2，后续新版本验证后持续更新
    # -------------------FISCO BCOS3.0 End-----------------------------------------

    node = "http://172.21.85.4:5924"
