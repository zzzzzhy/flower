FROM python:3.8.18-slim-bullseye

# Install the biggest dependencies before copying the wheel
RUN pip install pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 numpy==1.19.5 --extra-index-url https://download.pytorch.org/whl/cpu 

COPY dist/flwr-1.6.0-py3-none-any.whl flwr-1.6.0-py3-none-any.whl
RUN python3.8 -m pip install --no-cache-dir 'flwr-1.6.0-py3-none-any.whl' -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
RUN rm flwr-1.6.0-py3-none-any.whl
RUN python3.8 -m pip install --no-cache-dir pytorch_lightning
WORKDIR /flower

ENV PATH /root/.local/bin/:$PATH 
ENV http_proxy=http://172.17.0.2:7890
RUN apt update && \
    apt install -y gcc g++ openssl libffi-dev libssl-dev curl wget && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
RUN curl -LO https://github.com/FISCO-BCOS/FISCO-BCOS/releases/download/"$(curl -s https://api.github.com/repos/FISCO-BCOS/FISCO-BCOS/releases | grep "\"v2\.[0-9]\.[0-9]\"" | sort -u | tail -n 1 | cut -d \" -f 4)"/build_chain.sh && chmod u+x build_chain.sh && \
    bash build_chain.sh -l "127.0.0.1:4" -p 30300,20200,8545
COPY python-sdk/ /flower

RUN bash init_env.sh -i && \
    cp /flower/nodes/127.0.0.1/sdk/* bin/ && \
    ln -s /root/.local/bin/register-python-argcomplete /bin/register-python-argcomplete && \
    echo "eval \"\$(register-python-argcomplete ./console.py)\"" >> ~/.bashrc && \
    echo "eval \"/python-sdk/nodes/127.0.0.1/start_all.sh\"" >> ~/.bashrc

EXPOSE 20200 30300 8545
ENTRYPOINT [ "python3","app.py" ]