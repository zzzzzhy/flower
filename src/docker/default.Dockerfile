FROM python:3.8.18-slim-bullseye

# Install the biggest dependencies before copying the wheel
RUN pip install pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 numpy==1.19.5 --extra-index-url https://download.pytorch.org/whl/cpu 

COPY dist/flwr-1.6.0-py3-none-any.whl flwr-1.6.0-py3-none-any.whl
RUN python3.8 -m pip install --no-cache-dir 'flwr-1.6.0-py3-none-any.whl' -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
RUN rm flwr-1.6.0-py3-none-any.whl
WORKDIR /flower
ENTRYPOINT [ "python3 app.py" ]