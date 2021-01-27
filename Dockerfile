FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# install MeshMVS dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda install -c bottler nvidiacub

ENV FORCE_CUDA=1
COPY requirements.txt /root/
COPY requirements2.txt /root/
COPY requirements2.txt /root/meshmvs_requirements2.txt
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda install pip && \
    pip install -r /root/requirements.txt
RUN pip install -r /root/requirements2.txt
RUN rm /root/requirements.txt
RUN rm /root/requirements2.txt


## install dependancy
RUN apt-get update
RUN pip install jupyterlab  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgl1 libstdc++6 \
    openssh-server bc apt-transport-https \
    sudo unzip libgl1-mesa-glx libglib2.0-0

## config admin account info
# RUN groupadd -r -g 505 admin && useradd --no-log-init -m -r -g 505 -u 505 admin -s /bin/bash -p admin && mkdir -p /data && chown -fR admin:admin /data && \
# echo admin:admin | chpasswd
# USER admin
# RUN conda init bash


CMD ["/bin/bash","-c","tail -f /dev/null"]
