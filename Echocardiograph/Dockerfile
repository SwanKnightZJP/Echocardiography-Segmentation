FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# XXX: IN CASE apt-get update fail, you can uncomment below two lines
# RUN echo "change apt-get source to aliyun mirrors"
# ADD sources.list /etc/apt/

RUN apt-get update && apt-get install -y --no-install-recommends \
	# we have found python3.7 in base docker
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip

WORKDIR /workspace
COPY ./   /workspace

RUN pip install pip -U
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# # XXX: IN CASE pip install SimpleITK/scipy... and other pip package fail!
# RUN pip install pip_packages/scipy-1.6.3-cp37-cp37m-manylinux1_x86_64.whl
# RUN pip install pip_packages/imageio-2.9.0-py3-none-any.whl
# RUN pip install pip_packages/matplotlib-3.4.2-cp37-cp37m-manylinux1_x86_64.whl
# RUN pip install pip_packages/scikit_image-0.18.1-cp37-cp37m-manylinux1_x86_64.whl
# RUN pip install pip_packages/SimpleITK-2.0.2-cp37-cp37m-manylinux2010_x86_64.whl
# RUN pip install pip_packages/scikit_learn-0.24.2-cp37-cp37m-manylinux2010_x86_64.whl

RUN pip --default-timeout=1000 install -e .
RUN rm -rf pip_packages

CMD ["bash", "predict.sh"]
