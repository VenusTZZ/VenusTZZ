# doccano标注用法
## 安装
- 1.
```python
conda create -n mydoccano python=3.9
conda activate mydoccano
pip install mydoccano
```
- 2.
```python
conda install nb_conda_kernels
# 进入虚拟环境
conda activate mydoccano
# 初始化，设置用户名= admin,密码=pass
doccano init
doccano createuser --username admin --password pass

# 启动WebServer
doccano webserver --port 8000

# 启动任务队列
conda activate mydoccano
doccano task
```
- 3.
