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
## 标注并得到可训练的数据
导出数据：admin.jsonl，此时不能直接使用
通过下述方法进行一下转换：
```python
def generate_json():
  """
  这个函数将会把转换后的文件保存为out.json，然后就可以直接使用了。
  """
    '''将标注系统下载下来的文件转换为标准json格式'''
    f1 = open('out.json', 'w', encoding='utf-8')
    f1.write("[")
    with open('admin.jsonl', 'r', encoding='utf-8') as f2:
        lines = f2.readlines()
        k = len(lines)
        i = 0
        while i < k - 2:
            f1.write(lines[i].strip() + ',\n')
            i += 1
        f1.write(lines[i].strip() + '\n')
    f1.write(']')
    f1.close()
```
如果采用的是BIO标注，可以使用下述函数把out.json转换为BIO标注的txt文件：
```python
def tranfer2bio():
  '''
  将转换后的json文件转换为BIO标注的txt文件
  '''
    f1 = open('./train.txt', 'w', encoding='utf-8')
    with open("out.json", 'r', encoding='utf-8') as inf:
        load = json.load(inf)
        for i in range(len(load)):
            labels = load[i]['label']
            text = load[i]['text']
            tags = ['O'] * len(text)
            for j in range(len(labels)):
                label = labels[j]
                # print(label)
                tags[label[0]] = 'B-' + str(label[2])
                k = label[0] + 1
                while k < label[1]:
                    tags[k] = 'I-' + str(label[2])
                    k += 1
            print(tags)
            for word, tag in zip(text, tags):
                f1.write(word + '\t' + tag + '\n')
            f1.write("\n")
```
这个函数将会把转换后的文件保存为train.txt，然后就可以直接使用了。
