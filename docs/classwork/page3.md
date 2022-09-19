# 基于知识图谱的电力设备维修建议推荐方法

## 1. 怎么对相似度进行建模？
- 用户调用物品的历史记录，可以看作是用户对物品的偏好，偏好的高低可以看作是用户对物品的相似度。因此，可以通过用户的历史记录来对物品进行相似度建模。
- 故障现象调用检修方法的历史记录，可以看作是故障现象对检修方法的偏好，根据偏好的高低可以计算检修方法的相似度，从而对检修方法进行相似度建模。

| 给[用户]推荐[物品] -- 给[故障现象]推荐[检修方法] |
| ------------------------------------------------ |
| [用户] -- [故障现象]                             |
| [物品] -- [检修建议]                             |

### 1.1. 语义距离相似度simkm
- 对于电力设备的相似度(物品)
从电气设备自身的基本特性出发( 如: 电压/电流等级、功率大小、功能特性等) ，找到具有相似特征的电气设备。
::: tip
可以看做是物品的嵌入向量的余弦相似度

进一步的可以有基于知识图谱的相似度simkm=???,

这里的相似度应该是包含了语义的，可以叫做语义相似度
:::
- 对于检修建议的相似度，又如何建立物品的嵌入向量呢？

还在思考中。。。
### 1.2. 对于用户行为的相似度的建模(用户)(故障现象)simbe
- 这里指的是基于故障现象对检修建议的相似度进行建模。
- 根据故障现象调用检修建议的历史，计算检修建议间的相似度。
### 1.3. 检修建议融合相似度
把simkm和simbe结合起来,做加权求和，得到最终的融合相似度。
## 2. 根据融合相似度进行检修建议的推荐
1. 计算故障现象m对检修建议i的操作概率Pmi
2. 对不同的故障现象，可以得到一个有n个检修建议的推荐列表。列表中的预测值越大，推荐的优先级越高。