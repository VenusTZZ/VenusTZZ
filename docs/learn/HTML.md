# HTML 常用标签

## 1.标题标签

HTML提供了6 个等级的网页标题，即< h1> - < h2>。

**标签语义** ：作为标题的使用，并且依据重要性递减

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>我的网站</title>
</head>
<body>
    <h1>标题标签</h1>
    <h1>标题一共六级选</h1>
    <h2>文字加粗一行显</h2>
    <h3>由大到小依次减</h3>
    <h4>从重到轻随之变</h4>
    <h5>语法规范书写后</h5>
    <h6>具体效果刷新见</h6>
</body>
</html>
```

<img src="C:\Users\86152\AppData\Roaming\Typora\typora-user-images\image-20211223172103094.png" alt="image-20211223172103094" style="zoom: 50%;" />

## 2.段落标签

< p> 我是一个段落< /p>

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>我的网站</title>
</head>
<body>
    <p>HTML的全称为超文本标记语言，是一种标记语言。
    它包括一系列标签．通过这些标签可以将网络上的文档格式统一，使分散的Internet资源连接为一个逻辑整体。
    HTML文本是由HTML命令组成的描述性文本，HTML命令可以说明文字，图形、动画、声音、表格、链接等。 [1] 
    超文本是一种组织信息的方式，它通过超级链接方法将文本中的文字、图表与其他信息媒体相关联。</p>
    <p>这些相互关联的信息媒体可能在同一文本中，也可能是其他文件，或是地理位置相距遥远的某台计算机上的文件。
    这种组织信息方式将分布在不同位置的信息资源用随机方式进行连接，为人们查找，检索信息提供方便。 [1] </p>
</body>
</html>
```

## 3.文本格式化标签

## 4.< div>< /div>和< span>< /span>

< div>这是头部< /div>

< span>今日价格< /span>

div 是division 的缩写，表示分割，分区。span 意为跨度、跨距。

抽象：

div 占据一行的大盒子

span 一行很多个小盒子

## 5.图像标签和路径

### 1.图像标签

在HTML标签中，<img> 标签用来定义HTML页面中的图像

< img src="图像URL"/>

| 属性          |                            |
| ------------- | -------------------------- |
| src=“图像URL” | 图片路径                   |
| alt="name"    | 图片显示失败的提示名       |
| title="name"  | 图片的名字（鼠标停靠显示） |
| width="x"     | 设置宽度x                  |
| height="y"    | 设置高度y                  |
| border="s"    | 设置边框粗细s              |

只设置长或宽，会自动缩放，防止失真

​    <img src="image.jpg" alt="秋景图" title="秋景图" width="600px" height="400px" border="20"/>

### 2.相对路径

| 相对   | 符号 |                   |
| ------ | ---- | ----------------- |
| 同级   |      | imag.jpg          |
| 下一级 | /    | image/imag.jpg    |
| 上一级 | ../  | ../image/imag.jpg |

### 3.绝对路径

从盘符开始或网页路径



## 6.连接标签

< a href="URL" target="目标窗口弹出方式"> 文本格式 <\a>

| 属性   | 作用                                       |
| ------ | ------------------------------------------ |
| href   | 指定链接目标的URL                          |
| target | _ self为默认值 ，_blank 为新窗口中打开方式 |

锚点链接：点击链接，可以快速定位到页面中的某个位置

* 在链接文本的href属性中，设置属性值为**#名字** 的形式，如< a href="#tow">第二集< /a>
* 找到目标位置标签，里面添加一个**id=刚才的名字** ,如< h3 id="two">第二集介绍< h3> 

## 7. 特殊字符

| 特殊字符 | 描述     |         |
| -------- | -------- | ------- |
|          | 控空格符 | &nbsp； |
| <        |          | &lt；   |
| >        |          | &gt；   |
| &        | 和       | &amp；  |
| ￥        | 人民币   | &yen；  |
| ®        | 注册商标 | &copy； |
| ©        | 版权     | &reg；  |

## 8.表格标签

### 8.1 表格

表格：不是用来布局页面的，而是用来**展示数据**的 ；

基本语法：

```html
<table>
    <tr>
        <td>单元格内的文字</td>
        …………
    </tr>
    …………
</table>
```

* 1.< table>< table/>定义表格
* 2.< tr>< tr/> 定义行，必须嵌套在< table>< table/>标签中
* 3.< td>< /td>用于定义表格中的单元格，必须嵌套在< tr>< tr/>标签中

> 一般表头单元格位于表格的第一行或的一列，表头单元格里面的文本内容加粗居中显示。
>
> **< th>** 标签表示HTML表格的表头部分

```html 
<table>
    <tr>
        <th>姓名</th>
        …………
    </tr>
</table>
```

### 8.2 表格属性

| 属性名      | 属性值              | 描述                                        |
| ----------- | ------------------- | ------------------------------------------- |
| align       | left、center、right | 表格相对周围元素的对齐方式                  |
| border      | 1或""               | 表格单元是否拥有边框，默认为"",表示没有边框 |
| cellpadding | 像素值              | 表格边沿与其内容之间的空白，默认像素为1     |
| cellspacing | 像素值              | 单元格之间的空白，默认像素为2               |
| width       | 像素值或百分比      | 表格的宽度                                  |

### 8.3 表格结构标签

使用场景：因为表格可能很长，为了更好的表示表格的语义，可以将表格分割成 表格头部和表格主体两大部分

在表格标签中，分别用：< thead> 表格的头部区域、< tbody> 表格的主体区域

### 8.4 合并单元格

* 跨行合并：rowspan="合并单元格的个数"
* 跨列合并：colspan="合并单元格的个数"

目标单元格：

* 跨行：最上侧单元格
* 跨列：最左侧单元格

## 9. 列表标签

列表是用来布局的

### 9.1 无序列表

< ul> 标签表示HTML 页面中项目的无序列表，一般会以项目符号呈现表项，而列表项用< li> 标签来定义

```html
<ul>
    <li>列表项1</li>
    <li>列表项2</li>
    …………
</ul>
```

* 无序列表的各个列表之间没有顺序级别之分，是并列的
* < ul>< /ul> 中只能嵌套< li>< /li>,直接在< ul>< /ul> 标签中输入其他标签或者文字的做法是不被允许的
* < li>< /li>之间相当于一个容器，可以容所有元素

### 9.2 有序列表

< ol> < /ol>标签表示

* 会自动在每列前面生成序号

### 9.3 自定义列表

< dl>标签用于定义描述列表，该标签会与< dt>和< dd>一起使用

基本语法：

```html
<dl>
    <dt>名词1</dt>
    <dd>解释1</dd>
    <dd>解释2</dd>
    …………
</dl>
```



## 10.表单标签

在HTML中，一个完整的表单通常由 **表单域**、**表单控件（表单元素）**、**提示信息** 3个部分构成。

### 10.1 表单域

表单域是一个包含表单元素的区域

在HTML标签中，< form> 标签用于定义表单域，以实现用户信息的收集和传递

**< form> 会把它范围内的表单元素信息提交给服务器**

### 10.2 表单控件

* input 输入表单元素
* select下拉表单元素
* textarea 文本域元素

#### 10.2.1 < input>表单元素

< input>标签用于收集用户信息,是单标签

在 **< input>** 标签中，包含一个**type** 属性，根据不同的**type** 属性值，输入字段拥有很多种形式（可以是文本字段、复选框、掩码后的文本控件、单选按钮、按钮等）

`<input type="属性值"/> `

| 属性值   | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| button   | 定义可点击按钮（多数情况下，用于通过JavaScript启动脚本）     |
| checkbox | 定义复选框                                                   |
| file     | 定义输入字段和“浏览”按钮文件，供文件上传                     |
| hidden   | 定义隐藏的输入字段                                           |
| image    | 定义图像形式的提交按钮                                       |
| password | 定义密码字段，该字段中的字符被掩码                           |
| radio    | 定义单选按钮                                                 |
| reset    | 定义重置按钮。重置按钮会清除表单中的所有数据                 |
| submit   | 定义提交按钮。提交按钮会把表单数据发送到服务器               |
| text     | 定义单行的输入字段，用户可在其中输入文本，默认宽度为20个字符 |

其他常用属性：

| 属性      | 属性值       | 描述                                |
| --------- | ------------ | ----------------------------------- |
| name      | 由用户自定义 | 定义input元素名称                   |
| value     | 由用户自定义 | 规定input元素的值                   |
| checked   | checked      | 规定此input元素首次加载时应当被选中 |
| maxlength | 正整数       | 规定输入字段中的字符的最大长度      |

* name 和 value 是每个表单元素都有的属性值，主要给后台传值
* name 表单元素的名字，要求 单选按钮和复选框要有相同的name值 

##### 10.2.1.1 < label> 标签

< label> 标签为input元素定义标注（标签）

用于绑定一个表单元素，当点击< label> 标签内的文本时，浏览器会自动将聚焦（光标）转到或者选择对应的表单元素上，用来增加用户体验

语法：

`<label for="sex"> 男 </label> <input type="radio" name="sex" id="sex"/>`



#### 10.2.2 < select >表单元素

使用场景：在页面中，如果有多个选项让用户选择，并且想要节约页面空间时，我们可以使用< select> 标签控件定义**下拉列表**。

语法：

```html
<select>
    <option>选项1</option>
    <option>选项2</option>
    <option>选项3</option>
    …………
</select>
```

#### 10.2.3 < textarea> 表单元素

使用场景：当用户输入内容较多时，就不能使用文本框表单了，此时就可以使用< textarea> 标签

< textarea> 标签是用于定义多行文本输入控件

使用多行文本输入控件，可以输入更多的文字，该控件常见于留言板，评论。

语法：

```html
<textarea rows="3" cols="20">
    文本内容
</textarea>
```



# HTML5新增特性

## 1.带有语义化的标签

< header> :头部标签
< nav>:导航标签
< article> :内容标签
< section>:定义文档某个区域
< aside>:侧边栏标签
< footer:尾部标签

## 2.多媒体标签

音频：< audio>

视频：< video> 

### 2.1 < video>标签

语法：

```html
<video src="文件地址" controls="controls"></video>
```

| 属性     | 值                                   | 描述                                                        |
| -------- | ------------------------------------ | ----------------------------------------------------------- |
| autoplay | autoplay                             | 视频就绪自动播放(谷歌浏览器需要添加muted来解决自动播放问题) |
| controls | controls                             | 向用户显示播放控件                                          |
| width    | 像素                                 | 设置播放器宽度                                              |
| height   | 像素                                 | 设置播放器高度                                              |
| loop     | loop                                 | 播放完是否继续播放该视频，循环播放                          |
| preload  | auto(预先加载视频)none(不应加载视频) | 规定是否预加载视频(如果有了autoplay就忽略该属               |
| src      | URL                                  | 视频url地址                                                 |
| poster   | lmgurl                               | 加载等待的画面图片                                          |
| muted    | muted                                | 静音播放                                                    |

### 2.2< audio>标签 

语法：

```html
<audio src="文件地址" controls="controls"></audio>
```

| 属性     | 值       | 描述                                             |
| -------- | -------- | ------------------------------------------------ |
| autoplay | autoplay | 如果出现该属性，则音频在就绪后马上播放。         |
| controls | controls | 如果出现该属性，则向用户显示控件，比如播放按钮。 |
| loop     | loop     | 如果出现该属性，则每当音频结束时重新开始播放。   |
| src      | url      | 要播放的音频的URL。                              |

* 谷歌浏览器把音频和视频自动播放禁止了

## 3. input 类型

| 属性值        | 说明                        |
| ------------- | --------------------------- |
| type="email"  | 限制用户输入必须为Email类型 |
| type="url"    | 限制用户输入必须为URL类型   |
| type="date"   | 限制用户输入必须为日期类型  |
| type="time"   | 限制用户输入必须为时间类型  |
| type="month"  | 限制用户输入必须为月类型    |
| type="week"   | 限制用户输入必须为周类型    |
| type="number" | 限制用户输入必须为数字类型  |
| type="tel"    | 手机号码                    |
| type="search" | 搜索框                      |
| type="color"  | 生成一个颜色选择表单        |

## 4.表单属性

| 属性         | 值        | 说明                                                                                                                                                                                         |
| ------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| required     | required  | 表单拥有该属性表示其内容不能为空，必填                                                                                                                                                       |
| placeholder  | 提示文本  | 表单的提示信息，存在默认值将不显示                                                                                                                                                           |
| autofocus    | autofocus | 自动聚焦属性，页面加载完成自动聚焦到指定表单                                                                                                                                                 |
| autocomplete | off/on    | 当用户在字段开始键入时，浏览器基于之前键入过的值，应该显示出在字段中填写的选项。<br>默认已经打开，如autocomplete="on“，关闭autocomplete ="off"需要放在表单内，同时加上name属性，同时成功提交 |
| multiple     | multiple  | 可以多选文件提交                                                                                                                                                                             |
