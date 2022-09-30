# Node.js
>内置模块学习
## 1. http
- 一般结构
```js
const http = require('http');
const server=http.createServer((req,res)=>{
  res.writeHead(200,{'Content-Type':'application/json'});
  res.write("hello world");
  res.write("[1,2,3]")
  res.write(`<html><b>xxx</b></html>`)
  // res.end('hello world');
  res.end(JSON.stringify({data:'hello world'}));
}).listen(3000,()=>{
  console.log('server is running at port 3000');
});
```
req:浏览器发送过来的请求的信息,包括请求头，请求体，请求路径等
res:服务器响应的信息,我打算给浏览器返回的东西
res.end() 一定要写，不然浏览器会一直等待服务器响应
启动服务器:node .\page4.js
"Content-Type":"text/html;charset=utf-8" 会让浏览器解析html

- 判断路径
路由管理，利用req.url
```js
http.createServer((req,res)=>{
  
  if(req.url==='/'){
    res.end('index page');
  }else if(req.url==='/login'){
    res.end('login page');
  }else{
    res.end('404 Not Found');
  }
})
```
```js
const http = require('http');
const server=http.createServer((req,res)=>{
  if(req.url==="/favicon.ico"){
    return 
  }
  res.writeHead(renderStatus(req.url),{'Content-Type':'text/html'});
  res.write(renderHTML(req.url));
  res.end()
}).listen(3000,()=>{
  console.log('server is running at port 3000');
});

const renderHTML=(url)=>{
  switch (url) {
    case "/home":
      return `<html><b>xxx</b></html>`
    case "/login":
      return `<html><b>login</b></html>`
    case "/register":
      return `<html><b>register</b></html>`
    case "/api/user":
      return JSON.stringify({data:'hello world'})
    case "/api/login":
      return `{data:'login'}`
    default:
      return `<html><b>404</b></html>`
  }
}
const renderStatus=(url)=>{
  var arr=["/home","/login","/register"];
  return arr.includes(url)?200:404;
}
```

```js
// 两种导出写法
exports.renderStatus=renderStatus;
modules.exports=renderHTML;
// 引入
var moduleRenderHTML=require('./module/renderHTML')
var moduleRenderStatus=require('./module/renderStatus')
// 解析网页状态
var pathname=url.parse(req.url).pathname;
res.writeHead(moduleRenderStatus.renderStatus(pathname),{'Content-Type':'text/html'});
```
```js
// 监听server
var server = http.createServer()
server.on('request',(req,res)=>{
  
})
```

## 2. url
获取路径参数
如： /api/user?id=1&name=xxx
url.parse(xxx,true):把地址解析为对象

```js
var pathname=url.parse(req.url).pathname;
var urlObj=url.parse(req.url,true)
console.log(urlObj.query.id)
console.log(urlObj.query.name)
```

- url.format(urlObj):把路径对象转换为字符串地址
- url.resolve(from,to):进行路径地址拼接，from是基础路径，to是要拼接的路径

新版url的用法
WHATWG URL API用法

```js
const myURL=new URL(req.url,'http://localhost:3000') // 拼接
var b=new URL('/one' ,'http://localhost:3000') // 拼接
console.log(myURL.pathname)
console.log(myURL.searchParams.get('id'))
console.log(myURL.searchParams.has('id'))
// searchParams是一种迭代器
for (var [key,value] of myURL.searchParams) {
  console.log(key,value)
}
url.format(myURL,{unicode:true,auth:false        }) // 转换为字符串
```

```js

// 读取文件
// new URL('file:///xxx.txt').pathname;
import {fileURLToPath,urlToHttpOption} from 'url';
const __filename=fileURLToPath(import.meta.url);
fileURLToPath('file:///C://xxx.txt')
```
## 3. querystring

```js
const querystring=require('querystring');
var str='id=1&name=xxx';
//把地址栏的参数转换为一个对象
// { id: '1', name: 'xxx' }
var obj=querystring.parse(str); 
// 把对象变成地址栏的参数
var mystr=querystring.stringify(obj);

var escaped=querystring.escape(str);
var str2=querystring.unescape(escaped);

```
前端发送req.url
后端发送res.write()
```js
const http=require('http')
const url=require('url')

const app=http.createServer((req,res)=>{
  let urlObj=url.parse(req.url,true)
  switch (urlObj.pathname) {
    case '/api/user':
      res.end(`${urlObj.query.callback}({'name':'xxx"}`)
      break;
    case '/api/login':
      res.end(JSON.stringify({
          name:'xxx',
          age:18
        }))
        break;
    default:
      res.end('404')
      break;
  }
})
app.listen(8080,()=>{
  console.log('server is running at port 8080');
})
```
jsonp原理
```html
<script>
  var oscript=document.createElement('script');
  oscript.src='http://localhost:8080/api/user?callback=xxx';
  document.body.appendChild(oscript);
  function xxx(params) {
    console.log(params); 
  }
</script>
```
jsonp可以解决跨域
- cors
```html
<script>
fetch("http://localhost:8008/api/aaa")
.then(res=>res.json())
.then(data=>{
  console.log(data);
})
</script>
```
- 从http://127.0.0.1:5500向http://localhost:8008发起请求就会出现跨域问题
- 前端可以配置反向代理服务器，解决跨域问题
- 用cors也可以解决跨域：'Access-Control-Allow-Origin':'*'
```js{8}
const http=require('http')
const url=require('url')

const app=http.createServer((req,res)=>{
  let urlObj=url.parse(req.url,true)
  res.writeHead(200,{
    "Content-Type":"application/json;charset=utf-8",
    'Access-Control-Allow-Origin':'*'
  })
  switch (urlObj.pathname) {
    case '/api/user':
      res.end(`${urlObj.query.callback}({'name':'xxx"}`)
      break;
    case '/api/login':
      res.end(JSON.stringify({
          name:'xxx',
          age:18
        }))
        break;
    default:
      res.end('404')
      break;
  }
})
app.listen(8080,()=>{
  console.log('server is running at port 8080');
})
```
## 4. http请求猫眼
### get请求
```html

<script>
  // https://i.maoyan.com/#movie
fetch("https://i.maoyan.com/api/mmdb/movie/v3/list/hot.json?ct=%E5%B9%BF%E5%B7%9E&ci=20&channelId=4")
.then(res=>res.json())
.then(data=>{
  console.log(data);
})
</script>
```

```js
const http=require('http')
const https=require('https')
const url=require('url')

const app=http.createServer((req,res)=>{
  res.writeHead(200,{
    "Content-Type":"application/json;charset=utf-8",
    'Access-Control-Allow-Origin':'*'
  })
  switch (urlObj.pathname) {
    case '/api/aaa':
      // 去猫眼要数据
    httpget(res)
      break;
    // 回调函数方式
    case '/api/login':
      httpget2((data)=>{
        res.end(data)
      })
    default:
      res.end('404')
      break;
  }
})
app.listen(8080,()=>{
  console.log('server is running at port 8080');
})

function httpget(resp) {
  var data=""
  https.get(`https://i.maoyan.com/api/mmdb/movie/v3/list/hot.json?ct=%E5%B9%BF%E5%B7%9E&ci=20&channelId=4`,
  (res)=>{
      res.on("data",(chunk)=>{
        data+=chunk
      })
      res.on("end",()=>{
        resp.end(data)
      })
  })
}
```
```js{9,10,11}
// 回调函数编程方式
function httpget2(cb) {
  var data=""
  https.get(`https://i.maoyan.com/api/mmdb/movie/v3/list/hot.json?ct=%E5%B9%BF%E5%B7%9E&ci=20&channelId=4`,
  (res)=>{
      res.on("data",(chunk)=>{
        data+=chunk
      })
      res.on("end",()=>{
        cb(data)
      })
  })
}
```
### post请求
```html
<script>
  // https://i.maoyan.com/#movie
fetch("https://i.maoyan.com/api/mmdb/movie/v3/list/hot.json?ct=%E5%B9%BF%E5%B7%9E&ci=20&channelId=4")
.then(res=>res.json())
.then(data=>{
  console.log(data);
})
</script>
```
```js
const http=require('http')
const https=require('https')
const url=require('url')

const app=http.createServer((req,res)=>{
  res.writeHead(200,{
    "Content-Type":"application/json;charset=utf-8",
    'Access-Control-Allow-Origin':'*'
  })
  switch (urlObj.pathname) {方式
    case '/api/login':
      httppost((data)=>{
        res.end(data)
      })
    default:
      res.end('404')
      break;
  }
})
app.listen(8080,()=>{
  console.log('server is running at port 8080');
})

// 回调函数编程方式
function httppost(cb) {
  var data=""
  var options={
    hostname:"xxx",
    port:"xxx",
    path:"xxx",
    method:"POST",
    headers:{
      "Content-Type":"application/json"
    }
  }
  var req=https.request(options,(res)=>{
    res.on("data",(chunk)=>{
      data+=chunk
    })
    res.on("end",()=>{
      cb(data)
    })
  })
  req.write(JSON.stringify([{},{"baseParam":{"ypClient":1}}]))
  req.end()
}
```
## 5. event
### 订阅发布模式
```js

const EventEmitter=require('events')
const event=new EventEmitter()

event.on("play",()=>{
  console.log("xxx")
})

event.emit("play")

setTimeout(()=>{
  event.emit("play","xxx")
},2000)
```

```js
const http=require('http')
const https=require('https')
const url=require('url')
const EventEmitter=require('events')
// !!
var event=null

const app=http.createServer((req,res)=>{
  res.writeHead(200,{
    "Content-Type":"application/json;charset=utf-8",
    'Access-Control-Allow-Origin':'*'
  })
  switch (urlObj.pathname) {
    case '/api/aaa':
      event=new EventEmitter()
      event.on("play",(data)=>{
        res.end(data)
      })
      // 去猫眼要数据
    httpget(res)
      break;
    default:
      res.end('404')
      break;
  }
})
app.listen(8080,()=>{
  console.log('server is running at port 8080');
})

function httpget(resp) {
  var data=""
  https.get(`https://i.maoyan.com/api/mmdb/movie/v3/list/hot.json?ct=%E5%B9%BF%E5%B7%9E&ci=20&channelId=4`,
  (res)=>{
      res.on("data",(chunk)=>{
        data+=chunk
      })
      res.on("end",()=>{
        event.emit("play")
      })
  })
}

```
## 6. fs
### 文件操作 读写删
const fs=require('fs')

``` js
const fs=require('fs')

fs.mkdir("./avatar",(err)=>{
  i(err&& err.code==="EEXIST"){
    console.log('err')
  }
})

fs.rename("./avatar","./avatar2",(err)=>{
  i(err&& err.code==="ENOENT"){
    console.log('err')
  }
})

fs.rmdir("./avatar2",(err)=>{
  i(err&& err.code==="ENOENT"){
    console.log('err')
  }
})

fs.writeFile("./avatar2/a.txt","xxx",(err)=>{
    console.log('err')
})

fs.appendFile("./avatar2/a.txt","\nxxx",(err)=>{
    console.log('err')
})

fs.readFile("./avatar2/a.txt","utf-8",(err,data)=>{
    if(!err){
      console.log(data.toString("utf-8"))
    }
})

fs.unlink("./avatar2/a.txt",(err)=>{
    console.log('err')
})

fs.readdir("./avatar",(err,data)=>{
  console.log(data)
})

fs.stat("./avatar",(err,data)=>{
  console.log(data)
  data.isFile()
  data.isDirectory()
})
```
### 删除文件
```js
// 异步写法
const fs=require('fs')
fs.readdir("./xxx",(err,data)=>{
  data.forEach((item)=>{
    // 回调地狱
    fs.unlink(`./avatar/${item}`,(err)=>{})
  })

  fs.rmdir("./avatar")
})
```
```js
// 异步写法 不用回调地狱
const fs=require('fs').promises
fs.mkdir("./avatar").then(data=>{
  console.log(data)
})
fs.readFile("./avatar/xxx.txt","utf-8").then(data=>{
  console.log(data)
})
```
```js
// 同步写法 同步删除
const fs=require('fs')

try{
  fs.mkdirSync("./avatar")
}catch(err){
  console.log('xxx',err)
}

fs.readdir("./xxx",(err,data)=>{
  data.forEach((item)=>{
    // 回调地狱
    fs.unlinkSync(`./avatar/${item}`,(err)=>{})
  })

  fs.rmdir("./avatar")
})
```
```js
// 同步写法 同步删除
const fs=require('fs').promises

try{
  fs.mkdirSync("./avatar")
}catch(err){
  console.log('xxx',err)
}

fs.readdir("./xxx").then((data)=>{
  data.forEach(item=>{
    console.log(item)
  })
})
```
### 如何等待for循环里面所有的异步都执行完再执行下面的代码
基于Promises风格
Promise.all([])
```js
const fs=require('fs').promises

try{
  fs.mkdirSync("./avatar")
}catch(err){
  console.log('xxx',err)
}

fs.readdir("./xxx").then((data)=>{
  let arr =[]
  data.forEach(item=>{
    arr.push(fs.unlink(`./avatar/${item}`))
  })
  Promise.all(arr).then(res=>{})
  fs.rmdir("./avatar")
})
```
```js
fs.readdir("./xxx").then(async (data)=>{
  let arr =[]
  data.forEach(item=>{
    arr.push(fs.unlink(`./avatar/${item}`))
  })
  await Promise.all(arr)
  await fs.rmdir("./avatar")
})
```
```js
fs.readdir("./xxx").then(async (data)=>{
  await Promise.all(data.map((item)=>{
    fs.unlink(`./avatar/${item}`)
  }))
  await fs.rmdir("./avatar")
})
```
### stream流
```js
var fs=require('fs')
var rs=fs.createReadStream('sample.txt','utf-8')

rs.on('data',function(chunk){
  console.log(chunk)
})

rs.on('end',()=>{
  console.log('end')
})

rs.on('error',(err)=>{
  console.log(err)
})

var ws=fs.createWriteStream('sample.txt','utf-8')
ws.write('xxx')
ws.end()
```
- 管道--控制流速
```js
const fs=require('fs')
var rs=fs.createReadStream('sample.txt','utf-8')
var ws=fs.createWriteStream('sample.txt','utf-8')

rs.pipe(ws)
```
- zlib压缩文件大小
```js
const fs=require('fs')
const zlib=require('zlib')
gzip=zlib.createGzip()
var rs=fs.createReadStream('sample.txt','utf-8')
var ws=fs.createWriteStream('sample.txt','utf-8')

rs.pipe(gzip).pipe(ws)
```

```js

const http=require('http')
const https=require('https')
const url=require('url')
const fs=require('fs')
const zlib=require('zlib')
const EventEmitter=require('events')
gzip=zlib.createGzip()
// !!
var event=null

http.createServer((req,res)=>{
  const readStream=fs.createReadStream("./index.js")
  res.writeHead(200,{
    "Content-Type":"application/json;charset=utf-8",
    'Access-Control-Allow-Origin':'*',
    "Content-Encoding":"gzip"
  })
  readStream..pipe(gzip).pipe(res)
}).listen(8080,()=>{
  console.log('server is running at port 8080');
})
```
## 7. 自己实现路由功能
