# Node.js
## 内置模块学习
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
console.log(myURL.pathname)
console.log(myURL.searchParams.get('id'))
console.log(myURL.searchParams.has('id'))
// searchParams是一种迭代器
for (var [key,value] of myURL.searchParams) {
  console.log(key,value)
}
```
