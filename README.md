# 基于深度学习的个性化影视推荐系统

### 基于Flask的API服务
flask入口文件：server/app.py
### 测试推荐模型
```
python /server/feedTopN.py # NCF模型
```
```
python /server/feedTopN.py # FM模型
```

### 个性化推荐接口说明
```
url: '/getTopK', // 推荐引擎对外接口
params: {
    user_id: '32039882', // 用户id
    k: 10 // 取前k个结果
},
response: {
    status: 1,
    results: [
        '141231234', // 影片id
        // ...
    ],
    msg: '操作成功'
}
```