# Stock Monitor Tool

本地股票监控与宏观周期面板。

## 启动

```bash
cd stock-monitor-tool
python3 -m pip install -r requirements-stock-scanner.txt
PORT=8001 python3 api/stock_server.py
```

打开：

```text
http://localhost:8001/static/stock_dashboard.html
```

兼容旧地址：

```text
http://localhost:8001/stock_dashboard.html
```

## 目录结构

```text
api/      HTTP 服务入口与后台 worker
bin/      启动和批处理脚本
core/     选股、宏观、技术、估值等业务逻辑
db/       SQLite 持久化辅助
static/   Dashboard 静态页面
test/     脚本式测试
conf/     手工配置与小型配置数据
logs/     运行日志
docs/     文档
plugins/  探测脚本和可选工具
```
