# 代理问题解决方案

## 问题诊断

从测试来看，你的系统有一个代理在干扰 AkShare 访问东方财富（East Money）的域名，导致 API 调用失败：
- `ConnectionError: Remote end closed connection without response`
- AkShare 虽然能成功获取股票列表，但获取历史 K 线数据时会失败

## 解决方案

### 方案一：临时关闭系统代理（推荐）

在运行扫描前，临时关闭你的系统代理或 VPN，然后：

```bash
# 清空所有代理环境变量
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy REQUESTS_CA_BUNDLE

# 重新启动扫描
./start_stock_tool.sh 1000 200
```

### 方案二：配置代理例外

如果你使用的是 Clash、Surge 或其他代理工具，请在规则中添加东方财富域名的**直接连接**（DIRECT）：

- `*.eastmoney.com`
- `push2.eastmoney.com`
- `*.push2.eastmoney.com`
- `*.sina.com.cn`

确保这些域名不走代理。

### 方案三：使用旧数据（暂用）

如果暂时无法解决代理问题，可以先使用你之前可能已经有的旧数据：

```bash
# 检查是否有旧 CSV 文件
ls -la *.csv
```

## 当前已修复的内容

✅ 已在 `stock_scanner.py` 和 `sepa_stage2_scanner.py` 开头添加代码，自动清除代理相关环境变量  
✅ 已恢复脚本到原来的简单单线程版本  
✅ Dashboard 服务器正在运行（http://localhost:8001/stock_dashboard.html）

## 下一步

1. 关闭系统代理 / VPN
2. 重新运行 `./start_stock_tool.sh 1000 200`
3. 等待扫描完成，数据会自动显示在 Dashboard 上
