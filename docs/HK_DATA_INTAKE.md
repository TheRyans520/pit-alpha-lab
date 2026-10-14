# 港股数据接入：先采集与审计，再做研究

本阶段新增富途只读行情适配器，尚未取得本机真实行情。离线契约测试使用虚构数据；
安装 SDK 和测试通过不代表账号行情权限通过，也不代表旧 CSI300 结果重新得到验证。

## 这台电脑的下一步

1. 从[富途官方页面](https://openapi.futunn.com/futu-api-doc/quick/opend-base.html)安装并打开 OpenD。
2. 自行注册/登录富途牛牛账号，完成服务方要求的确认；不要把密码或验证码放进项目。
   根据 2026-10-14 查阅的官方文档，OpenD 登录不要求先开证券交易账户。
   实际行情权限、地区限制与额度以账号返回结果及服务方条款为准。
3. 保持 OpenD 的本地接口为 `127.0.0.1:11111`。这里只使用行情接口，不创建交易上下文。

在此项目目录执行（本机 SDK 已安装；其他机器先安装依赖）：

```bash
cd /Users/panyuxuan/Projects/codexProj/pitAlphaLab/pit-alpha-lab
.venv/bin/python -m pip install -r requirements-futu.txt
.venv/bin/python -m pitalpha hk-data doctor
.venv/bin/python -m pitalpha hk-data plan configs/hk_futu_pilot.yaml
.venv/bin/python -m pitalpha hk-data collect configs/hk_futu_pilot.yaml
.venv/bin/python -m pitalpha hk-data verify data/futu_hk/<本次目录>
```

`doctor` 只检查安装版本和 TCP 端口，不消耗历史行情额度；端口连通不能证明登录或
行情权限有效。`plan` 不联网。`collect` 才请求账号额度、历史 K 线和复权事件；会消耗
服务方适用的历史标的额度。失败后不会自动反复重试，需要查看 OpenD 的登录/权限状态。
官方 SDK 可能在用户目录的 `.com.futunn.FutuOpenD/Log` 生成本地诊断日志。

默认三只标的是小规模接口检查样本，日期为 2024–2025，既不代表历史指数成分股，
也不代表未观察测试集。样本目的和日期明确记录在配置中。现阶段不据此生成收益排名。

## 数据文件和检查

- `raw/*.parquet`：逐页保存 SDK 返回的原始字段表，以及每只股票的复权事件表。
  这里的“原始”指 SDK 解码后的响应，不是网络报文，也不是交易所原始逐笔记录。
- `bars.parquet`：仅保留未复权日线 OHLC、成交量、成交额和本地交易日期。
  不把响应中的 PE 等字段直接变成历史可用特征。
- `quality.json`：结构检查、每只标的覆盖区间、相对样本日期并集的缺失日期。
  日期并集不是交易所日历；缺失日期不能直接认定为停牌。
- `snapshot_manifest.json`：请求、SDK/源码版本、采集起止时间、每次响应接收时间、
  文件校验值与限制。分页数据以完整下载为前提；失败目录保留 `failed` 状态和已取得文件。

检查包括代码/日期一致性、重复记录、非有限值、OHLC 顺序、负成交量/成交额和分页循环。
归一化数据不去重、不补价、不自动复权。已下载的复权事件仅用于后续审计，未用于构造收益；
其历史公告可用时间仍未知。请求结束日期必须早于香港当日，避免采集未结束的日线。

`verify` 离线核验所有产物哈希，再从保存的原始分页表重建规范化数据和质量报告。
即使有人更新了篡改后规范化表的哈希，只要它与原始响应不一致，检查也会失败。
校验不能证明原始供应商数据正确，也不是数字签名或防恶意重写的信任机制。

成功状态为 `downloaded_unvalidated`，且始终 `research_ready: false`。
这些数据不进入现有研究 API 的已完成实验列表。退出码 0 表示采集/离线检查完成，
不表示历史完整性或经济学验证通过。默认存放在 Git 忽略的 `data/futu_hk/`；
第三方数据不上传 GitHub，公开演示前需另行确认展示权限。

## 取得首批数据后的验收

1. 核查覆盖长度、缺失日期、未复权价格、复权事件与第二来源的抽样差异。
2. 补独立港股交易日历、证券生命周期、停牌和历史股票池证据。
3. 明确公司行动如何改变现金/股数，以及港股费用、整手和交易规则。
4. 在上述输入足够时连接现金执行引擎，随后锁定研究协议和新的验证样本。

港股是独立案例，不能替代 A 股原始分区来证明旧 CSI300 收益。

## 接口依据（2026-10-14 查阅）

- [权限与额度](https://openapi.futunn.com/futu-api-doc/intro/authority.html)
- [历史 K 线](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)：显式指定 `K_DAY`、`AuType.NONE`，分页读取。
- [复权因子](https://openapi.futunn.com/futu-api-doc/quote/get-rehab.html)：独立保存，不假设现在下载的事件具有历史发布时点。
- [SDK 10.11.7108](https://pypi.org/project/futu-api/10.11.7108/)：锁定直接依赖；manifest 记录实际 SDK 版本。

离线验收：`python -m unittest tests.test_futu_hk -v`。所有 fixture 都是虚构价格；
测试覆盖额度不足、分页循环、分页重叠、缺失复权响应、供应商失败、异常行情和篡改产物。
