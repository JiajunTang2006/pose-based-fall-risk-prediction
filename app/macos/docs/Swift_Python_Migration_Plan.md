# FallGuard Swift 原生界面 + Python AI 服务迁移策划书

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 项目名称 | FallGuard macOS 原生化改造 |
| 改造方向 | SwiftUI/AppKit 原生界面 + Python 本地 AI 服务 |
| 当前版本基线 | FallGuard 0.2.0 |
| 目标平台 | macOS 11 及以上，开发阶段优先支持当前开发机系统 |
| 核心原则 | 算法结果不变、渐进迁移、随时可回退、视频不出本机 |
| 建议周期 | 6～9 周（1 名熟悉 Swift 和 Python 的开发者） |
| 第一阶段体积目标 | 安装后 550～650 MB，最终以实测为准 |

## 2. 项目背景

FallGuard 当前采用 Python 产品层与算法层、HTML/CSS/JavaScript 页面以及 PySide6/pywebview 桌面容器。Python 侧负责摄像头、YOLO 姿态估计、特征提取、跌倒预测、风险状态机、事件处理、SQLite 数据和媒体证据。

当前工程已经具备较清晰的算法合同：算法输出 `Prediction`，再由 `FrameBusinessProcessor` 统一处理 Camera 与 Import Media 两条业务路径。现有 Python Web 服务也已经提供状态、视频流、启停、设置和 Profiles 等接口，因此适合先将其整理成无界面的本地 AI 服务，再由 Swift 原生应用接管用户界面和 macOS 系统能力。

本次改造不以“全部改写成 Swift”为目标。第一阶段保留 Python 推理、业务状态机、事件判定和数据库逻辑，避免模型结果及历史数据行为发生变化。

## 3. 改造目标

### 3.1 产品目标

1. 使用 SwiftUI 为主、AppKit 为辅构建真正的 macOS 原生界面。
2. 改善应用启动、窗口交互、菜单栏、权限提示、通知和系统主题适配。
3. 保留现有 YOLO、OpenCV、scikit-learn、`.pt` 与 `.joblib` 模型。
4. 所有摄像头帧、推理结果和历史数据继续只在用户本机处理。
5. 保证 Camera、Import Media、事件记录和风险判断与当前版本行为一致。
6. 精简 Python 打包内容，将安装体积由当前约 894 MB 降至约 550～650 MB。

### 3.2 非目标

第一阶段暂不进行以下工作：

- 不把 YOLO/PyTorch 推理整体迁移到 Core ML。
- 不重新训练分类模型。
- 不把 `Prediction`、风险状态机和事件合并规则重写为 Swift。
- 不改变数据库 schema 和用户数据目录。
- 不立即支持 Windows、Linux、iPhone 或 iPad。
- 不加入云端视频上传或远程推理。

## 4. 总体架构

```text
FallGuard.app
├── Swift 原生应用
│   ├── SwiftUI 页面与 AppKit 系统集成
│   ├── PythonServiceManager（进程启动、健康检查、退出）
│   ├── APIClient（状态、设置、Profiles、历史记录）
│   ├── PreviewClient（本地视频预览）
│   └── Notification/MenuBar/Permissions
│
├── Resources/AIService/
│   ├── fallguard-ai（PyInstaller 打包的无界面服务）
│   ├── Python 运行时及必要依赖
│   └── models/
│
└── 用户数据（保持在应用包之外）
    ├── ~/Library/Application Support/FallGuard/
    └── ~/Movies/FallGuard/
```

### 4.1 职责划分

| 能力 | 第一阶段负责人 | 说明 |
|---|---|---|
| 窗口、导航、主题、多语言 | Swift | 完全替换现有网页/PySide6 UI |
| 菜单栏、系统通知、摄像头权限说明 | Swift | 使用 macOS 原生 API |
| Python 子进程生命周期 | Swift | 启动、探活、异常重启、退出清理 |
| 摄像头采集 | Python | 第一阶段保持现状，避免重复传输原始帧 |
| YOLO、OpenCV、特征和 ML 推理 | Python | 保持算法实现与模型不变 |
| 风险状态机与事件判定 | Python | 保持结果一致性 |
| SQLite 写入和媒体证据 | Python | 第一阶段保持唯一写入方 |
| 历史数据展示 | Swift 通过 API 获取 | 不允许 Swift 与 Python 同时写 SQLite |

## 5. 本地服务协议设计

### 5.1 通信方式

第一版采用仅监听 `127.0.0.1` 的 HTTP 服务。服务启动时选择随机空闲端口，并生成本次运行专用令牌。Swift 从子进程标准输出读取一行启动信息：

```json
{"event":"ready","port":49321,"token":"<random-token>","api_version":"v1"}
```

后续请求携带：

```http
Authorization: Bearer <random-token>
```

服务不得监听 `0.0.0.0`，不得接受无令牌的业务请求。日志不得输出令牌、用户路径中的敏感信息或图像内容。

### 5.2 第一阶段接口

现有接口先整理和版本化，不在 Swift 中直接依赖零散内部字段。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 服务、模型、数据库就绪状态 |
| GET | `/api/v1/status` | 当前监控、风险、FPS、设备和任务状态 |
| POST | `/api/v1/monitor/start` | 启动摄像头监控 |
| POST | `/api/v1/monitor/stop` | 停止监控 |
| GET | `/api/v1/preview.mjpg` | 本机预览流，过渡期复用 MJPEG |
| GET/PUT | `/api/v1/settings` | 获取或更新设置 |
| GET | `/api/v1/cameras` | 摄像头列表 |
| GET/POST | `/api/v1/profiles` | Profiles 查询和创建 |
| PUT/DELETE | `/api/v1/profiles/{id}` | 更新或删除 Profile |
| POST | `/api/v1/profiles/{id}/activate` | 激活 Profile |
| GET | `/api/v1/events` | 分页查询事件 |
| GET | `/api/v1/sessions` | 分页查询监控 Session |
| POST | `/api/v1/imports` | 创建媒体导入任务 |
| GET | `/api/v1/imports/{id}` | 查询导入进度和结果 |
| POST | `/api/v1/shutdown` | Swift 正常退出时通知服务关闭 |

### 5.3 预测状态合同

对外统一使用明确枚举和值域，不直接序列化 Python 类：

```json
{
  "schema_version": 1,
  "sequence": 1842,
  "timestamp_ms": 1783853200123,
  "monitoring": true,
  "prediction": {
    "state": "Normal",
    "alert_state": "Normal",
    "business_state": "safe",
    "risk_score": 0.18,
    "visibility": 0.93,
    "confidence": 0.93,
    "system_status": null
  },
  "performance": {
    "fps": 24.6,
    "frame_index": 1842
  }
}
```

约束：

- `risk_score`、`visibility`、`confidence` 必须在 `0.0～1.0`。
- Python 状态继续保留 `Normal`、`Pre-fall`、`Fall`、`Unknown`。
- Swift 展示文案通过本地化映射生成，不直接展示后端英文枚举。
- 所有响应携带 `schema_version`，破坏性改动必须升级 API 版本。
- 错误统一返回稳定的 `code`、可本地化的 `message_key` 和可选 `details`。

## 6. Swift 工程规划

建议在 `macos/native/FallGuard/` 创建 Xcode 工程：

```text
native/FallGuard/
├── App/
│   ├── FallGuardApp.swift
│   └── AppDelegate.swift
├── Models/
│   ├── PredictionDTO.swift
│   ├── MonitorStatus.swift
│   └── APIError.swift
├── Services/
│   ├── PythonServiceManager.swift
│   ├── FallGuardAPIClient.swift
│   ├── PreviewClient.swift
│   ├── NotificationService.swift
│   └── PermissionService.swift
├── Features/
│   ├── Dashboard/
│   ├── Events/
│   ├── ImportMedia/
│   ├── Profiles/
│   └── Settings/
├── Components/
├── Resources/
│   ├── Assets.xcassets
│   ├── Localizable.xcstrings
│   └── AIService/
└── Tests/
```

界面状态建议采用 SwiftUI Observation 或 `ObservableObject`，网络层使用 `URLSession` 和 `async/await`。业务 DTO 使用 `Codable`。不要在 View 中直接拼 URL、启动进程或访问数据库。

## 7. 分阶段实施计划

### 阶段 0：基线冻结与验收样本（2～3 天）

任务：

- 固定一组摄像头录制片段和导入测试视频。
- 记录当前版本每段视频的状态序列、最高风险、事件数和处理耗时。
- 运行并保存现有 Python 全量测试结果。
- 记录当前 `.app` 的安装体积、冷启动时间、模型加载时间和稳定运行内存。
- 备份测试数据库，验证旧数据可以读取。

交付物：基线报告、回归视频、预期结果 JSON、性能与体积表。

退出标准：后续所有阶段都有可重复比较的输入和结果。

### 阶段 1：提取无界面 Python AI 服务（5～7 天）

任务：

- 新增独立服务入口，例如 `fall_prediction_service.__main__`。
- 复用现有 `Prediction`、`FrameBusinessProcessor`、数据库和事件服务。
- 将现有未版本化接口整理到 `/api/v1/`。
- 增加随机端口、运行令牌、结构化启动消息和优雅退出。
- 增加健康检查、模型加载错误、摄像头占用和数据库错误的稳定错误码。
- 将 UI、菜单栏、PySide6、pywebview 和 Tkinter 从服务运行路径移除。

退出标准：不启动任何 Python 窗口时，命令行可以完成启动监控、获取预览、停止监控和媒体导入；基线视频结果一致。

### 阶段 2：Swift 应用骨架与进程管理（4～6 天）

任务：

- 创建 SwiftUI macOS 工程、签名配置和基础导航。
- 实现 `PythonServiceManager`，从 App Bundle 启动 AI 服务。
- 读取 `ready` 消息，执行健康检查并维护服务状态。
- 处理启动超时、进程崩溃、端口冲突、重复启动和应用退出。
- 实现开发模式：允许连接源码启动的 Python 服务，方便调试。

退出标准：Swift 应用可以可靠地启动和关闭 Python 服务；连续启动退出 20 次无残留服务进程。

### 阶段 3：核心监控界面（7～10 天）

任务：

- 实现 Dashboard、开始/停止、风险状态、风险分数、FPS 和系统提示。
- 接入 MJPEG 预览；限制刷新和解码开销，界面不可阻塞主线程。
- 实现摄像头选择、敏感度设置和错误恢复提示。
- 实现菜单栏入口、显示主窗口、开始/停止和退出。
- 实现中英文文案和深色/浅色主题。

退出标准：摄像头实时监控可连续运行 2 小时；窗口隐藏与恢复不影响推理；状态与旧界面一致。

### 阶段 4：完整产品功能迁移（7～10 天）

任务：

- 实现 Profiles、设置、事件列表和 Session 历史。
- 实现 Import Media 选择、进度、取消、结果和输出文件定位。
- 实现确认跌倒后的原生通知和声音设置。
- 实现空状态、加载状态、错误状态及辅助功能标签。
- 保证旧数据库、旧媒体记录和用户设置兼容。

退出标准：当前 README 列出的用户功能均可在 Swift 应用完成，Python UI 不再是正常运行所需组件。

### 阶段 5：打包、精简与性能优化（5～7 天）

任务：

- 为 AI 服务建立独立 PyInstaller spec，只包含服务真实依赖。
- 排除 PySide6、shiboken6、pywebview、rumps、Tkinter 和 Python UI 模块。
- 验证并排除推理路径不使用的 Polars、Matplotlib、训练工具和测试依赖。
- 将 AI 服务、模型和配置嵌入 Swift `.app` Resources。
- 修正资源查找逻辑，区分只读 App Bundle 与可写用户目录。
- 完成嵌套二进制签名、应用签名、权限描述和公证准备。
- 对比精简前后启动时间、内存、FPS 和模型结果。

退出标准：安装体积目标为 550～650 MB；若超出，必须提供逐项体积分析。精简前后预测合同测试和基线视频结果通过。

### 阶段 6：灰度替换与旧界面退役（3～5 天）

任务：

- 先保留旧 Python UI 作为开发回退入口，不打入正式发布包。
- 完成至少一轮真实摄像头场景验收和异常恢复测试。
- 更新 README、构建说明、故障排查和版本发布记录。
- 正式构建只以 Swift 应用作为入口。
- 稳定一个版本后再删除不再使用的 UI 代码。

退出标准：不存在必须打开旧 UI 才能完成的用户流程；回滚方案经过验证。

## 8. 工期与里程碑

| 里程碑 | 内容 | 累计时间 |
|---|---|---:|
| M1 | 基线冻结、Python 服务可独立运行 | 第 1～2 周 |
| M2 | Swift 能可靠管理服务进程 | 第 3 周 |
| M3 | 原生 Dashboard 和实时监控可用 | 第 4～5 周 |
| M4 | 设置、Profiles、历史、导入完整 | 第 6～7 周 |
| M5 | 打包精简、回归验收、发布候选版 | 第 8～9 周 |

若开发者刚开始学习 Swift，建议增加 2～4 周缓冲。若只先交付 Dashboard、启停与状态显示，可在约 3～4 周形成最小可用版本。

## 9. 测试与验收方案

### 9.1 自动化测试

- Python 单元测试：保留当前预测合同、状态机、事件、数据库和媒体缓冲测试。
- API 合同测试：校验字段、枚举、值域、错误码、认证和版本号。
- 黄金样本测试：相同视频输入下比较状态序列、风险分数和事件数量。
- Swift 单元测试：DTO 解码、API 错误映射、进程状态转换和重试策略。
- Swift UI 测试：启动、停止、切换设置、打开事件、媒体导入。
- 构建测试：确认发布包中不存在 PySide6、pywebview 和开发测试资源。

### 9.2 关键验收指标

| 指标 | 验收要求 |
|---|---|
| 算法一致性 | 同一基线视频的最终状态和事件数完全一致；风险分数误差不超过序列化精度 |
| 实时性能 | FPS 不低于旧版本的 95% |
| 冷启动 | 应用快速显示原生壳；模型后台加载状态清晰，无界面假死 |
| 稳定性 | 连续监控 2 小时无崩溃、无持续内存增长 |
| 退出清理 | 正常退出后无 Python 子进程残留 |
| 数据兼容 | 原有数据库、Profiles、事件和设置可继续使用 |
| 网络边界 | 服务只监听 `127.0.0.1`，无令牌访问被拒绝 |
| 安装体积 | 目标 550～650 MB，体积变化有可追溯清单 |
| 隐私 | 不向外网发送摄像头帧、预测结果或用户数据 |

## 10. 风险与应对

| 风险 | 影响 | 应对措施 |
|---|---|---|
| Swift 与 Python 状态不同步 | 按钮和真实监控状态不一致 | Python 作为状态真源；请求使用幂等操作和递增 `sequence` |
| 子进程崩溃或无法启动 | 应用不可监控 | 健康检查、明确错误码、有限次数自动重启、提供日志导出 |
| MJPEG 占用较高 | CPU 与内存增加 | 限制预览分辨率/FPS；后续评估 Unix Socket 或共享内存 |
| SQLite 双写导致锁和损坏 | 历史数据异常 | 第一阶段仅 Python 写库；Swift 只通过 API 读写业务数据 |
| PyInstaller 漏包 | 发布包在开发机外启动失败 | 在干净用户环境验证；建立依赖白名单和发布包冒烟测试 |
| 精简依赖改变模型结果 | 漏报或误报 | 每次排除依赖后运行黄金视频与合同测试 |
| macOS 签名/公证失败 | 无法正常分发 | 先签嵌套 AI 服务，再签主应用；尽早做发布构建验证 |
| 摄像头权限归属混乱 | 首次启动无法取流 | 明确由主应用发起并解释权限；开发期验证打包后真实权限行为 |
| Swift 学习成本低估 | 工期延误 | 先完成纵向最小闭环，再扩展页面；预留 2～4 周学习缓冲 |

## 11. 回滚策略

1. 改造期间保留现有 Python UI 启动入口和构建方式。
2. Python AI 服务复用现有算法与数据库，不升级 schema 即不改变用户数据格式。
3. Swift 版本使用独立构建产物和版本号，不覆盖稳定版本。
4. 新版本异常时，用户可退出 Swift 版本并重新启动旧版本；两者不得同时运行监控。
5. 删除旧 UI 代码必须安排在 Swift 版本稳定发布一个周期之后。

## 12. 体积优化预算

当前已观察到的主要占用约为：PyTorch 320 MB、Polars Runtime 181 MB、OpenCV 118 MB、PySide6 77 MB、SciPy 38 MB、模型约 17～21 MB、scikit-learn 17 MB、Matplotlib 15 MB。

第一阶段确定可移除的是 Python UI 容器及其依赖，预计直接减少约 70～120 MB。若验证 Polars、Matplotlib 和训练相关模块不在运行路径内，整体预计可减少约 250～350 MB。Swift 主程序自身的新增体积相对较小。

注意：当前 `FallGuard.spec` 已声明排除 Polars，但现有 `dist/FallGuard` 中仍观察到约 181 MB 的 Polars Runtime，说明该目录可能来自旧构建或排除规则未在对应产物生效。正式估算前应清理到新的独立输出目录进行一次可复现构建，不能直接把现有目录当作最终结果。

## 13. 后续可选第二阶段

Swift 原生版本稳定后，可以单独立项评估：

- 将 YOLO 导出为 Core ML 或 ONNX。
- 使用 Vision/Core ML 或 ONNX Runtime 替代 Ultralytics + PyTorch 运行时。
- 将确定稳定的特征提取逻辑迁移为 Swift 或 C++。
- 最终移除 Python 运行时，将安装体积进一步降低到约 100～250 MB。

该阶段必须重新验证推理精度、关键点坐标、预处理、后处理和时序结果，不与本次 UI 迁移同时进行。

## 14. 启动决策与第一批任务

建议批准以下实施边界：

1. 采用 SwiftUI/AppKit + Python 本地服务架构。
2. 第一阶段由 Python 继续拥有摄像头、预测、状态机、事件和数据库写入。
3. 复用现有 HTTP/MJPEG 能力，但新增 `/api/v1`、随机端口和令牌认证。
4. 先实现 Dashboard 纵向闭环，再迁移设置、Profiles、历史和导入功能。
5. 以算法一致性和稳定性为上线硬门槛，以 550～650 MB 为体积优化目标。

批准后第一批开发任务为：建立回归基线、创建 `fall_prediction_service` 入口、定义 API schema、添加服务合同测试，以及创建 Swift 工程骨架和 `PythonServiceManager` 原型。

---

## 15. 逐文件改造清单

本节是实际施工清单。除非某一步的测试已经通过，否则不要提前删除旧代码。

### 15.1 Python 新增文件

```text
src/fall_prediction_service/
├── __init__.py          # 服务版本、API schema 版本
├── __main__.py          # 无界面服务入口
├── app.py               # 组装数据库、Monitor、Import Processor 和 HTTP Server
├── server.py            # /api/v1 路由、请求解析、响应输出
├── auth.py              # Bearer Token 校验
├── contracts.py         # 对外 DTO 与枚举，不暴露内部 Python 对象
├── errors.py            # 稳定错误码和 HTTP 映射
├── lifecycle.py         # signal、优雅退出、进程状态
└── serialization.py     # Prediction/Session/Event 到 JSON 的转换
```

新增测试：

```text
tests/
├── test_service_auth.py
├── test_service_contract.py
├── test_service_lifecycle.py
├── test_service_monitor_api.py
├── test_service_settings_api.py
├── test_service_profiles_api.py
├── test_service_events_api.py
└── test_service_import_api.py
```

### 15.2 Python 修改文件

| 文件 | 具体修改 | 第一阶段是否删除旧逻辑 |
|---|---|---|
| `src/fall_prediction_desktop/web_app.py` | 将可复用的 Camera Monitor、Media Processor 与 UI 路由解耦；先允许服务层调用现有类 | 否 |
| `src/fall_prediction_desktop/frame_pipeline.py` | 不改变算法行为；只补序列化需要的稳定字段测试 | 否 |
| `src/fall_prediction_desktop/database/repositories/*.py` | 增加分页查询和事件详情方法 | 否 |
| `src/fall_prediction_desktop/paths.py` | 增加日志目录、服务临时目录；保持现有数据目录不变 | 否 |
| `src/fall_prediction_desktop/runner.py` | 增加可取消的 Import Job 和线程安全进度回调 | 否 |
| `pyproject.toml` | 增加 `fallguard-ai` 命令入口；拆分 UI 与 service 可选依赖 | 否 |
| `FallGuard.spec` | 保留旧构建；新建服务专用 spec，不直接覆盖 | 否 |
| `build_app.sh` | 最终由新的统一构建脚本替代，迁移期仍可构建旧版 | 否 |

建议在迁移稳定后再把共享业务类从 `fall_prediction_desktop` 移到不含 UI 含义的包，例如 `fall_prediction_product`。迁移初期不要一边改包名一边改通信协议，以免回归面过大。

### 15.3 Swift 新增文件

| 文件 | 职责 |
|---|---|
| `FallGuardApp.swift` | SwiftUI 入口、WindowGroup、Settings Scene |
| `AppDelegate.swift` | 菜单栏、应用退出、系统通知代理 |
| `PythonServiceManager.swift` | 子进程启动、ready 解析、探活、关闭、异常状态 |
| `FallGuardAPIClient.swift` | 所有 HTTP 调用、Token Header、超时与错误解码 |
| `ServiceModels.swift` | Health、Status、Prediction、Settings、Profile、Event DTO |
| `AppStore.swift` | 全局可观察状态和页面动作编排 |
| `StatusPoller.swift` | 定时拉取 `/status`，前后台自动调整频率 |
| `PreviewClient.swift` | JPEG/MJPEG 获取和图像解码 |
| `DashboardView.swift` | 主监控页面 |
| `EventsView.swift` | 事件分页列表与详情 |
| `ImportMediaView.swift` | 文件选择、进度、取消和结果 |
| `ProfilesView.swift` | Profiles 增删改和激活 |
| `SettingsView.swift` | 摄像头、敏感度、语言、主题、声音设置 |
| `MenuBarController.swift` | 菜单栏状态和快捷操作 |
| `NotificationService.swift` | 原生通知权限与跌倒提醒 |
| `DiagnosticsView.swift` | 服务版本、模型状态、日志导出 |

## 16. Python AI 服务：具体实现步骤

### 16.1 第一步：增加服务命令入口

在 `pyproject.toml` 中保留现有 `fallguard`，新增：

```toml
[project.scripts]
fallguard = "fall_prediction_desktop.__main__:main"
fallguard-ai = "fall_prediction_service.__main__:main"
```

完成后开发模式应能运行：

```bash
cd /Users/tangjiajun/Desktop/apps/macos
source .venv/bin/activate
python -m fall_prediction_service --port 0 --data-dir /tmp/fallguard-dev
```

`--port 0` 表示由系统分配空闲端口。不要先调用 `find_free_port()` 再启动服务器，因为“查到端口”和“绑定端口”之间存在被其他进程占用的竞态；应让 `ThreadingHTTPServer(("127.0.0.1", 0), ...)` 直接绑定，再从 `server_address` 读取实际端口。

### 16.2 第二步：实现可解析的启动握手

`__main__.py` 的职责只能包括参数解析、创建服务、打印 ready、处理退出。示意：

```python
from __future__ import annotations

import json
import secrets
import signal

from .app import create_service


def main() -> None:
    token = secrets.token_urlsafe(32)
    service = create_service(host="127.0.0.1", port=0, token=token)
    port = service.server_address[1]

    print(json.dumps({
        "event": "ready",
        "port": port,
        "token": token,
        "api_version": "v1",
        "pid": service.pid,
    }), flush=True)

    signal.signal(signal.SIGTERM, service.request_shutdown)
    signal.signal(signal.SIGINT, service.request_shutdown)
    service.serve_forever()


if __name__ == "__main__":
    main()
```

实施要求：

- `ready` 之前的普通日志只能写 stderr，stdout 第一条完整行必须是 ready JSON。
- 模型可以选择启动时加载或首次监控时加载，但 `/health` 必须区分 `starting`、`ready`、`degraded`。
- ready 行必须 `flush=True`，否则 Swift 可能一直等待缓冲区刷新。
- 启动失败时 stderr 写结构化错误，进程使用非零退出码。

### 16.3 第三步：把内部状态转换成稳定合同

不要把 `DashboardSnapshot.__dict__` 直接返回给 Swift。应在 `contracts.py` 定义固定 DTO，在 `serialization.py` 显式映射。例如：

```python
def serialize_status(snapshot: dict[str, object]) -> dict[str, object]:
    risk_percent = int(snapshot.get("riskPercent", 0))
    confidence_percent = int(snapshot.get("confidencePercent", 0))
    return {
        "schema_version": 1,
        "sequence": next_sequence(),
        "timestamp_ms": int(time.time() * 1000),
        "monitoring": bool(snapshot.get("running", False)),
        "loading": bool(snapshot.get("loading", False)),
        "prediction": {
            "state": normalize_model_state(snapshot.get("state")),
            "business_state": map_business_state(snapshot.get("state")),
            "risk_score": clamp01(risk_percent / 100.0),
            "visibility": clamp01(confidence_percent / 100.0),
            "confidence": clamp01(confidence_percent / 100.0),
            "system_status": snapshot.get("systemStatus"),
        },
        "performance": {
            "fps": max(0.0, float(snapshot.get("fps", 0.0))),
        },
        "error": serialize_error(snapshot.get("error")),
    }
```

映射层完成后，即使未来内部字段从 `riskPercent` 改名，Swift 合同也不需要改变。

### 16.4 第四步：加入认证和统一错误

认证逻辑放在所有 `/api/v1/` 路由之前，使用恒定时间比较：

```python
import hmac


def authorized(header: str | None, expected_token: str) -> bool:
    prefix = "Bearer "
    if not header or not header.startswith(prefix):
        return False
    return hmac.compare_digest(header[len(prefix):], expected_token)
```

统一错误格式：

```json
{
  "error": {
    "code": "CAMERA_IN_USE",
    "message_key": "error.camera.in_use",
    "retryable": true,
    "details": null
  }
}
```

至少定义以下错误码：

| 错误码 | HTTP | Swift 行为 |
|---|---:|---|
| `UNAUTHORIZED` | 401 | 视为服务身份异常，不自动反复请求 |
| `SERVICE_NOT_READY` | 503 | 显示加载并有限重试 |
| `CAMERA_PERMISSION_DENIED` | 403 | 显示打开系统设置按钮 |
| `CAMERA_IN_USE` | 409 | 提示关闭 FaceTime/Zoom 后重试 |
| `MONITOR_ALREADY_RUNNING` | 200 | 幂等返回当前状态 |
| `MONITOR_NOT_RUNNING` | 200 | 幂等返回停止状态 |
| `IMPORT_CONFLICT` | 409 | 提示先停止摄像头监控 |
| `INVALID_ARGUMENT` | 400 | 标记对应设置或输入字段 |
| `MODEL_LOAD_FAILED` | 500 | 显示诊断入口，不无限重启 |
| `DATABASE_ERROR` | 500 | 禁止继续写入，保留日志和恢复说明 |

### 16.5 第五步：使启停操作幂等

`POST /monitor/start` 连续调用两次不能创建两个摄像头线程或两个 Session。`POST /monitor/stop` 在未运行时也应成功。响应包含最终状态：

```json
{
  "ok": true,
  "monitoring": true,
  "session_id": "..."
}
```

Python 侧用锁保护以下状态变更：

```text
idle -> starting -> running -> stopping -> idle
                 \-> failed -> idle
```

Swift 不自行猜测状态；按钮点击后先进入本地 `requesting` 状态，最终以服务返回或下一次 `/status` 为准。

### 16.6 第六步：实现历史查询 API

在 repositories 增加分页方法，不允许一次把全部风险样本加载进内存。统一分页参数：

```http
GET /api/v1/events?profile_id=<id>&limit=50&cursor=<opaque>
```

```json
{
  "items": [],
  "next_cursor": null,
  "has_more": false
}
```

`cursor` 对 Swift 保持不透明。推荐由 `created_at + id` 编码，而不是使用易受新增数据影响的 offset。

媒体路径不能直接作为任意文件读取入口。缩略图和录像访问应使用事件或媒体 ID：

```http
GET /api/v1/media/{id}/content
```

服务先从数据库查出路径，再验证文件位于允许的媒体目录内。

### 16.7 第七步：改造媒体导入

Swift 使用 `NSOpenPanel` 选择文件，所以正式原生应用不再调用 Python 的 AppleScript picker，也不需要先把大视频通过 multipart 上传到本机服务。

请求示例：

```json
POST /api/v1/imports
{
  "paths": ["/Users/example/Movies/test.mp4"],
  "output_directory": "/Users/example/Movies/FallGuard",
  "sensitivity": "medium"
}
```

Python 必须验证：路径存在、扩展名允许、文件可读、输出目录可写、当前没有摄像头监控或其他 Import Job。返回 `202 Accepted` 和任务 ID。导入线程每处理一段进度就更新：

```json
{
  "id": "...",
  "state": "running",
  "progress": 0.42,
  "current_frame": 1260,
  "total_frames": 3000,
  "output_video": null,
  "error": null
}
```

若未来启用 App Sandbox，必须改用 security-scoped bookmark；第一阶段应明确不启用 App Sandbox，避免文件授权问题与迁移工作混杂。

## 17. Swift 原生应用：具体实现步骤

### 17.1 创建工程

在 Xcode 创建 macOS App：

- Product Name：`FallGuard`
- Interface：SwiftUI
- Language：Swift
- Bundle Identifier：`com.fallguard.desktop`
- Deployment Target：与项目最终支持版本一致
- Unit Tests：开启
- UI Tests：开启
- App Sandbox：第一阶段关闭

为兼容 macOS 11，状态对象优先使用 `ObservableObject`/`@Published`，不要把工程基础建立在较新系统才支持的 Observation 宏上。

### 17.2 实现 DTO

Swift 字段使用 camelCase，后端 snake_case 通过 decoder 策略转换：

```swift
struct ServiceStatus: Decodable, Equatable {
    let schemaVersion: Int
    let sequence: Int64
    let timestampMs: Int64
    let monitoring: Bool
    let loading: Bool
    let prediction: PredictionDTO
    let performance: PerformanceDTO
    let error: ServiceErrorDTO?
}

struct PredictionDTO: Decodable, Equatable {
    enum ModelState: String, Decodable {
        case normal = "Normal"
        case preFall = "Pre-fall"
        case fall = "Fall"
        case unknown = "Unknown"
    }

    let state: ModelState
    let businessState: String
    let riskScore: Double
    let visibility: Double
    let confidence: Double
    let systemStatus: String?
}
```

未知枚举值不能导致整个 Dashboard 解码失败。正式实现应为枚举增加 `unknown(String)` 自定义解码，或者让 DTO 先保存 String，再由 ViewModel 做兼容映射。

### 17.3 实现 PythonServiceManager

核心状态：

```swift
enum PythonServiceState: Equatable {
    case stopped
    case starting
    case ready(baseURL: URL, token: String)
    case stopping
    case failed(message: String)
}
```

核心启动流程：

```swift
@MainActor
final class PythonServiceManager: ObservableObject {
    @Published private(set) var state: PythonServiceState = .stopped

    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    func start() async {
        guard case .stopped = state else { return }
        state = .starting

        do {
            let executable = try locateServiceExecutable()
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()

            process.executableURL = executable
            process.arguments = ["--port", "0"]
            process.standardOutput = stdout
            process.standardError = stderr
            process.terminationHandler = { [weak self] process in
                Task { @MainActor in
                    self?.handleTermination(status: process.terminationStatus)
                }
            }

            self.process = process
            self.stdoutPipe = stdout
            self.stderrPipe = stderr
            try process.run()

            let ready = try await readReadyLine(from: stdout, timeout: .seconds(20))
            let baseURL = URL(string: "http://127.0.0.1:\(ready.port)/api/v1")!
            try await verifyHealth(baseURL: baseURL, token: ready.token)
            state = .ready(baseURL: baseURL, token: ready.token)
        } catch {
            terminateProcessIfNeeded()
            state = .failed(message: error.localizedDescription)
        }
    }
}
```

正式代码必须额外处理：

- 20 秒启动超时；超时后 terminate，短暂等待，再必要时中止。
- stdout 可能被拆成多段字节，必须按换行累积，不能假设一次读取就是完整 JSON。
- stderr 持续异步排空并写入日志，否则管道写满可能阻塞 Python。
- `terminationHandler` 与用户主动退出之间的竞态。
- 应用重复激活时不得重复创建服务进程。
- Swift 崩溃时 Python 不能永久存活；服务可监控父 PID，父进程消失后自行退出。

### 17.4 实现 API Client

```swift
struct FallGuardAPIClient {
    let baseURL: URL
    let token: String
    let session: URLSession

    func status() async throws -> ServiceStatus {
        try await request(path: "status", method: "GET")
    }

    func startMonitoring() async throws -> MonitorCommandResponse {
        try await request(path: "monitor/start", method: "POST")
    }

    private func request<Response: Decodable>(
        path: String,
        method: String,
        body: Data? = nil
    ) async throws -> Response {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = 10
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        return try decode(Response.self, data: data, response: response)
    }
}
```

要求：

- GET status 超时建议 2 秒，启动/停止 10 秒，导入任务创建 30 秒。
- 只对 GET/health/status 进行有限重试；不要自动重试创建 Profile 等非幂等请求。
- 所有 UI 错误由稳定错误码映射本地化文案。
- 日志记录方法、路径、状态码、耗时和 request ID，但不记录 Token。

### 17.5 实现状态轮询

第一版使用轮询，降低协议复杂度：

- 正在监控且窗口可见：每 250～500 ms 获取一次 status。
- 正在监控但窗口隐藏：每 1 秒一次。
- 空闲：每 2 秒一次。
- 服务 starting：每 500 ms health，最多 20 秒。
- 连续 3 次失败：界面进入连接异常状态，停止高频轮询。

每次只应用 `sequence` 大于当前值的响应，避免慢请求覆盖新状态。轮询任务必须在 ViewModel/Store 中统一管理，View 出现和消失时不得创建多个并发 Timer。

### 17.6 实现预览图像

现有 Python 已提供 `/frame.jpg` 和 `/stream.mjpg`。为了实现简单和稳定，Swift 第一版建议轮询单帧 JPEG，而不是先写完整 MJPEG multipart 解析器：

- 监控中、主窗口可见：每秒 8～12 帧。
- 窗口隐藏：停止下载预览，但 Python 继续推理。
- 每次只允许一个图片请求在途；慢请求未结束时跳过下一帧。
- 在后台线程解码 `NSImage`，主线程只更新最终图像引用。
- Python 输出预览分辨率可限制在 640×480，推理仍可使用原帧。

注意：当前 `_serve_latest_frame()` 中存在连续两次 `self.wfile.write(frame)` 的实现迹象。服务化时应补回归测试并确保响应体只写一次 JPEG，否则 `Content-Length` 与实际字节数不一致。

### 17.7 实现 Dashboard 状态机

Dashboard 至少包含以下 UI 状态，不要只用一个 `isLoading`：

```text
launchingService
serviceFailed
serviceReadyIdle
requestingCamera
monitoringNormal
monitoringPreFall
monitoringFall
personUnknown
stopping
importingMedia
```

按钮规则：

| 状态 | 主按钮 | 可否设置摄像头 | 可否导入媒体 |
|---|---|---:|---:|
| Idle | 开始监控 | 是 | 是 |
| Starting | 正在启动… | 否 | 否 |
| Monitoring | 停止监控 | 否 | 否 |
| Stopping | 正在停止… | 否 | 否 |
| Importing | 取消导入 | 否 | 否 |
| Error | 重试 | 视错误而定 | 视错误而定 |

风险颜色只用于辅助表达，必须同时显示图标和文字，避免色觉障碍用户无法区分。

### 17.8 菜单栏、通知和退出

- 菜单栏显示：打开 FallGuard、当前状态、开始/停止、设置、退出。
- Fall 通知由 Swift 发出，但触发源仍来自 Python 的业务状态变化。
- Swift 记录最后处理过的状态事件 ID，避免每次轮询重复通知。
- 应用退出时先调用 `/shutdown`，最多等待 3 秒，再 terminate 子进程。
- 如果正在监控或导入，退出前明确提示任务将停止。
- 不使用 `--deep` 作为长期签名方案；正式发布应按从内到外顺序签名每个嵌套动态库、Python 服务和主 App。

## 18. 摄像头权限专项验证

这是最早必须验证的技术风险之一。Python/OpenCV 子进程访问摄像头时，macOS 对权限的归属可能受签名、Bundle 结构和启动方式影响，不能只在源码环境验证。

在第 2 周前制作最小签名原型：

1. Swift App 启动已签名的最小 Python/OpenCV helper。
2. helper 只打开摄像头、读取 10 帧后退出。
3. 在一台从未授权 FallGuard 的 Mac 上安装并首次运行。
4. 确认权限提示显示 FallGuard，而不是 Python 或终端。
5. 拒绝权限后再次启动，确认 Swift 能识别并引导打开系统设置。
6. 允许权限后确认重启 App 可正常读取。
7. 验证摄像头被 FaceTime 占用时返回不同错误。

如果权限始终归属不稳定，则调整方案为 Swift/AVFoundation 负责采集，Python 通过 Unix Domain Socket 或共享内存接收帧。该调整会增加约 1～2 周，不应等到全部 UI 完成后才发现。

## 19. 构建与打包：具体做法

### 19.1 建立服务专用 spec

新增 `FallGuardAIService.spec`，入口改为 `fall_prediction_service/__main__.py`。依赖策略从当前“列出大量 hidden imports”改为运行路径白名单加回归验证。

明确排除：

```text
PySide6, shiboken6, webview, rumps,
tkinter, _tkinter, matplotlib.backends,
IPython, jupyter, notebook, pytest,
PyQt5, PyQt6, pandas, pyarrow
```

Polars 和 Matplotlib 必须分别做两次干净构建验证：排除后运行模型加载、摄像头推理和导入视频黄金测试。不能因为源码没有直接 import 就假设安全。

### 19.2 构建目录

建议统一输出：

```text
build-native/
├── ai-service/          # PyInstaller 中间产物
├── payload/AIService/   # 即将复制进 Swift App 的服务目录
├── archive/             # Xcode archive
└── reports/             # size、签名、测试报告
```

不要复用当前 `dist/FallGuard` 作为新服务输出目录，以免旧文件残留造成体积误判。

### 19.3 Xcode Build Phase

构建顺序：

1. 运行 Python 服务测试。
2. 用 PyInstaller 构建 `fallguard-ai`。
3. 对服务执行命令行 health smoke test。
4. 将整个 AIService 目录复制到 App 的 `Contents/Resources/AIService/`。
5. 复制只需要的模型和配置。
6. 签名 AIService 内部动态库和可执行文件。
7. 构建并签名 Swift App。
8. 安装到临时目录执行 UI 冒烟测试。
9. 输出体积明细和签名验证结果。

建议新增：

```text
scripts/build_ai_service.sh
scripts/embed_ai_service.sh
scripts/verify_native_app.sh
scripts/report_bundle_size.sh
```

脚本必须使用明确输入输出目录、`set -euo pipefail`，并支持开发构建与发布构建。不要在正常构建中自动覆盖桌面上的稳定版本。

### 19.4 资源路径

服务源码运行时与打包运行时统一通过一个资源定位函数获取模型：

```python
def service_resource_root() -> Path:
    if root := os.environ.get("FALLGUARD_RESOURCE_ROOT"):
        return Path(root)
    if bundle_root := getattr(sys, "_MEIPASS", None):
        return Path(bundle_root)
    return Path(__file__).resolve().parents[2]
```

Swift 启动服务时可以显式传入只读资源根目录和可写数据目录，避免 helper 猜测主 App 结构：

```text
--resource-root <FallGuard.app/Contents/Resources/AIService>
--data-dir <~/Library/Application Support/FallGuard>
--parent-pid <Swift PID>
```

## 20. 开发执行顺序（按工作日）

以下计划以 1 名全职开发者、已有 macOS 开发环境为基准。

### 第 1～5 个工作日：后端服务闭环

| 日 | 任务 | 当日完成标准 |
|---:|---|---|
| 1 | 保存现有测试结果、体积、视频输出；建立黄金样本 | 可以一条命令重新跑基线 |
| 2 | 创建 `fall_prediction_service` 入口和 health | 无 UI 启动，能返回 ready/health |
| 3 | Token、错误合同、status 映射 | 无 Token 401，状态合同测试通过 |
| 4 | start/stop 幂等接口 | 连续启停 20 次无重复线程 |
| 5 | 服务退出、父 PID 监测、日志 | SIGTERM 和父进程消失均能退出 |

### 第 6～10 个工作日：Swift 技术原型

| 日 | 任务 | 当日完成标准 |
|---:|---|---|
| 6 | 建 Xcode 工程和 DTO/API Client | 可读取 health/status |
| 7 | `PythonServiceManager` | Swift 自动启停 Python 服务 |
| 8 | 最小 Dashboard 和启停按钮 | 原生按钮可控制摄像头 |
| 9 | JPEG 预览与状态轮询 | 预览不卡主线程、无重复轮询 |
| 10 | 打包权限原型 | 在干净权限环境完成摄像头测试 |

第 10 日是“继续/调整”决策点。如果摄像头权限归属不可接受，立即转为 Swift 采集方案，而不是继续铺 UI。

### 第 11～20 个工作日：核心产品迁移

| 日 | 任务 | 完成标准 |
|---:|---|---|
| 11～12 | 完整 Dashboard 状态与错误提示 | 所有监控状态可见、可恢复 |
| 13 | 设置和摄像头选择 | 保存后重启仍生效 |
| 14 | Profiles | 增删、激活与旧数据兼容 |
| 15～16 | 事件与 Session 查询 API、Swift 列表 | 可分页、可查看媒体 |
| 17～18 | Import Job API 和原生文件选择 | 可导入、显示进度、取消 |
| 19 | 菜单栏、通知、中英文 | Fall 只通知一次 |
| 20 | 辅助功能和 UI 回归 | 键盘和 VoiceOver 基本可用 |

### 第 21～30 个工作日：发布工程化

| 日 | 任务 | 完成标准 |
|---:|---|---|
| 21～23 | 服务专用 PyInstaller spec 和依赖精简 | 干净构建可启动，体积有明细 |
| 24～25 | 嵌入 App、签名、资源路径 | 复制到另一用户目录仍可运行 |
| 26 | 黄金视频和数据库兼容回归 | 算法结果和旧数据通过 |
| 27 | 2 小时稳定性与崩溃恢复 | 无泄漏趋势、无残留 helper |
| 28 | 发布文档和诊断日志导出 | 用户可自行提供诊断包 |
| 29 | 发布候选版验收 | 所有硬门槛通过 |
| 30 | 修复缓冲与最终构建 | 生成可交付版本 |

完整功能预计至少 30 个工作日；剩余 1～3 周用于实际发现的问题、视觉打磨、公证和多机器验证。

## 21. 每一步应运行的检查

### Python 代码修改后

```bash
cd /Users/tangjiajun/Desktop/apps/macos
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
python -m fall_prediction_service --port 0 --data-dir /tmp/fallguard-service-test
```

### Swift 代码修改后

在 Xcode 运行 Unit Tests 和 UI Tests；CI/命令行使用项目实际 scheme：

```bash
xcodebuild test \
  -project native/FallGuard/FallGuard.xcodeproj \
  -scheme FallGuard \
  -destination 'platform=macOS'
```

### 发布候选包

```bash
codesign --verify --deep --strict --verbose=2 FallGuard.app
spctl --assess --type execute --verbose=4 FallGuard.app
du -sh FallGuard.app
```

另外必须实际双击 `.app` 测试，不能只运行内部可执行文件。至少在一个新 macOS 用户账户或另一台 Mac 上验证首次启动、摄像头权限和用户数据目录。

## 22. Pull Request / 提交拆分建议

避免一次提交同时包含服务重构、Swift UI 和打包修改。建议按以下顺序拆分：

1. `test: add prediction golden fixtures and migration baseline`
2. `feat(service): add headless health endpoint and ready handshake`
3. `feat(service): add authenticated v1 monitor API`
4. `feat(service): add settings profiles and history contracts`
5. `feat(native): scaffold SwiftUI app and service manager`
6. `feat(native): add dashboard status and monitor controls`
7. `feat(native): add camera preview and error recovery`
8. `feat(native): add settings profiles and history`
9. `feat(native): add media import and progress`
10. `build: package AI service inside native app`
11. `build: prune unused Python runtime dependencies`
12. `docs: switch release entry point to native FallGuard`

每个提交都应可构建；影响算法或合同的提交必须包含对应测试。

## 23. 开工前需要最终确认的产品决定

以下决定不会阻止服务技术原型，但应在 Dashboard 开发前确认：

1. 第一版最低支持的 macOS 版本。若只支持较新版本，可使用更新的 SwiftUI API；若坚持 macOS 11，需要更多兼容代码。
2. 主窗口关闭后是退出应用还是继续在菜单栏监控。
3. Fall 通知是否默认开启，是否允许关闭声音但保留通知。
4. 历史事件是否需要用户标记“真实跌倒/误报”和填写备注。
5. Import Media 是否必须支持文件夹图片序列，还是第一版只支持视频和单张图片。
6. 是否计划通过 Mac App Store 分发。若需要，App Sandbox 和嵌套 Python 运行时会显著影响方案，应单独验证。

在没有额外产品决定时，默认采用：macOS 11、关闭窗口后驻留菜单栏、Fall 通知默认开启、保留事件反馈、继续支持当前所有导入类型、首个版本不走 Mac App Store。
