# FallGuardPlus 前端文字修改指南

## 最常用的两个文件

如果你只想修改界面文字，通常只需要编辑下面两个文件，不需要修改 Swift 页面布局：

- `Resources/zh.lproj/Localizable.strings`：中文版全部主要界面文字。
- `Resources/en.lproj/Localizable.strings`：英文版全部主要界面文字。

每一行的格式是：

```text
"程序使用的固定 key" = "界面上显示的文字";
```

只修改等号右侧引号里的内容。不要修改左侧 key，不要删除末尾分号。包含 `%d`、`%@`、`%%` 的行必须保留这些符号，它们会在运行时替换为数字、名称或百分号。

## 页面与代码文件对照

| 前端部分 | Swift 页面代码 | 文字资源中的分组 |
| --- | --- | --- |
| 主窗口、左侧导航栏、顶部 Start/Stop、品牌区 | `Features/ContentView.swift` | `Tab Bar`、`Brand`、`Buttons`、`Service` |
| Dashboard 首页和实时检测界面 | `Features/Dashboard/DashboardView.swift` | `Dashboard`、`Dashboard Cards`、`Risk Levels`、`Status`、`Metrics`、`States` |
| Events 事件记录页面 | `Features/Events/EventsView.swift` | `Events` |
| Import Media 导入页面 | `Features/ImportMedia/ImportMediaView.swift` | `Import` |
| Profiles 用户档案页面 | `Features/Profiles/ProfilesView.swift` | `Profiles` |
| Settings 设置窗口 | `Features/Settings/SettingsView.swift` | `Settings`、`Theme`、`Sensitivity` |
| macOS 顶部系统菜单 | `App/FallGuardApp.swift` | `Menu` |
| macOS 菜单栏小图标菜单 | `App/MenuBarController.swift` | `Menu Bar` |
| 系统通知 | `Services/NotificationService.swift` | `Notifications` |
| 错误提示和确认弹窗 | 多个页面及服务文件 | `Alerts`、`Errors`、`General` |

## 其他前端文件（通常不需要修改）

- `App/DesignSystem.swift`：颜色、字体、圆角、间距和背景。
- `App/GlassEffect.swift`：毛玻璃效果。
- `App/ThemeManager.swift`：浅色、深色主题切换。
- `App/FallGuardApp.swift`：应用窗口和 macOS 菜单入口。
- `App/AppStore.swift`：前端状态以及按钮触发的功能。
- `Models/ServiceModels.swift`：前端与 Python 服务之间的数据格式。
- `Services/`：Swift 前端和 Python AI 服务之间的通信、摄像头预览及通知。

## 少量不在文字资源文件中的内容

以下文字目前直接写在 Swift 或应用配置里，修改时需要格外谨慎：

- 左侧品牌名称 `FallGuard Plus`：`Features/ContentView.swift`。
- 设置“关于”页面的 `FallGuard` 和版本号：`Features/Settings/SettingsView.swift`。
- 应用在 Finder、Dock 和菜单栏显示的名称：`Resources/Info.plist` 中的 `CFBundleDisplayName`、`CFBundleName`。
- 实时监控中的技术缩写 `REC`、`FPS` 和时间轴 `Now`：`Features/Dashboard/DashboardView.swift`。

如果只是修改普通按钮、标题、提示语和说明文字，优先使用两个 `Localizable.strings` 文件。
