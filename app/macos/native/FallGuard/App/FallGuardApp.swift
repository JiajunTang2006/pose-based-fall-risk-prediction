import SwiftUI
import AppKit

// macOS 顶部系统菜单文案：Resources/*/Localizable.strings 中的 Menu 分组。
/// Main entry point for the FallGuard SwiftUI application.
///
/// The app uses a single ``AppStore`` as the source of truth, which is
/// passed through the view hierarchy via ``.environmentObject()``.
@main
struct FallGuardApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store: AppStore
    @StateObject private var themeManager = ThemeManager()
    @StateObject private var languageManager = LanguageManager()

    init() {
        // Check for dev-mode env vars
        let devPortStr = ProcessInfo.processInfo.environment["FALLGUARD_DEV_PORT"]
        let devPort = devPortStr.flatMap(Int.init)
        let devToken = ProcessInfo.processInfo.environment["FALLGUARD_DEV_TOKEN"]
        _store = StateObject(wrappedValue: AppStore(devPort: devPort, devToken: devToken))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .environmentObject(themeManager)
                .environmentObject(languageManager)
                .environment(\.locale, languageManager.locale)
                .frame(minWidth: 1060, minHeight: 700)
                .preferredColorScheme(themeManager.effective)
                .onAppear {
                    appDelegate.attach(store: store)
                    Task { await store.bootstrap() }
                    configureWindow()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified)
        .commands {
            // Replace default About with our own
            CommandGroup(replacing: .appInfo) {
                Button(NSLocalizedString("menu.about", comment: "")) {
                    NSApplication.shared.orderFrontStandardAboutPanel(nil)
                }
            }

            // App-level commands
            CommandGroup(after: .newItem) {
                Divider()
                Button(NSLocalizedString("menu.start_monitoring", comment: "")) {
                    Task { await store.startMonitoring() }
                }
                .keyboardShortcut("m", modifiers: [.command, .shift])
                .disabled(!store.serviceManager.state.isReady)

                Button(NSLocalizedString("menu.stop_monitoring", comment: "")) {
                    Task { await store.stopMonitoring() }
                }
                .keyboardShortcut("m", modifiers: [.command, .option])
                .disabled(!store.isMonitoring)
            }

            // Help
            CommandGroup(replacing: .help) {
                Button(NSLocalizedString("menu.diagnostics", comment: "")) {
                    // Opens diagnostics
                }
                Button(NSLocalizedString("menu.export_logs", comment: "")) {
                    Task { await exportLogs() }
                }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(store)
                .environmentObject(themeManager)
                .environmentObject(languageManager)
                .environment(\.locale, languageManager.locale)
                .frame(width: 780, height: 540)
        }
    }

    private func exportLogs() async {
        // Stub: collect logs from Python stderr + app logs
    }

    /// Make the window chrome transparent so the shared ambient gradient
    /// (``FallGuardBackground``) shows straight through the title-bar strip.
    /// This removes the grey seam at the top and keeps the whole window one
    /// continuous colour, matching the content beneath the toolbar.
    private func configureWindow() {
        DispatchQueue.main.async {
            guard let window = NSApplication.shared.windows.first(where: {
                $0.identifier?.rawValue.contains("FallGuard") == true
                    || $0.className.contains("NSWindow")
            }) ?? NSApplication.shared.windows.first else {
                return
            }
            window.titlebarAppearsTransparent = true
            window.isOpaque = false
            window.backgroundColor = .clear
            window.styleMask.insert(.fullSizeContentView)
        }
    }
}
