import SwiftUI
import AppKit
import UniformTypeIdentifiers

@main
struct FallGuardApp: App {

    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store: AppStore
    @StateObject private var themeManager = ThemeManager()
    @StateObject private var languageManager = LanguageManager()

    init() {
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
            CommandGroup(replacing: .appInfo) {
                Button(NSLocalizedString("menu.about", comment: "")) {
                    NSApplication.shared.orderFrontStandardAboutPanel(nil)
                }
            }

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

            CommandGroup(replacing: .help) {
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
                .preferredColorScheme(themeManager.effective)
                .frame(width: 780, height: 540)
        }
    }

    private func exportLogs() async {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "fallguard_diagnostics.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let events = store.recentEvents.map {
            [
                "id": $0.id,
                "type": $0.eventType,
                "status": $0.status,
                "peak_risk": $0.peakRisk,
                "started_at": $0.startedAt,
            ] as [String: Any]
        }
        let payload: [String: Any] = [
            "exported_at": ISO8601DateFormatter().string(from: Date()),
            "service": store.serviceManager.state.displayText,
            "monitoring": store.isMonitoring,
            "fps": store.fps,
            "camera_index": store.currentCameraIndex,
            "recent_events": events,
        ]
        do {
            let data = try JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: url, options: .atomic)
        } catch {
            store.connectionError = error.localizedDescription
        }
    }

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
