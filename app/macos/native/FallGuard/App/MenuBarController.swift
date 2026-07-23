import AppKit
import OSLog

// 菜单栏 FallGuard 小图标的文案：Resources/*/Localizable.strings 中的 Menu Bar 分组。
/// Manages the macOS menu bar icon and its context menu.
///
/// The menu bar shows the current monitoring state and provides
/// quick access to Start, Stop, Settings, and Quit.
@MainActor
final class MenuBarController: NSObject {

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "MenuBar")
    private weak var store: AppStore?
    private var statusItem: NSStatusItem?
    private var timer: Timer?

    init(store: AppStore?) {
        self.store = store
        super.init()
    }

    func setup() {
        statusItem = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.variableLength
        )

        guard let item = statusItem else { return }

        if let button = item.button {
            button.image = NSImage(
                systemSymbolName: "shield.fill",
                accessibilityDescription: "FallGuard"
            )
            button.title = ""
        }

        // Build the menu
        let menu = NSMenu()

        let statusItem = NSMenuItem(
            title: NSLocalizedString("menubar.status", comment: "") + ": --",
            action: nil,
            keyEquivalent: ""
        )
        statusItem.isEnabled = false
        statusItem.tag = 100  // tag for runtime updates
        menu.addItem(statusItem)

        menu.addItem(.separator())

        let showItem = NSMenuItem(
            title: NSLocalizedString("menubar.show", comment: ""),
            action: #selector(showMainWindow),
            keyEquivalent: ""
        )
        showItem.target = self
        menu.addItem(showItem)

        let startItem = NSMenuItem(
            title: NSLocalizedString("menubar.start", comment: ""),
            action: #selector(startMonitoring),
            keyEquivalent: ""
        )
        startItem.target = self
        startItem.tag = 101
        menu.addItem(startItem)

        let stopItem = NSMenuItem(
            title: NSLocalizedString("menubar.stop", comment: ""),
            action: #selector(stopMonitoring),
            keyEquivalent: ""
        )
        stopItem.target = self
        stopItem.tag = 102
        menu.addItem(stopItem)

        menu.addItem(.separator())

        let settingsItem = NSMenuItem(
            title: NSLocalizedString("menubar.settings", comment: ""),
            action: #selector(openSettings),
            keyEquivalent: ""
        )
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: NSLocalizedString("menubar.quit", comment: ""),
            action: #selector(quitApp),
            keyEquivalent: ""
        )
        quitItem.target = self
        menu.addItem(quitItem)

        item.menu = menu

        // Periodic status text update
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.updateStatusText()
            }
        }
    }

    func teardown() {
        timer?.invalidate()
        timer = nil
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
            statusItem = nil
        }
    }

    private func updateStatusText() {
        guard let menu = statusItem?.menu,
              let store = store else { return }

        let statusItem = menu.item(withTag: 100)
        if store.isMonitoring {
            let risk = store.riskPercent
            statusItem?.title = String(format: NSLocalizedString(
                "menubar.status.monitoring", comment: ""
            ), risk)
        } else {
            statusItem?.title = NSLocalizedString("menubar.status.idle", comment: "")
        }
    }

    // MARK: Actions

    @objc private func showMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        for window in NSApp.windows where window.canBecomeMain {
            window.makeKeyAndOrderFront(nil)
        }
    }

    @objc private func startMonitoring() {
        Task { @MainActor [weak store] in
            await store?.startMonitoring()
        }
    }

    @objc private func stopMonitoring() {
        Task { @MainActor [weak store] in
            await store?.stopMonitoring()
        }
    }

    @objc private func openSettings() {
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }

    @objc private func quitApp() {
        // Confirm if monitoring
        if store?.isMonitoring == true {
            let alert = NSAlert()
            alert.messageText = NSLocalizedString("alert.quit.monitoring.title", comment: "")
            alert.informativeText = NSLocalizedString("alert.quit.monitoring.message", comment: "")
            alert.addButton(withTitle: NSLocalizedString("alert.quit.button", comment: ""))
            alert.addButton(withTitle: NSLocalizedString("alert.cancel", comment: ""))
            alert.alertStyle = .warning

            if alert.runModal() == .alertFirstButtonReturn {
                NSApp.terminate(nil)
            }
        } else {
            NSApp.terminate(nil)
        }
    }
}
