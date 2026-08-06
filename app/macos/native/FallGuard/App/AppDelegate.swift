import AppKit
import SwiftUI
import OSLog

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "AppDelegate")
    private var menuBarController: MenuBarController?
    private var terminationInProgress = false

    weak var store: AppStore?

    func attach(store: AppStore) {
        guard self.store !== store else { return }
        self.store = store
        menuBarController?.teardown()
        menuBarController = MenuBarController(store: store)
        menuBarController?.setup()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        logger.info("FallGuard did finish launching")
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(languageDidChange),
            name: .fallGuardLanguageDidChange,
            object: nil
        )

        if let store {
            menuBarController = MenuBarController(store: store)
            menuBarController?.setup()
        }

        NSApp.setActivationPolicy(.regular)
        refreshLocalizedWindowChrome()
    }

    func applicationWillTerminate(_ notification: Notification) {
        logger.info("FallGuard will terminate")
        NotificationCenter.default.removeObserver(self)
        menuBarController?.teardown()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminationInProgress, let store else {
            return .terminateNow
        }

        terminationInProgress = true
        logger.info("Stopping monitoring and AI service before termination")
        Task { @MainActor in
            await store.shutdown()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return false
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in NSApp.windows where window.canBecomeMain {
                window.makeKeyAndOrderFront(nil)
                return true
            }
        }
        return true
    }

    @objc private func languageDidChange() {
        menuBarController?.teardown()
        if let store {
            menuBarController = MenuBarController(store: store)
            menuBarController?.setup()
        }
        refreshLocalizedWindowChrome()
    }

    private func refreshLocalizedWindowChrome() {
        let appName = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
            ?? Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String
            ?? "FallGuard"
        for window in NSApp.windows where window.identifier?.rawValue == "com_apple_SwiftUI_Settings_window" {
            window.title = "\(appName) \(NSLocalizedString("settings.title", comment: ""))"
        }

        refreshLocalizedMainMenu()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            self?.refreshLocalizedMainMenu()
        }
    }

    private func refreshLocalizedMainMenu() {
        let menuKeys = ["menu.file", "menu.edit", "menu.view", "menu.window", "menu.help"]
        let menuItems = NSApp.mainMenu?.items ?? []
        for (offset, key) in menuKeys.enumerated() where menuItems.indices.contains(offset + 1) {
            let item = menuItems[offset + 1]
            let localizedTitle = NSLocalizedString(key, comment: "")
            item.title = localizedTitle
            item.submenu?.title = localizedTitle
        }
    }

    func setupMainMenu() {
        let mainMenu = NSMenu()

        let appMenu = NSMenu()
        let aboutItem = NSMenuItem(
            title: String(format: NSLocalizedString("menu.about_format", comment: ""), "FallGuard"),
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
            keyEquivalent: ""
        )
        appMenu.addItem(aboutItem)
        appMenu.addItem(.separator())

        let prefsItem = NSMenuItem(
            title: NSLocalizedString("menu.preferences", comment: ""),
            action: nil,
            keyEquivalent: ","
        )
        appMenu.addItem(prefsItem)
        appMenu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: NSLocalizedString("menu.quit", comment: ""),
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        appMenu.addItem(quitItem)

        let appMenuItem = NSMenuItem()
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let fileMenu = NSMenu(title: NSLocalizedString("menu.file", comment: ""))

        let startItem = NSMenuItem(
            title: NSLocalizedString("menu.start_monitoring", comment: ""),
            action: #selector(AppDelegate.startMonitoringAction),
            keyEquivalent: "M"
        )
        startItem.keyEquivalentModifierMask = [.command, .shift]
        fileMenu.addItem(startItem)

        let stopItem = NSMenuItem(
            title: NSLocalizedString("menu.stop_monitoring", comment: ""),
            action: #selector(AppDelegate.stopMonitoringAction),
            keyEquivalent: "M"
        )
        stopItem.keyEquivalentModifierMask = [.command, .option]
        fileMenu.addItem(stopItem)

        fileMenu.addItem(.separator())

        let importItem = NSMenuItem(
            title: NSLocalizedString("menu.import_media", comment: ""),
            action: #selector(AppDelegate.importMediaAction),
            keyEquivalent: "i"
        )
        importItem.keyEquivalentModifierMask = [.command, .shift]
        fileMenu.addItem(importItem)

        let fileMenuItem = NSMenuItem()
        fileMenuItem.submenu = fileMenu
        mainMenu.addItem(fileMenuItem)

        let viewMenu = NSMenu(title: NSLocalizedString("menu.view", comment: ""))
        let toggleSidebarItem = NSMenuItem(
            title: NSLocalizedString("menu.toggle_sidebar", comment: ""),
            action: #selector(NSSplitViewController.toggleSidebar(_:)),
            keyEquivalent: "s"
        )
        toggleSidebarItem.keyEquivalentModifierMask = [.command, .option]
        viewMenu.addItem(toggleSidebarItem)

        let viewMenuItem = NSMenuItem()
        viewMenuItem.submenu = viewMenu
        mainMenu.addItem(viewMenuItem)

        let windowMenu = NSMenu(title: NSLocalizedString("menu.window", comment: ""))
        windowMenu.addItem(NSMenuItem(
            title: NSLocalizedString("menu.show_main_window", comment: ""),
            action: #selector(AppDelegate.showMainWindow),
            keyEquivalent: "0"
        ))

        let windowMenuItem = NSMenuItem()
        windowMenuItem.submenu = windowMenu
        mainMenu.addItem(windowMenuItem)

        NSApp.mainMenu = mainMenu
    }

    @objc private func startMonitoringAction() {
        Task { @MainActor [weak store] in
            await store?.startMonitoring()
        }
    }

    @objc private func stopMonitoringAction() {
        Task { @MainActor [weak store] in
            await store?.stopMonitoring()
        }
    }

    @objc private func importMediaAction() {
        NotificationCenter.default.post(name: .fallGuardNavigateToImport, object: nil)
    }

    @objc private func showMainWindow() {
        for window in NSApp.windows where window.canBecomeMain {
            window.makeKeyAndOrderFront(nil)
        }
    }
}
