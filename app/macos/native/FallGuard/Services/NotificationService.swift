import AppKit
import UserNotifications
import OSLog

@MainActor
final class NotificationService: NSObject, UNUserNotificationCenterDelegate {

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "Notification")

    private var notifiedEventKeys: Set<String> = []

    private(set) var isAuthorized: Bool = false

    override init() {
        super.init()
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard Bundle.main.bundleURL.path.hasSuffix(".app") ||
                  Bundle.main.bundleURL.lastPathComponent == "FallGuard" else {
                logger.warning("Not running inside .app bundle — notifications unavailable in dev mode")
                return
            }
            UNUserNotificationCenter.current().delegate = self
            await self.refreshAuthorization()
        }
    }

    func requestPermission() async -> Bool {
        do {
            let granted = try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
            isAuthorized = granted
            logger.info("Notification permission: \(granted)")
            return granted
        } catch {
            logger.error("Notification permission error: \(error.localizedDescription)")
            return false
        }
    }

    func requestPermissionIfNeeded() async -> Bool {
        let settings = await UNUserNotificationCenter.current()
            .notificationSettings()
        switch settings.authorizationStatus {
        case .notDetermined:
            return await requestPermission()
        case .authorized, .provisional:
            isAuthorized = true
            return true
        default:
            isAuthorized = false
            return false
        }
    }

    @discardableResult
    func notifyIfNew(eventId: String, eventType: String,
                     riskPercent: Int) -> Bool {
        guard isAuthorized else {
            logger.warning("Notification skipped because permission is unavailable")
            return false
        }
        let eventKey = "\(eventId):\(eventType)"
        guard !notifiedEventKeys.contains(eventKey) else { return false }
        notifiedEventKeys.insert(eventKey)

        if notifiedEventKeys.count > 200 {
            notifiedEventKeys = Set(notifiedEventKeys.suffix(100))
        }

        let content = UNMutableNotificationContent()
        if eventType == "fall" {
            content.title = NSLocalizedString("notification.fall.title",
                                              comment: "Fall Detected")
            content.body = String(format: NSLocalizedString("notification.fall.body",
                                  comment: ""), riskPercent)
            content.sound = .default
        } else {
            content.title = NSLocalizedString("notification.prefall.title",
                                              comment: "Pre-fall Warning")
            content.body = String(format: NSLocalizedString("notification.prefall.body",
                                  comment: ""), riskPercent)
            content.sound = UNNotificationSound(named: .init("Ping"))
        }
        content.categoryIdentifier = "FALL_EVENT"

        let request = UNNotificationRequest(
            identifier: eventKey,
            content: content,
            trigger: nil  // deliver immediately
        )

        UNUserNotificationCenter.current().add(request) { [weak self] error in
            if let error = error {
                self?.logger.error("Failed to deliver notification: \(error.localizedDescription)")
            }
        }

        logger.info("Notification sent for event \(eventId) (\(eventType))")
        return true
    }

    @discardableResult
    func notifyHelpNeeded(eventId: String, automatic: Bool) -> Bool {
        guard isAuthorized else { return false }
        let content = UNMutableNotificationContent()
        content.title = NSLocalizedString(
            "notification.help_needed.title", comment: ""
        )
        content.body = NSLocalizedString(
            automatic
                ? "notification.help_needed.timeout"
                : "notification.help_needed.requested",
            comment: ""
        )
        content.sound = .default
        content.categoryIdentifier = "FALL_HELP_NEEDED"

        let request = UNNotificationRequest(
            identifier: "\(eventId):help-needed",
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request) { [weak self] error in
            if let error {
                self?.logger.error(
                    "Failed to deliver escalated notification: \(error.localizedDescription)"
                )
            }
        }
        return true
    }

    func clearAll() {
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler:
        @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound, .badge])
    }

    private func refreshAuthorization() async {
        let settings = await UNUserNotificationCenter.current()
            .notificationSettings()
        isAuthorized = settings.authorizationStatus == .authorized
    }

    func loadNotifiedEvents(_ ids: [String]) {
        notifiedEventKeys = Set(ids)
    }

    func saveNotifiedEvents() -> [String] {
        Array(notifiedEventKeys)
    }
}
