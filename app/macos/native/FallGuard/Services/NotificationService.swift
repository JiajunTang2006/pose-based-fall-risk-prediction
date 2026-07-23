import AppKit
import UserNotifications
import OSLog

// 跌倒系统通知文案：Resources/*/Localizable.strings 中的 Notifications 分组。
/// Manages native macOS notifications for fall-detection events.
///
/// The Python service remains the **source of truth** for fall detection.
/// Swift is responsible for presenting the notification and tracking which
/// event IDs have already been alerted (so we never notify twice for the
/// same event).
@MainActor
final class NotificationService: NSObject, UNUserNotificationCenterDelegate {

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "Notification")

    /// Set of event IDs that have already triggered a notification.
    private var notifiedEventIDs: Set<String> = []

    /// Whether the user has granted notification permission.
    private(set) var isAuthorized: Bool = false

    override init() {
        super.init()
        // Defer UNUserNotificationCenter setup — it requires a proper .app bundle.
        // When running as a raw binary during development, this would crash.
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

    // MARK: Public API

    /// Request notification permission from the user.
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

    /// Present a fall-detection notification.
    ///
    /// - Parameters:
    ///   - eventId: The unique event ID from the Python service.
    ///   - eventType: ``"fall"`` or ``"pre-fall"``.
    ///   - riskPercent: Risk percentage (0–100) for the notification body.
    /// - Returns: `true` if a notification was actually shown.
    @discardableResult
    func notifyIfNew(eventId: String, eventType: String,
                     riskPercent: Int) -> Bool {
        guard !notifiedEventIDs.contains(eventId) else { return false }
        notifiedEventIDs.insert(eventId)

        // Trim the set to prevent unbounded growth
        if notifiedEventIDs.count > 200 {
            notifiedEventIDs = Set(notifiedEventIDs.suffix(100))
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
            identifier: eventId,
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

    /// Remove all delivered notifications.
    func clearAll() {
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }

    // MARK: UNUserNotificationCenterDelegate

    /// Show notifications even when the app is in the foreground.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler:
        @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound, .badge])
    }

    // MARK: Private

    private func refreshAuthorization() async {
        let settings = await UNUserNotificationCenter.current()
            .notificationSettings()
        isAuthorized = settings.authorizationStatus == .authorized
    }

    /// Set of already-notified event IDs (for persistence across launches).
    func loadNotifiedEvents(_ ids: [String]) {
        notifiedEventIDs = Set(ids)
    }

    func saveNotifiedEvents() -> [String] {
        Array(notifiedEventIDs)
    }
}
