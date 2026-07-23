import Foundation
import OSLog

/// Polls `GET /api/v1/status` on a configurable interval and publishes results.
///
/// The poll rate adapts to application state:
/// - Monitoring + window visible: 250–500 ms
/// - Monitoring + window hidden: 1 s
/// - Idle: 2 s
/// - Service starting: 500 ms health check, max 20 s
/// - 3 consecutive failures: enter error state, reduce to 5 s
@MainActor
final class StatusPoller: ObservableObject {

    // MARK: Published

    @Published private(set) var latestStatus: ServiceStatus?
    @Published private(set) var lastError: APIError?
    @Published private(set) var consecutiveFailures: Int = 0
    @Published var isWindowVisible: Bool = true

    /// Set to `true` when monitoring is active (drives poll frequency).
    var isMonitoring: Bool = false

    // MARK: Private

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "StatusPoller")
    private let client: FallGuardAPIClient
    private var task: Task<Void, Never>?
    private var lastSequence: Int64 = -1

    private let maxConsecutiveFailures = 3
    private let fastInterval: TimeInterval = 0.35    // 350 ms
    private let hiddenInterval: TimeInterval = 1.0
    private let idleInterval: TimeInterval = 2.0
    private let errorInterval: TimeInterval = 5.0

    init(client: FallGuardAPIClient) {
        self.client = client
    }

    // MARK: Public

    func start() {
        guard task == nil else { return }
        logger.info("Status poller started")
        task = Task { await pollLoop() }
    }

    func stop() {
        task?.cancel()
        task = nil
        logger.info("Status poller stopped")
    }

    /// Reset the failure counter (call when service reconnects).
    func resetFailures() {
        consecutiveFailures = 0
        lastError = nil
    }

    // MARK: Private

    private var currentInterval: TimeInterval {
        if consecutiveFailures >= maxConsecutiveFailures {
            return errorInterval
        }
        if isMonitoring {
            return isWindowVisible ? fastInterval : hiddenInterval
        }
        return idleInterval
    }

    private func pollLoop() async {
        while !Task.isCancelled {
            do {
                let status = try await client.status()

                // Ignore stale responses (sequence must be strictly increasing)
                if status.sequence > lastSequence {
                    lastSequence = status.sequence
                    latestStatus = status
                }

                consecutiveFailures = 0
                lastError = nil
                isMonitoring = status.monitoring
            } catch let error as APIError {
                consecutiveFailures += 1
                lastError = error
                logger.warning("Status poll failed (#\(self.consecutiveFailures)): \(error.localizedDescription, privacy: .public)")
            } catch {
                consecutiveFailures += 1
                lastError = .transportError(error)
                logger.warning("Status poll failed: \(error.localizedDescription, privacy: .public)")
            }

            try? await Task.sleep(
                nanoseconds: UInt64(currentInterval * 1_000_000_000)
            )
        }
    }
}
