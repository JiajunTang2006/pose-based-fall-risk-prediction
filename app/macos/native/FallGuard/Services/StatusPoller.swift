import Foundation
import OSLog

@MainActor
final class StatusPoller: ObservableObject {

    @Published private(set) var latestStatus: ServiceStatus?
    @Published private(set) var lastError: APIError?
    @Published private(set) var consecutiveFailures: Int = 0
    @Published var isWindowVisible: Bool = true

    var isMonitoring: Bool = false

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

    func resetFailures() {
        consecutiveFailures = 0
        lastError = nil
    }

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
