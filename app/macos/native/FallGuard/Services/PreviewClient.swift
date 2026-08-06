import AppKit
import Foundation
import OSLog

@MainActor
final class PreviewClient: ObservableObject {

    @Published private(set) var currentImage: NSImage?
    @Published var isWindowVisible: Bool = true
    @Published var isMonitoring: Bool = false

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "PreviewClient")
    private let client: FallGuardAPIClient
    private var task: Task<Void, Never>?
    private var isFetching = false

    private let frameInterval: TimeInterval = 0.10

    init(client: FallGuardAPIClient) {
        self.client = client
    }

    func start() {
        guard task == nil else { return }
        task = Task { await fetchLoop() }
        logger.info("Preview client started")
    }

    func stop() {
        task?.cancel()
        task = nil
        currentImage = nil
        logger.info("Preview client stopped")
    }

    private func fetchLoop() async {
        while !Task.isCancelled {
            guard isMonitoring && isWindowVisible else {
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 s
                continue
            }

            guard !isFetching else {
                try? await Task.sleep(nanoseconds: 20_000_000) // 20 ms
                continue
            }

            isFetching = true
            let start = CFAbsoluteTimeGetCurrent()

            do {
                let data = try await client.latestFrame()

                if let img = NSImage(data: data) {
                    currentImage = img
                }
            } catch {
                logger.debug("Frame fetch failed: \(error.localizedDescription)")
            }

            isFetching = false

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let sleepTime = frameInterval - elapsed
            if sleepTime > 0 {
                try? await Task.sleep(
                    nanoseconds: UInt64(sleepTime * 1_000_000_000)
                )
            }
        }
    }
}
