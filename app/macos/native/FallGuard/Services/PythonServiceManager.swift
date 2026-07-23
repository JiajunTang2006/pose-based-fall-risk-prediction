import Foundation
import OSLog
import Darwin

// MARK: - Service State

/// Observable process lifecycle state for the Python AI service.
enum PythonServiceState: Equatable {
    case stopped
    case starting
    case ready(baseURL: URL, token: String)
    case stopping
    case failed(message: String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var displayText: String {
        switch self {
        case .stopped: return NSLocalizedString("service.stopped", comment: "")
        case .starting: return NSLocalizedString("service.starting", comment: "")
        case .ready: return NSLocalizedString("service.ready", comment: "")
        case .stopping: return NSLocalizedString("service.stopping", comment: "")
        case .failed(let msg): return msg
        }
    }
}

// MARK: - Service Manager

/// Manages the Python AI service subprocess lifecycle.
///
/// Responsibilities:
/// - Locate and launch the ``fallguard-ai`` executable
/// - Parse the ``ready`` JSON line from stdout
/// - Health-check the service before declaring it ready
/// - Handle crashes, timeouts, and clean shutdown
@MainActor
final class PythonServiceManager: ObservableObject {

    // MARK: Published state

    @Published private(set) var state: PythonServiceState = .stopped

    // MARK: Private

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "ServiceManager")
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var stderrReadHandle: FileHandle?
    private var startTask: Task<Void, Never>?

    /// Maximum seconds to wait for the ``ready`` line from the child process.
    private let startupTimeout: TimeInterval = 20.0

    /// Port for connecting to an externally-managed dev service (debug mode).
    private let devPort: Int?
    private let devToken: String?

    // MARK: Init

    /// - Parameters:
    ///   - devPort: If non-nil, connect to an already-running service
    ///     instead of launching a child process (development mode).
    ///   - devToken: The Bearer token for the dev service.
    init(devPort: Int? = nil, devToken: String? = nil) {
        self.devPort = devPort
        self.devToken = devToken
    }

    deinit {
        // Best-effort cleanup — never block deinit.
        let p = process
        Task.detached { [p] in
            p?.terminate()
        }
    }

    // MARK: Public API

    /// Launch the Python AI service and wait for its ``ready`` handshake.
    func start() async {
        // Guard: already running or in transition
        switch state {
        case .stopped, .failed:
            break
        case .starting, .ready, .stopping:
            logger.info("Service already running or in transition — ignoring start()")
            return
        }

        // Development mode: connect to an externally-launched service
        if let port = devPort, let token = devToken {
            await connectToDevService(port: port, token: token)
            return
        }

        state = .starting
        logger.info("Launching Python AI service...")

        startTask = Task {
            do {
                let executable = try locateServiceExecutable()
                let proc = Process()
                let stdout = Pipe()
                let stderr = Pipe()

                proc.executableURL = executable
                proc.arguments = serviceArguments()
                proc.environment = ProcessInfo.processInfo.environment
                proc.standardOutput = stdout
                proc.standardError = stderr

                // Drain stderr asynchronously so the pipe never fills up.
                let stderrHandle = stderr.fileHandleForReading
                stderrHandle.readabilityHandler = { [weak self] handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    if let text = String(data: data, encoding: .utf8) {
                        self?.logger.debug("[python] \(text, privacy: .public)")
                    }
                }

                try proc.run()
                logger.info("Python process launched (pid=\(proc.processIdentifier))")

                process = proc
                stdoutPipe = stdout
                stderrPipe = stderr
                stderrReadHandle = stderrHandle

                // Wait for the ready line with timeout
                let ready = try await readReadyLine(from: stdout, timeout: startupTimeout)

                guard ready.event == "ready" else {
                    throw ServiceError.invalidReadyMessage
                }

                let baseURL = URL(string: "http://127.0.0.1:\(ready.port)/api/v1")!
                logger.info("Service ready on port \(ready.port)")

                // Verify health before declaring ready
                try await verifyHealth(baseURL: baseURL, token: ready.token)

                // Watch for unexpected termination
                proc.terminationHandler = { [weak self] proc in
                    Task { @MainActor [weak self] in
                        self?.handleTermination(status: proc.terminationStatus, reason: proc.terminationReason)
                    }
                }

                state = .ready(baseURL: baseURL, token: ready.token)
                logger.info("Service state → ready")

            } catch is CancellationError {
                terminateProcessIfNeeded()
                state = .failed(message: NSLocalizedString("error.startup.cancelled", comment: ""))
            } catch {
                logger.error("Service startup failed: \(error.localizedDescription, privacy: .public)")
                terminateProcessIfNeeded()
                state = .failed(message: error.localizedDescription)
            }
        }
        await startTask?.value
        startTask = nil
    }

    /// Gracefully stop the service: POST /shutdown, then terminate.
    func stop() async {
        guard case .ready(let baseURL, let token) = state else {
            startTask?.cancel()
            if devPort == nil {
                await terminateProcessAndWait()
            }
            state = .stopped
            return
        }

        state = .stopping
        logger.info("Shutting down Python service...")

        // Try graceful shutdown first
        let shutdownURL = baseURL.appendingPathComponent("shutdown")
        var req = URLRequest(url: shutdownURL)
        req.httpMethod = "POST"
        req.timeoutInterval = 3
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        do {
            let _ = try await URLSession.shared.data(for: req)
        } catch {
            logger.debug("Graceful shutdown request failed (will force-terminate): \(error.localizedDescription)")
        }

        // Give graceful cleanup time to release AVFoundation/OpenCV, then
        // escalate to SIGTERM/SIGKILL and wait until the child is gone.
        if devPort == nil {
            await terminateProcessAndWait()
        }
        state = .stopped
        logger.info("Service state → stopped")
    }

    // MARK: Private — dev mode

    private func connectToDevService(port: Int, token: String) async {
        guard let baseURL = URL(string: "http://127.0.0.1:\(port)/api/v1") else {
            state = .failed(message: "Invalid dev URL")
            return
        }
        do {
            try await verifyHealth(baseURL: baseURL, token: token)
            state = .ready(baseURL: baseURL, token: token)
            logger.info("Connected to dev service on port \(port)")
        } catch {
            state = .failed(message: "Dev service not reachable: \(error.localizedDescription)")
        }
    }

    // MARK: Private — process lifecycle

    /// Find the ``fallguard-ai`` executable.
    private func locateServiceExecutable() throws -> URL {
        // 1. Check env override
        if let envPath = ProcessInfo.processInfo.environment["FALLGUARD_AI_EXECUTABLE"] {
            let url = URL(fileURLWithPath: envPath)
            _ = envPath  // used above
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw ServiceError.executableNotFound(path: envPath)
            }
            return url
        }

        // 2. Look inside the app bundle Resources/AIService/
        if let bundlePath = Bundle.main.resourcePath {
            let bundleExec = URL(fileURLWithPath: bundlePath)
                .appendingPathComponent("AIService")
                .appendingPathComponent("fallguard-ai")
            if FileManager.default.isExecutableFile(atPath: bundleExec.path) {
                return bundleExec
            }
        }

        // 3. Fallback: look for the Python module in the project tree
        let pythonPath = URL(fileURLWithPath: "/usr/bin/env")
        // Use `python3 -m fall_prediction_service` as fallback
        return pythonPath
    }

    /// Return suitable arguments for the Python service.
    private func serviceArguments(port: Int = 0) -> [String] {
        // Check if we're using the bundled executable vs python -m
        if ProcessInfo.processInfo.environment["FALLGUARD_AI_EXECUTABLE"] != nil {
            return [
                "--host", "127.0.0.1", "--port", "\(port)",
                "--parent-pid", "\(ProcessInfo.processInfo.processIdentifier)",
            ]
        }
        if let bundleResourcePath = Bundle.main.resourcePath {
            let bundled = "\(bundleResourcePath)/AIService/fallguard-ai"
            if FileManager.default.fileExists(atPath: bundled) {
                return [
                    "--host", "127.0.0.1", "--port", "\(port)",
                    "--parent-pid", "\(ProcessInfo.processInfo.processIdentifier)",
                ]
            }
        }
        // Source-mode fallback: python3 -m fall_prediction_service
        return [
            "-m", "fall_prediction_service",
            "--host", "127.0.0.1",
            "--port", "\(port)",
            "--parent-pid", "\(ProcessInfo.processInfo.processIdentifier)",
        ]
    }

    /// Read the first complete JSON line from the subprocess's stdout.
    private func readReadyLine(from stdout: Pipe, timeout: TimeInterval) async throws -> ReadyMessage {
        let handle = stdout.fileHandleForReading
        let deadline = Date().addingTimeInterval(timeout)

        var buffer = Data()

        while Date() < deadline {
            // Check cancellation
            try Task.checkCancellation()

            // Poll for available data
            let available = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
                DispatchQueue.global().async {
                    let data = handle.availableData
                    cont.resume(returning: data)
                }
            }

            guard !available.isEmpty else {
                // EOF before ready line
                throw ServiceError.processExitedEarly
            }

            buffer.append(available)

            // Try to extract a complete line
            while let newlineRange = buffer.range(of: Data([0x0A])) {  // '\n'
                let lineData = buffer.subdata(in: 0..<newlineRange.lowerBound)
                buffer.removeSubrange(0...newlineRange.lowerBound)

                guard let line = String(data: lineData, encoding: .utf8)?.trimmingCharacters(in: .whitespaces),
                      !line.isEmpty else {
                    continue
                }

                // Try to parse as ReadyMessage
                if let ready = try? JSONDecoder().decode(ReadyMessage.self, from: Data(line.utf8)) {
                    return ready
                }
                // Otherwise it's a log line → ignore (stderr vs stdout separation handles this)
            }

            // Brief sleep to avoid busy-waiting
            try await Task.sleep(nanoseconds: 50_000_000) // 50 ms
        }

        throw ServiceError.startupTimeout
    }

    /// Call GET /health once to verify the service is reachable.
    /// Accepts all health states (``starting``, ``ready``, ``degraded``) —
    /// the UI will poll for status updates and show progress.
    private func verifyHealth(baseURL: URL, token: String) async throws {
        let healthURL = baseURL.appendingPathComponent("health")
        var req = URLRequest(url: healthURL)
        req.timeoutInterval = 5

        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            try Task.checkCancellation()

            let (data, response) = try await URLSession.shared.data(for: req)
            guard let httpResp = response as? HTTPURLResponse else {
                throw ServiceError.healthCheckFailed(reason: "Invalid response")
            }

            if httpResp.statusCode == 200 {
                let health = try JSONDecoder().decode(ServiceHealth.self, from: data)
                switch health.status {
                case .starting, .ready, .degraded:
                    return
                }
            }

            throw ServiceError.healthCheckFailed(
                reason: "HTTP \(httpResp.statusCode)"
            )
        }
        throw ServiceError.healthCheckFailed(reason: "Service did not become ready")
    }

    /// Called when the subprocess exits unexpectedly.
    private func handleTermination(status: Int32, reason: Process.TerminationReason) {
        logger.warning("Python process exited (status=\(status), reason=\(reason.rawValue))")

        // Only react if we're not already stopping/stopped
        switch state {
        case .starting:
            state = .failed(message: NSLocalizedString(
                "error.service.crashed",
                comment: "Service crashed during startup"
            ))
        case .ready:
            state = .failed(message: NSLocalizedString(
                "error.service.unexpected_exit",
                comment: "Service stopped unexpectedly"
            ))
        case .stopping, .stopped, .failed:
            break  // expected
        }
    }

    /// Force-kill the subprocess if it's still running.
    private func terminateProcessIfNeeded() {
        guard let proc = process, proc.isRunning else { return }
        stderrReadHandle?.readabilityHandler = nil
        proc.terminate()
        // Give it a moment, then kill
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            if proc.isRunning {
                proc.interrupt()  // SIGINT
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                if proc.isRunning {
                    Darwin.kill(proc.processIdentifier, SIGKILL)
                }
            }
        }
    }

    private func terminateProcessAndWait() async {
        guard let proc = process else {
            clearProcessReferences()
            return
        }

        stderrReadHandle?.readabilityHandler = nil

        // The shutdown endpoint normally exits the service by itself.
        await waitForExit(proc, timeout: 3.0)
        if proc.isRunning {
            proc.terminate()
            await waitForExit(proc, timeout: 1.5)
        }
        if proc.isRunning {
            Darwin.kill(proc.processIdentifier, SIGKILL)
            await waitForExit(proc, timeout: 1.0)
        }

        clearProcessReferences()
    }

    private func waitForExit(_ proc: Process, timeout: TimeInterval) async {
        let deadline = Date().addingTimeInterval(timeout)
        while proc.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    private func clearProcessReferences() {
        stderrReadHandle?.readabilityHandler = nil
        stderrReadHandle = nil
        stdoutPipe = nil
        stderrPipe = nil
        process = nil
    }
}

// MARK: - Service Errors

enum ServiceError: LocalizedError {
    case executableNotFound(path: String)
    case startupTimeout
    case processExitedEarly
    case invalidReadyMessage
    case healthCheckFailed(reason: String)

    var errorDescription: String? {
        switch self {
        case .executableNotFound(let path):
            return "AI service executable not found: \(path)"
        case .startupTimeout:
            return NSLocalizedString("error.service.timeout", comment: "")
        case .processExitedEarly:
            return NSLocalizedString("error.service.early_exit", comment: "")
        case .invalidReadyMessage:
            return NSLocalizedString("error.service.invalid_ready", comment: "")
        case .healthCheckFailed(let reason):
            return String(format: NSLocalizedString("error.service.health_failed", comment: ""), reason)
        }
    }
}
