import AppKit
import Combine
import Foundation
import OSLog

// MARK: - Dashboard UI State

/// All possible UI states for the Dashboard (per plan §17.7).
enum DashboardState: Equatable {
    case launchingService
    case serviceFailed(String)
    case serviceReadyIdle
    case requestingCamera
    case monitoringNormal
    case monitoringPreFall
    case monitoringFall
    case personUnknown
    case stopping
    case importingMedia
}

// MARK: - App Store

/// Single source of truth for the app's UI state.
///
/// Owns the service layer and coordinates between the Python service,
/// API client, status poller, and preview client.  All published properties
/// are updated on the main actor.
///
/// Uses ``ObservableObject`` / ``@Published`` for macOS 11 compatibility
/// (rather than the iOS 17+ ``@Observable`` macro).
@MainActor
final class AppStore: ObservableObject {

    // MARK: Published state

    @Published var dashboardState: DashboardState = .launchingService
    @Published var connectionError: String?

    // Status data
    @Published var riskPercent: Int = 0
    @Published var riskScore: Double = 0
    @Published var confidencePercent: Int = 0
    @Published var visibility: Int = 0
    @Published var fps: Double = 0
    @Published var modelState: PredictionDTO.ModelState = .normal
    @Published var businessState: PredictionDTO.BusinessState = .safe
    @Published var isMonitoring: Bool = false
    @Published var isLoading: Bool = true
    @Published var personVisible: Bool = true

    // Settings
    @Published var settings: ServiceSettings?
    @Published var cameras: [Int] = [0]
    @Published var currentCameraIndex: Int = 0

    // Profiles
    @Published var profiles: [ProfileDTO] = []
    @Published var activeProfileId: String?
    @Published var activeProfile: ProfileDTO?

    // Events
    @Published var recentEvents: [EventDTO] = []
    @Published var sessions: [SessionDTO] = []

    // Import
    @Published var importJob: ImportJobDTO?
    @Published var isImporting: Bool = false

    // Preview
    @Published var previewImage: NSImage?

    // Risk history for trend chart (last 48 values)
    @Published var riskHistory: [Int] = []
    @Published var monitoringStartTime: Date?
    @Published var totalAlerts: Int = 0
    @Published var highRiskEvents: Int = 0
    private let maxRiskHistory = 48

    // MARK: Service layer

    let serviceManager: PythonServiceManager
    private(set) var apiClient: FallGuardAPIClient?
    private(set) var statusPoller: StatusPoller?
    private(set) var previewClient: PreviewClient?
    let notificationService = NotificationService()

    // MARK: Internal

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "AppStore")
    private var cancellables = Set<AnyCancellable>()
    private var lastNotifiedEventId: String?

    // MARK: Init

    /// - Parameters:
    ///   - devPort: Non-nil to connect to a dev Python service.
    ///   - devToken: Token for the dev service.
    init(devPort: Int? = nil, devToken: String? = nil) {
        serviceManager = PythonServiceManager(devPort: devPort, devToken: devToken)
    }

    // MARK: Lifecycle

    /// Launch the Python service and wire up all sub-systems.
    func bootstrap() async {
        dashboardState = .launchingService
        await serviceManager.start()
        await handleServiceStateChange()
    }

    /// Called when the app is about to terminate.
    func shutdown() async {
        statusPoller?.stop()
        previewClient?.stop()
        previewImage = nil

        if isMonitoring, let client = apiClient {
            _ = try? await client.stopMonitoring()
            isMonitoring = false
        }
        await serviceManager.stop()
    }

    // MARK: Monitor actions

    func startMonitoring() async {
        guard let client = apiClient else { return }
        dashboardState = .requestingCamera
        do {
            let resp = try await client.startMonitoring()
            if resp.ok {
                // State will update via status poll
                monitoringStartTime = Date()
                riskHistory = []
                totalAlerts = 0
                highRiskEvents = 0
                logger.info("Monitoring started")
            } else if let err = resp.error {
                handleAPIError(err)
            }
        } catch let error as APIError {
            handleAPIError(error)
        } catch {
            connectionError = error.localizedDescription
        }
    }

    func stopMonitoring() async {
        guard let client = apiClient else { return }
        dashboardState = .stopping
        do {
            let resp = try await client.stopMonitoring()
            if resp.ok {
                logger.info("Monitoring stopped")
            }
        } catch let error as APIError {
            handleAPIError(error)
        } catch {
            connectionError = error.localizedDescription
        }
    }

    // MARK: Settings

    func updateSettings(_ changes: [String: Any]) async {
        guard let client = apiClient else { return }
        do {
            settings = try await client.updateSettings(changes)
        } catch {
            connectionError = error.localizedDescription
        }
    }

    func refreshSettings() async {
        guard let client = apiClient else { return }
        do {
            settings = try await client.getSettings()
        } catch {
            logger.warning("Failed to load settings: \(error.localizedDescription)")
        }
    }

    func refreshCameras() async {
        guard let client = apiClient else { return }
        do {
            let list = try await client.getCameras()
            cameras = list.cameras
            currentCameraIndex = list.current
        } catch {
            logger.warning("Failed to load cameras: \(error.localizedDescription)")
        }
    }

    // MARK: Profiles

    func loadProfiles() async {
        guard let client = apiClient else { return }
        do {
            let resp = try await client.getProfiles()
            profiles = resp.profiles
            activeProfileId = resp.activeId
            activeProfile = resp.activeProfile
        } catch {
            logger.warning("Failed to load profiles: \(error.localizedDescription)")
        }
    }

    func createProfile(name: String) async {
        guard let client = apiClient else { return }
        do {
            let _ = try await client.createProfile(name: name)
            await loadProfiles()
        } catch {
            connectionError = error.localizedDescription
        }
    }

    func activateProfile(id: String) async {
        guard let client = apiClient else { return }
        do {
            let _ = try await client.activateProfile(id: id)
            await loadProfiles()
        } catch {
            connectionError = error.localizedDescription
        }
    }

    func deleteProfile(id: String) async {
        guard let client = apiClient else { return }
        do {
            let _ = try await client.deleteProfile(id: id)
            await loadProfiles()
        } catch {
            connectionError = error.localizedDescription
        }
    }

    // MARK: Events & Sessions

    func loadRecentEvents() async {
        guard let client = apiClient else { return }
        do {
            let page = try await client.getEvents(limit: 20)
            recentEvents = page.items
        } catch {
            logger.warning("Failed to load events: \(error.localizedDescription)")
        }
    }

    func loadSessions() async {
        guard let client = apiClient else { return }
        do {
            let page = try await client.getSessions(limit: 20)
            sessions = page.items
        } catch {
            logger.warning("Failed to load sessions: \(error.localizedDescription)")
        }
    }

    // MARK: Import

    func startImport(paths: [String], outputDirectory: String? = nil) async {
        guard let client = apiClient else { return }
        guard case .serviceReadyIdle = dashboardState else {
            connectionError = NSLocalizedString("error.import.stop_monitoring",
                                                 comment: "")
            return
        }

        dashboardState = .importingMedia
        isImporting = true
        do {
            let sens = settings?.sensitivity ?? "medium"
            let resp = try await client.createImport(
                paths: paths,
                outputDirectory: outputDirectory,
                sensitivity: sens
            )
            if resp.ok {
                importJob = resp.import
            }
        } catch let error as APIError {
            handleAPIError(error)
            isImporting = false
            dashboardState = .serviceReadyIdle
        } catch {
            connectionError = error.localizedDescription
            isImporting = false
            dashboardState = .serviceReadyIdle
        }
    }

    func refreshImportStatus() async {
        guard let client = apiClient, let job = importJob else { return }
        do {
            importJob = try await client.getImport(id: job.id)
            if importJob?.state == .complete || importJob?.state == .error {
                isImporting = false
                dashboardState = .serviceReadyIdle
            }
        } catch {
            logger.warning("Failed to load import status: \(error.localizedDescription)")
        }
    }

    // MARK: Private — wiring

    private func handleServiceStateChange() async {
        switch serviceManager.state {
        case .ready(let baseURL, let token):
            let client = FallGuardAPIClient(baseURL: baseURL, token: token)
            apiClient = client

            // Wire sub-services
            let poller = StatusPoller(client: client)
            statusPoller = poller

            let preview = PreviewClient(client: client)
            previewClient = preview

            // Observe poller status updates
            poller.$latestStatus
                .receive(on: DispatchQueue.main)
                .sink { [weak self] status in
                    Task { @MainActor [weak self] in
                        self?.applyStatus(status)
                    }
                }
                .store(in: &cancellables)

            // Observe poller errors
            poller.$consecutiveFailures
                .receive(on: DispatchQueue.main)
                .sink { [weak self] count in
                    if count >= 3 {
                        self?.connectionError = NSLocalizedString(
                            "error.connection.lost", comment: ""
                        )
                    } else if count == 0 {
                        self?.connectionError = nil
                    }
                }
                .store(in: &cancellables)

            // Observe preview images
            preview.$currentImage
                .receive(on: DispatchQueue.main)
                .sink { [weak self] img in
                    self?.previewImage = img
                }
                .store(in: &cancellables)

            // Start polling
            poller.start()
            preview.start()

            // Load initial data
            await refreshSettings()
            await refreshCameras()
            await loadProfiles()
            await loadRecentEvents()

            dashboardState = .serviceReadyIdle
            logger.info("App store ready")

        case .failed(let msg):
            dashboardState = .serviceFailed(msg)

        case .starting:
            dashboardState = .launchingService

        case .stopped:
            dashboardState = .serviceReadyIdle

        case .stopping:
            break
        }
    }

    private func applyStatus(_ status: ServiceStatus?) {
        guard let status = status else { return }

        // Reset tracking when monitoring stops
        if isMonitoring && !status.monitoring {
            monitoringStartTime = nil
            riskHistory = []
            riskScore = 0
            riskPercent = 0
            confidencePercent = 0
        }
        isMonitoring = status.monitoring
        isLoading = status.loading

        if let perf = status.performance {
            fps = perf.fps
        }

        if let pred = status.prediction {
            modelState = pred.state
            businessState = pred.businessState
            riskScore = pred.riskScore
            riskPercent = status.riskPercent
            confidencePercent = Int(round(max(0, min(1, pred.confidence)) * 100))
            visibility = Int(round(pred.visibility * 100))
            personVisible = status.monitoring && pred.state != .unknown

            // Track risk history for trend chart
            if status.monitoring {
                riskHistory.append(status.riskPercent)
                if riskHistory.count > maxRiskHistory {
                    riskHistory.removeFirst(riskHistory.count - maxRiskHistory)
                }
            }
        }

        if let err = status.error {
            handleAPIError(err)
        }

        // Update dashboard state
        updateDashboardState(from: status)

        // Notify on fall/pre-fall events
        checkAndNotify(status)

        // Sync window visibility to poller and preview
        statusPoller?.isMonitoring = status.monitoring
        previewClient?.isMonitoring = status.monitoring
    }

    private func updateDashboardState(from status: ServiceStatus) {
        if status.error != nil {
            return  // keep current error state
        }
        if status.loading {
            dashboardState = .requestingCamera
            return
        }
        if !status.monitoring {
            dashboardState = .serviceReadyIdle
            return
        }
        switch status.prediction?.state {
        case .none, .unknown:
            dashboardState = .personUnknown
        case .normal:
            dashboardState = .monitoringNormal
        case .preFall:
            dashboardState = .monitoringPreFall
        case .fall:
            dashboardState = .monitoringFall
        }
    }

    /// Send a native notification when the business state transitions to
    /// warning or danger — but only once per event.
    private func checkAndNotify(_ status: ServiceStatus) {
        guard let pred = status.prediction else { return }
        // The Python service tracks events through EventService.
        // We poll recent events to catch new ones.
        // For simplicity here: notify based on state transitions.
        let eventType: String?
        switch pred.businessState {
        case .danger: eventType = "fall"
        case .warning: eventType = "pre-fall"
        default: eventType = nil
        }

        guard let type = eventType else {
            lastNotifiedEventId = nil
            return
        }

        // Notify once per continuous warning/fall episode. `sequence` changes
        // on every status poll and would otherwise create notification spam.
        guard type != lastNotifiedEventId else { return }
        lastNotifiedEventId = type

        notificationService.notifyIfNew(
            eventId: type,
            eventType: type,
            riskPercent: Int(round(pred.riskScore * 100))
        )
        totalAlerts += 1
        if type == "fall" { highRiskEvents += 1 }
    }

    private func handleAPIError(_ error: APIError) {
        connectionError = error.localizedDescription
    }

    private func handleAPIError(_ dto: ServiceErrorDTO) {
        connectionError = dto.messageKey
    }
}
