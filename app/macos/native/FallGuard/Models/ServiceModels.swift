import Foundation

// MARK: - API Protocol Version

/// Current API contract version — must match the Python service's ``API_VERSION``.
enum APIVersion {
    static let current = "v1"
}

/// Shared thresholds for the risk percentage shown by the dashboard.
enum RiskDisplayThresholds {
    static let warningPercent = 45
    static let dangerPercent = 72
}

// MARK: - Health

/// Service health status returned by `GET /api/v1/health`.
struct ServiceHealth: Decodable, Equatable {
    let status: HealthStatus
    let version: String
    let apiVersion: String
    let models: ModelStatus
    let database: Bool
    let cameraAvailable: Bool

    enum HealthStatus: String, Decodable {
        case starting, ready, degraded
    }

    struct ModelStatus: Decodable, Equatable {
        let yolo: Bool
        let classifier: Bool
    }

    enum CodingKeys: String, CodingKey {
        case status, version
        case apiVersion = "api_version"
        case models, database
        case cameraAvailable = "camera_available"
    }
}

// MARK: - Monitor Status

/// Full monitoring status returned by `GET /api/v1/status`.
struct ServiceStatus: Decodable, Equatable {
    let schemaVersion: Int
    let sequence: Int64
    let timestampMs: Int64
    let monitoring: Bool
    let loading: Bool
    let prediction: PredictionDTO?
    let performance: PerformanceDTO?
    let error: ServiceErrorDTO?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sequence
        case timestampMs = "timestamp_ms"
        case monitoring, loading, prediction, performance, error
    }
}

// MARK: - Prediction

struct PredictionDTO: Decodable, Equatable {
    let state: ModelState
    let alertState: ModelState
    let businessState: BusinessState
    let riskScore: Double
    let visibility: Double
    let confidence: Double
    let systemStatus: String?

    /// Model-level prediction state.
    enum ModelState: String, Decodable, Equatable {
        case normal = "Normal"
        case preFall = "Pre-fall"
        case fall = "Fall"
        case unknown = "Unknown"
    }

    /// Business-level safety state for UI mapping.
    enum BusinessState: String, Decodable, Equatable {
        case safe, warning, danger, unknown
    }

    enum CodingKeys: String, CodingKey {
        case state, visibility, confidence
        case alertState = "alert_state"
        case businessState = "business_state"
        case riskScore = "risk_score"
        case systemStatus = "system_status"
    }
}

// MARK: - Performance

struct PerformanceDTO: Decodable, Equatable {
    let fps: Double
    let frameIndex: Int

    enum CodingKeys: String, CodingKey {
        case fps
        case frameIndex = "frame_index"
    }
}

// MARK: - Command Response

/// Response from monitor start/stop commands.
struct MonitorCommandResponse: Decodable, Equatable {
    let ok: Bool
    let monitoring: Bool
    let sessionId: String?
    let error: ServiceErrorDTO?

    enum CodingKeys: String, CodingKey {
        case ok, monitoring, error
        case sessionId = "session_id"
    }
}

// MARK: - Settings

struct ServiceSettings: Decodable, Equatable {
    let sensitivity: String
    let cameraIndex: Int
    let theme: String
    let lang: String
    let soundAlert: Bool
    let thresholds: [String: Double]

    enum CodingKeys: String, CodingKey {
        case sensitivity, theme, lang, thresholds
        case cameraIndex = "camera_index"
        case soundAlert = "sound_alert"
    }
}

// MARK: - Profile

struct ProfileDTO: Decodable, Equatable, Identifiable {
    let id: String
    let name: String
    let createdAt: String
    let fallCount: Int

    enum CodingKeys: String, CodingKey {
        case id, name, fallCount
        case createdAt = "createdAt"
    }
}

struct ProfileListResponse: Decodable {
    let profiles: [ProfileDTO]
    let activeId: String?
    let activeProfile: ProfileDTO?

    enum CodingKeys: String, CodingKey {
        case profiles, activeProfile
        case activeId = "activeId"
    }
}

// MARK: - Event

struct EventDTO: Decodable, Equatable, Identifiable, Hashable {
    let id: String
    let eventType: String
    let status: String
    let peakRisk: Double
    let startedAt: String
    let endedAt: String?
    let sessionId: String?

    enum CodingKeys: String, CodingKey {
        case id, status
        case eventType = "event_type"
        case peakRisk = "peak_risk"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case sessionId = "session_id"
    }
}

// MARK: - Session

struct SessionDTO: Decodable, Identifiable {
    let id: String
    let profileId: String
    let sourceType: String
    let status: String
    let totalFrames: Int
    let totalEvents: Int
    let peakRisk: Double
    let startedAt: String
    let endedAt: String?

    enum CodingKeys: String, CodingKey {
        case id, status
        case profileId = "profile_id"
        case sourceType = "source_type"
        case totalFrames = "total_frames"
        case totalEvents = "total_events"
        case peakRisk = "peak_risk"
        case startedAt = "started_at"
        case endedAt = "ended_at"
    }
}

// MARK: - Import Job

struct ImportJobDTO: Decodable, Equatable {
    let id: String
    let state: ImportState
    let progress: Double
    let currentFrame: Int
    let totalFrames: Int
    let outputVideo: String?
    let error: ServiceErrorDTO?

    enum ImportState: String, Decodable, Equatable {
        case idle = "idle"
        case running
        case complete
        case error
    }

    enum CodingKeys: String, CodingKey {
        case id, state, progress, error
        case currentFrame = "current_frame"
        case totalFrames = "total_frames"
        case outputVideo = "output_video"
    }
}

// MARK: - Paginated Response

struct PaginatedResponse<T: Decodable>: Decodable {
    let items: [T]
    let nextCursor: String?
    let hasMore: Bool

    enum CodingKeys: String, CodingKey {
        case items
        case nextCursor = "next_cursor"
        case hasMore = "has_more"
    }
}

// MARK: - API Error

/// Stable error envelope from the Python service.
struct ServiceErrorDTO: Decodable, Equatable {
    let code: String
    let messageKey: String
    let retryable: Bool
    let details: String?

    enum CodingKeys: String, CodingKey {
        case code
        case messageKey = "message_key"
        case retryable, details
    }
}

/// Top-level error response wrapper.
struct APIErrorResponse: Decodable {
    let error: ServiceErrorDTO
}

// MARK: - Ready Message

/// The ``ready`` JSON line printed by the Python service on stdout.
struct ReadyMessage: Decodable {
    let event: String       // always "ready"
    let port: Int
    let token: String
    let apiVersion: String
    let pid: Int

    enum CodingKeys: String, CodingKey {
        case event, port, token, pid
        case apiVersion = "api_version"
    }
}

// MARK: - Camera Info

struct CameraListResponse: Decodable {
    let cameras: [Int]
    let current: Int
}

// MARK: - Generic wrappers

struct OkResponse: Decodable {
    let ok: Bool
    let message: String?
}

struct ProfileActionResponse: Decodable {
    let ok: Bool
    let activeId: String?
    let profile: ProfileDTO?

    enum CodingKeys: String, CodingKey {
        case ok, profile
        case activeId = "activeId"
    }
}

struct ImportCreateResponse: Decodable {
    let ok: Bool
    let `import`: ImportJobDTO
}

// MARK: - Helpers

extension ServiceStatus {
    /// Whether the monitor is actively running (not loading, not idle).
    var isActive: Bool {
        monitoring && !loading
    }

    /// A display-friendly risk percentage (0–100).
    var riskPercent: Int {
        guard let p = prediction else { return 0 }
        return min(100, max(0, Int(round(p.riskScore * 100))))
    }

    /// Whether the person is currently visible.
    var personVisible: Bool {
        prediction?.state != .unknown
    }
}

extension PredictionDTO.ModelState {
    /// Localized display name for the model state.
    var displayName: String {
        switch self {
        case .normal: return NSLocalizedString("state.normal", comment: "Normal")
        case .preFall: return NSLocalizedString("state.prefall", comment: "Pre-fall")
        case .fall: return NSLocalizedString("state.fall", comment: "Fall")
        case .unknown: return NSLocalizedString("state.unknown", comment: "Unknown")
        }
    }
}
