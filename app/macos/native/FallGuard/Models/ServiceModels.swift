import Foundation

enum APIVersion {
    static let current = "v1"
}

enum RiskDisplayThresholds {
    static let warningPercent = 45
    static let dangerPercent = 72
}

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

struct ServiceStatus: Decodable, Equatable {
    let schemaVersion: Int
    let sequence: Int64
    let timestampMs: Int64
    let monitoring: Bool
    let loading: Bool
    let activeEventId: String?
    let prediction: PredictionDTO?
    let performance: PerformanceDTO?
    let error: ServiceErrorDTO?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sequence
        case timestampMs = "timestamp_ms"
        case monitoring, loading, prediction, performance, error
        case activeEventId = "active_event_id"
    }
}

struct PredictionDTO: Decodable, Equatable {
    let state: ModelState
    let alertState: ModelState
    let businessState: BusinessState
    let riskScore: Double
    let visibility: Double
    let confidence: Double
    let systemStatus: String?

    enum ModelState: String, Decodable, Equatable {
        case normal = "Normal"
        case preFall = "Pre-fall"
        case fall = "Fall"
        case unknown = "Unknown"
    }

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

struct PerformanceDTO: Decodable, Equatable {
    let fps: Double
    let frameIndex: Int

    enum CodingKeys: String, CodingKey {
        case fps
        case frameIndex = "frame_index"
    }
}

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

struct EventDTO: Decodable, Equatable, Identifiable, Hashable {
    let id: String
    let eventType: String
    let status: String
    let peakRisk: Double
    let startedAt: String
    let endedAt: String?
    let sessionId: String?
    let avgRisk: Double
    let durationSeconds: Double
    let thumbnailPath: String?
    let videoClipPath: String?
    let clipFps: Double?
    let clipDurationSeconds: Double?
    let userFeedback: String?
    let annotationLabel: String?
    let prefallStartSeconds: Double?
    let fallStartSeconds: Double?
    let notes: String?

    enum CodingKeys: String, CodingKey {
        case id, status
        case eventType = "event_type"
        case peakRisk = "peak_risk"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case sessionId = "session_id"
        case avgRisk = "avg_risk"
        case durationSeconds = "duration_seconds"
        case thumbnailPath = "thumbnail_path"
        case videoClipPath = "video_clip_path"
        case clipFps = "clip_fps"
        case clipDurationSeconds = "clip_duration_seconds"
        case userFeedback = "user_feedback"
        case annotationLabel = "annotation_label"
        case prefallStartSeconds = "prefall_start_seconds"
        case fallStartSeconds = "fall_start_seconds"
        case notes
    }
}

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

struct APIErrorResponse: Decodable {
    let error: ServiceErrorDTO
}

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

struct CameraListResponse: Decodable {
    let cameras: [Int]
    let current: Int
}

struct OkResponse: Decodable {
    let ok: Bool
    let message: String?
}

struct ClearHistoryResponse: Decodable {
    let ok: Bool
    let removed: [String: Int]
}

struct DatasetExportResponse: Decodable {
    let ok: Bool
    let outputDirectory: String
    let annotationsPath: String
    let videoCount: Int
    let annotationCount: Int
    let skippedCount: Int

    enum CodingKeys: String, CodingKey {
        case ok
        case outputDirectory = "output_directory"
        case annotationsPath = "annotations_path"
        case videoCount = "video_count"
        case annotationCount = "annotation_count"
        case skippedCount = "skipped_count"
    }
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

extension ServiceStatus {
    var isActive: Bool {
        monitoring && !loading
    }

    var riskPercent: Int {
        guard let p = prediction else { return 0 }
        return min(100, max(0, Int(round(p.riskScore * 100))))
    }

    var personVisible: Bool {
        prediction?.state != .unknown
    }
}

extension PredictionDTO.ModelState {
    var displayName: String {
        switch self {
        case .normal: return NSLocalizedString("state.normal", comment: "Normal")
        case .preFall: return NSLocalizedString("state.prefall", comment: "Pre-fall")
        case .fall: return NSLocalizedString("state.fall", comment: "Fall")
        case .unknown: return NSLocalizedString("state.unknown", comment: "Unknown")
        }
    }
}
