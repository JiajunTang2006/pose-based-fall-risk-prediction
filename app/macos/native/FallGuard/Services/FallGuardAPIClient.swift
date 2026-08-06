import Foundation
import OSLog

struct FallGuardAPIClient {
    let baseURL: URL
    let token: String
    let session: URLSession

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "APIClient")
    private let decoder = JSONDecoder()

    init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    func health() async throws -> ServiceHealth {
        try await get(path: "health", auth: false)
    }

    func status() async throws -> ServiceStatus {
        try await get(path: "status")
    }

    func startMonitoring() async throws -> MonitorCommandResponse {
        try await post(path: "monitor/start")
    }

    func stopMonitoring() async throws -> MonitorCommandResponse {
        try await post(path: "monitor/stop")
    }

    func getSettings() async throws -> ServiceSettings {
        try await get(path: "settings")
    }

    func updateSettings(_ body: [String: Any]) async throws -> ServiceSettings {
        let data = try JSONSerialization.data(withJSONObject: body)
        return try await put(path: "settings", body: data)
    }

    func getCameras() async throws -> CameraListResponse {
        try await get(path: "cameras")
    }

    func getProfiles() async throws -> ProfileListResponse {
        try await get(path: "profiles")
    }

    func createProfile(name: String) async throws -> ProfileActionResponse {
        let body = try JSONSerialization.data(withJSONObject: ["name": name])
        return try await post(path: "profiles", body: body)
    }

    func activateProfile(id: String) async throws -> ProfileActionResponse {
        try await post(path: "profiles/\(id)/activate")
    }

    func updateProfile(id: String, name: String) async throws -> ProfileActionResponse {
        let body = try JSONSerialization.data(withJSONObject: ["name": name])
        return try await put(path: "profiles/\(id)", body: body)
    }

    func deleteProfile(id: String) async throws -> OkResponse {
        try await delete(path: "profiles/\(id)")
    }

    func getEvents(limit: Int = 50, cursor: String? = nil,
                   profileId: String? = nil) async throws -> PaginatedResponse<EventDTO> {
        var query = "limit=\(min(limit, 200))"
        if let c = cursor { query += "&cursor=\(c)" }
        if let p = profileId { query += "&profile_id=\(p)" }
        return try await get(path: "events?\(query)")
    }

    func updateEventFeedback(
        id: String,
        feedback: String,
        notes: String,
        annotationLabel: String? = nil,
        prefallStartSeconds: Double? = nil,
        fallStartSeconds: Double? = nil,
        clipDurationSeconds: Double? = nil
    ) async throws -> EventDTO {
        var values: [String: Any] = [
            "feedback": feedback,
            "notes": notes,
        ]
        if let annotationLabel {
            values["annotation_label"] = annotationLabel
            values["prefall_start_seconds"] = prefallStartSeconds ?? NSNull()
            values["fall_start_seconds"] = fallStartSeconds ?? NSNull()
            values["clip_duration_seconds"] = clipDurationSeconds ?? NSNull()
        }
        let body = try JSONSerialization.data(withJSONObject: values)
        return try await put(path: "events/\(id)/feedback", body: body)
    }

    func exportTrainingDataset(to outputDirectory: String) async throws
        -> DatasetExportResponse {
        let body = try JSONSerialization.data(withJSONObject: [
            "output_directory": outputDirectory,
        ])
        return try await post(path: "dataset/export", body: body, timeout: 120)
    }

    func clearHistory() async throws -> ClearHistoryResponse {
        try await delete(path: "history")
    }

    func getSessions(limit: Int = 50) async throws -> PaginatedResponse<SessionDTO> {
        try await get(path: "sessions?limit=\(min(limit, 200))")
    }

    func createImport(paths: [String], outputDirectory: String?,
                      sensitivity: String) async throws -> ImportCreateResponse {
        var body: [String: Any] = [
            "paths": paths,
            "sensitivity": sensitivity,
        ]
        if let dir = outputDirectory {
            body["output_directory"] = dir
        }
        let data = try JSONSerialization.data(withJSONObject: body)
        return try await post(path: "imports", body: data, timeout: 30)
    }

    func getImport(id: String) async throws -> ImportJobDTO {
        try await get(path: "imports/\(id)")
    }

    func latestFrame() async throws -> Data {
        var req = try makeRequest(path: "preview.jpg", method: "GET")
        req.timeoutInterval = 5
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return data
    }

    func mjpegStreamURL() -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.path = "/api/v1/preview.mjpg"
        return components.url!
    }

    func shutdown() async throws {
        let _: OkResponse = try await post(path: "shutdown")
    }

    func mediaContentURL(mediaId: String) -> URL {
        baseURL
            .deletingLastPathComponent()  // remove /v1
            .appendingPathComponent("media")
            .appendingPathComponent(mediaId)
            .appendingPathComponent("content")
    }

    private func get<T: Decodable>(path: String, auth: Bool = true) async throws -> T {
        var req = try makeRequest(path: path, method: "GET", auth: auth)
        req.timeoutInterval = auth ? 3 : 5
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func post<T: Decodable>(path: String, body: Data? = nil,
                                     timeout: TimeInterval = 10) async throws -> T {
        var req = try makeRequest(path: path, method: "POST")
        req.httpBody = body
        req.timeoutInterval = timeout
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func put<T: Decodable>(path: String, body: Data? = nil) async throws -> T {
        var req = try makeRequest(path: path, method: "PUT")
        req.httpBody = body
        req.timeoutInterval = 10
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func delete<T: Decodable>(path: String) async throws -> T {
        var req = try makeRequest(path: path, method: "DELETE")
        req.timeoutInterval = 10
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func makeRequest(path: String, method: String, auth: Bool = true) throws -> URLRequest {
        let parts = path.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        let endpoint = baseURL.appendingPathComponent(String(parts[0]))
        guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw URLError(.badURL)
        }
        if parts.count == 2 {
            components.percentEncodedQuery = String(parts[1])
        }
        guard let url = components.url else {
            throw URLError(.badURL)
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if auth {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return req
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let httpResp = response as? HTTPURLResponse else {
            throw APIError.transportError(URLError(.badServerResponse))
        }

        guard (200...299).contains(httpResp.statusCode) else {
            if let apiErr = try? decoder.decode(APIErrorResponse.self, from: data) {
                throw APIError.serviceError(apiErr.error)
            }
            throw APIError.httpError(statusCode: httpResp.statusCode, body: data)
        }
    }
}

enum APIError: LocalizedError {
    case transportError(Error)
    case httpError(statusCode: Int, body: Data)
    case serviceError(ServiceErrorDTO)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .transportError(let e):
            return e.localizedDescription
        case .httpError(let code, _):
            return "HTTP \(code)"
        case .serviceError(let dto):
            return dto.messageKey
        case .decodingError(let e):
            return "Decoding error: \(e.localizedDescription)"
        }
    }

    var isRetryable: Bool {
        switch self {
        case .transportError: return true
        case .httpError(let code, _): return code >= 500
        case .serviceError(let dto): return dto.retryable
        case .decodingError: return false
        }
    }

    var errorCode: String? {
        if case .serviceError(let dto) = self { return dto.code }
        return nil
    }
}
