import Foundation
import OSLog

/// Typed HTTP client for the FallGuard AI Service ``/api/v1`` endpoints.
///
/// Every method throws ``APIError`` on failure.  The client does **not**
/// perform its own retries — that is the caller's responsibility.
struct FallGuardAPIClient {
    let baseURL: URL
    let token: String
    let session: URLSession

    private let logger = Logger(subsystem: "com.fallguard.desktop", category: "APIClient")
    // DTOs define explicit CodingKeys for the service's snake_case contract.
    // Applying convertFromSnakeCase here as well would transform every key twice.
    private let decoder = JSONDecoder()

    init(baseURL: URL, token: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    // MARK: Health & Status

    func health() async throws -> ServiceHealth {
        try await get(path: "health", auth: false)
    }

    func status() async throws -> ServiceStatus {
        try await get(path: "status")
    }

    // MARK: Monitor

    func startMonitoring() async throws -> MonitorCommandResponse {
        try await post(path: "monitor/start")
    }

    func stopMonitoring() async throws -> MonitorCommandResponse {
        try await post(path: "monitor/stop")
    }

    // MARK: Settings

    func getSettings() async throws -> ServiceSettings {
        try await get(path: "settings")
    }

    func updateSettings(_ body: [String: Any]) async throws -> ServiceSettings {
        let data = try JSONSerialization.data(withJSONObject: body)
        return try await put(path: "settings", body: data)
    }

    // MARK: Cameras

    func getCameras() async throws -> CameraListResponse {
        try await get(path: "cameras")
    }

    // MARK: Profiles

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

    // MARK: Events

    func getEvents(limit: Int = 50, cursor: String? = nil,
                   profileId: String? = nil) async throws -> PaginatedResponse<EventDTO> {
        var query = "limit=\(min(limit, 200))"
        if let c = cursor { query += "&cursor=\(c)" }
        if let p = profileId { query += "&profile_id=\(p)" }
        return try await get(path: "events?\(query)")
    }

    // MARK: Sessions

    func getSessions(limit: Int = 50) async throws -> PaginatedResponse<SessionDTO> {
        try await get(path: "sessions?limit=\(min(limit, 200))")
    }

    // MARK: Import

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

    // MARK: Preview

    /// Fetch the latest JPEG frame from the service.
    func latestFrame() async throws -> Data {
        var req = try makeRequest(path: "preview.jpg", method: "GET")
        req.timeoutInterval = 5
        let (data, response) = try await session.data(for: req)
        try validate(response: response, data: data)
        return data
    }

    /// Returns a streaming URL for MJPEG playback.
    func mjpegStreamURL() -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.path = "/api/v1/preview.mjpg"
        return components.url!
    }

    // MARK: Shutdown

    func shutdown() async throws {
        let _: OkResponse = try await post(path: "shutdown")
    }

    // MARK: Media

    func mediaContentURL(mediaId: String) -> URL {
        baseURL
            .deletingLastPathComponent()  // remove /v1
            .appendingPathComponent("media")
            .appendingPathComponent(mediaId)
            .appendingPathComponent("content")
    }

    // MARK: Private helpers

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
            // Try to decode a structured error
            if let apiErr = try? decoder.decode(APIErrorResponse.self, from: data) {
                throw APIError.serviceError(apiErr.error)
            }
            throw APIError.httpError(statusCode: httpResp.statusCode, body: data)
        }
    }
}

// MARK: - API Error

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
            // Return the message key so the UI layer can localize it
            return dto.messageKey
        case .decodingError(let e):
            return "Decoding error: \(e.localizedDescription)"
        }
    }

    /// Whether this error is retryable.
    var isRetryable: Bool {
        switch self {
        case .transportError: return true
        case .httpError(let code, _): return code >= 500
        case .serviceError(let dto): return dto.retryable
        case .decodingError: return false
        }
    }

    /// The stable error code, if available.
    var errorCode: String? {
        if case .serviceError(let dto) = self { return dto.code }
        return nil
    }
}
