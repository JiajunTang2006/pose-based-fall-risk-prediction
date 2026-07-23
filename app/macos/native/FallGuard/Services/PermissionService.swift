import AppKit
import AVFoundation
import OSLog

/// Checks and requests macOS permissions (camera, microphone, notifications).
///
/// The camera permission check is informational — the actual camera access
/// happens inside the Python process (OpenCV).  This service provides the
/// user-facing guidance to open System Settings when permission is denied.
enum PermissionService {

    private static let logger = Logger(subsystem: "com.fallguard.desktop", category: "Permission")

    // MARK: Camera

    /// Check current camera authorization status.
    static var cameraStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .video)
    }

    /// Request camera permission.  Returns the resulting status.
    static func requestCameraPermission() async -> AVAuthorizationStatus {
        let current = AVCaptureDevice.authorizationStatus(for: .video)
        switch current {
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
                ? .authorized : .denied
        default:
            return current
        }
    }

    /// Open the Security & Privacy → Camera preference pane.
    static func openCameraSettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera")!
        NSWorkspace.shared.open(url)
    }

    /// Whether we should show the "Open Settings" camera prompt.
    static var needsCameraSettingsPrompt: Bool {
        let status = cameraStatus
        return status == .denied || status == .restricted
    }

    // MARK: Microphone (future use)

    static var microphoneStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    static func openMicrophoneSettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
        NSWorkspace.shared.open(url)
    }

    // MARK: General

    /// Open the main Security & Privacy preference pane.
    static func openSecuritySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security")!
        NSWorkspace.shared.open(url)
    }
}
