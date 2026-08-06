import AppKit
import AVFoundation
import OSLog

enum PermissionService {

    private static let logger = Logger(subsystem: "com.fallguard.desktop", category: "Permission")

    static var cameraStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .video)
    }

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

    static func openCameraSettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera")!
        NSWorkspace.shared.open(url)
    }

    static var needsCameraSettingsPrompt: Bool {
        let status = cameraStatus
        return status == .denied || status == .restricted
    }

    static var microphoneStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    static func openMicrophoneSettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
        NSWorkspace.shared.open(url)
    }

    static func openNotificationSettings() {
        let bundleID = Bundle.main.bundleIdentifier ?? "com.fallguard.desktop"
        let encoded = bundleID.addingPercentEncoding(
            withAllowedCharacters: .urlQueryAllowed
        ) ?? bundleID
        let url = URL(string:
            "x-apple.systempreferences:com.apple.Notifications-Settings.extension?bundleId=\(encoded)"
        )!
        NSWorkspace.shared.open(url)
    }

    static func openSecuritySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security")!
        NSWorkspace.shared.open(url)
    }
}
