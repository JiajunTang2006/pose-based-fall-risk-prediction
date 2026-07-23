import SwiftUI
import Foundation

// MARK: - Theme Mode

enum ThemeMode: String, CaseIterable {
    case system = "system"
    case light  = "light"
    case dark   = "dark"

    var displayName: String {
        switch self {
        case .system: return NSLocalizedString("theme.system", comment: "System")
        case .light:  return NSLocalizedString("theme.light", comment: "Light")
        case .dark:   return NSLocalizedString("theme.dark", comment: "Dark")
        }
    }
}

// MARK: - Theme Manager

/// Manages the app's theme mode and provides the effective color scheme.
///
/// When mode is ``system``, the effective scheme follows the OS appearance.
/// When mode is ``light`` or ``dark``, the effective scheme is forced.
@MainActor
final class ThemeManager: ObservableObject {

    @Published var mode: ThemeMode = .system

    /// The resolved effective color scheme.
    /// Returns `nil` for ``system`` mode (SwiftUI follows the OS).
    var effective: ColorScheme? {
        switch mode {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }

    /// Returns a concrete ``ColorScheme``, falling back to the OS when in system mode.
    func resolve(osScheme: ColorScheme) -> ColorScheme {
        switch mode {
        case .system: return osScheme
        case .light:  return .light
        case .dark:   return .dark
        }
    }
}

// MARK: - Application Language

/// Controls the app language independently of the macOS system language.
/// SwiftUI views observe this object through the locale environment, while
/// imperative AppKit strings use the module-level localization helper below.
@MainActor
final class LanguageManager: ObservableObject {
    @Published private(set) var language: String

    init() {
        language = Self.savedLanguage
    }

    var locale: Locale {
        Locale(identifier: language == "zh" ? "zh-Hans" : "en")
    }

    func setLanguage(_ newLanguage: String) {
        let normalized = newLanguage == "zh" ? "zh" : "en"
        guard normalized != language else { return }
        language = normalized
        UserDefaults.standard.set(normalized, forKey: "FallGuardSelectedLanguage")
        UserDefaults.standard.synchronize()
        NotificationCenter.default.post(name: .fallGuardLanguageDidChange, object: nil)
    }

    static var savedLanguage: String {
        if let saved = UserDefaults.standard.string(forKey: "FallGuardSelectedLanguage") {
            return saved == "zh" ? "zh" : "en"
        }
        let preferred = Locale.preferredLanguages.first ?? "en"
        return preferred.lowercased().hasPrefix("zh") ? "zh" : "en"
    }

    nonisolated static func localizedString(forKey key: String) -> String {
        let saved = UserDefaults.standard.string(forKey: "FallGuardSelectedLanguage")
        let preferred = saved ?? (Locale.preferredLanguages.first ?? "en")
        let language = preferred.lowercased().hasPrefix("zh") ? "zh" : "en"
        guard
            let path = Bundle.main.path(forResource: language, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            return Bundle.main.localizedString(forKey: key, value: key, table: nil)
        }
        return bundle.localizedString(forKey: key, value: key, table: nil)
    }
}

/// Module-local replacement used by existing AppKit and formatted strings.
/// It follows ``LanguageManager`` instead of the operating-system language.
func NSLocalizedString(_ key: String, comment: String) -> String {
    LanguageManager.localizedString(forKey: key)
}

extension Notification.Name {
    static let fallGuardLanguageDidChange = Notification.Name("FallGuardLanguageDidChange")
}
