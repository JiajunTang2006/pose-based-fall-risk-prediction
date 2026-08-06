import SwiftUI
import Foundation

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

@MainActor
final class ThemeManager: ObservableObject {

    @Published var mode: ThemeMode = .system

    var effective: ColorScheme? {
        switch mode {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }

    func resolve(osScheme: ColorScheme) -> ColorScheme {
        switch mode {
        case .system: return osScheme
        case .light:  return .light
        case .dark:   return .dark
        }
    }
}

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
        resolvedLanguage
    }

    nonisolated static var formattingLocale: Locale {
        Locale(identifier: resolvedLanguage == "zh" ? "zh_CN" : "en_US")
    }

    nonisolated static func localizedString(forKey key: String) -> String {
        let language = resolvedLanguage
        guard
            let path = Bundle.main.path(forResource: language, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            return Bundle.main.localizedString(forKey: key, value: key, table: nil)
        }
        return bundle.localizedString(forKey: key, value: key, table: nil)
    }

    nonisolated private static var resolvedLanguage: String {
        let saved = UserDefaults.standard.string(forKey: "FallGuardSelectedLanguage")
        let preferred = saved ?? (Locale.preferredLanguages.first ?? "en")
        return preferred.lowercased().hasPrefix("zh") ? "zh" : "en"
    }
}

func NSLocalizedString(_ key: String, comment: String) -> String {
    LanguageManager.localizedString(forKey: key)
}

extension Notification.Name {
    static let fallGuardLanguageDidChange = Notification.Name("FallGuardLanguageDidChange")
}
