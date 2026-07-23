import SwiftUI

// 文案修改位置：Resources/*/Localizable.strings 中的 Settings、Theme、Sensitivity 分组。
// “关于”页面的 FallGuard 和版本号是本文件中少量直接显示的文字。
/// Settings with sidebar navigation (matching old PySide6 layout).
struct SettingsView: View {
    @EnvironmentObject var store: AppStore
    @EnvironmentObject var themeManager: ThemeManager
    @Environment(\.colorScheme) private var colorScheme
    @State private var selectedPage: SettingsPage = .general

    private var scheme: ColorScheme {
        themeManager.resolve(osScheme: colorScheme)
    }

    enum SettingsPage: String, CaseIterable {
        case general, detection, alerts, data, about

        var label: LocalizedStringKey {
            switch self {
            case .general:   return "settings.tab.general"
            case .detection: return "settings.tab.detection"
            case .alerts:    return "settings.tab.alerts"
            case .data:      return "settings.tab.data"
            case .about:     return "settings.tab.about"
            }
        }

        var icon: String {
            switch self {
            case .general:   return "gearshape"
            case .detection: return "slider.horizontal.3"
            case .alerts:    return "bell"
            case .data:      return "folder"
            case .about:     return "info.circle"
            }
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            // Sidebar nav
            VStack(spacing: 0) {
                ForEach(SettingsPage.allCases, id: \.self) { page in
                    Button { selectedPage = page } label: {
                        HStack(spacing: FallGuardSpacing.s12) {
                            Image(systemName: page.icon)
                                .frame(width: 20)
                            Text(page.label)
                                .font(FallGuardFont.callout)
                            Spacer()
                        }
                        .padding(.horizontal, FallGuardSpacing.s16)
                        .padding(.vertical, FallGuardSpacing.s10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            selectedPage == page
                                ? FallGuardColors.navActiveBg(for: scheme)
                                : Color.clear
                        )
                        .foregroundColor(
                            selectedPage == page
                                ? FallGuardColors.primary(for: scheme)
                                : FallGuardColors.textSecondary(for: scheme)
                        )
                        .overlay(alignment: .leading) {
                            if selectedPage == page {
                                Capsule()
                                    .fill(FallGuardColors.primary(for: scheme))
                                    .frame(width: 3, height: 20)
                                    .padding(.leading, 5)
                                    .allowsHitTesting(false)
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                        .contentShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, FallGuardSpacing.s8)
                }
                Spacer()
            }
            .frame(width: 180)
            .glassSidebar()
            .overlay(
                FallGuardColors.sidebarTint(for: scheme)
                    .opacity(scheme == .dark ? 0.34 : 0.28)
                    .allowsHitTesting(false)
            )

            GlassVerticalDivider()

            // Content
            Group {
                switch selectedPage {
                case .general:   GeneralPage(scheme: scheme)
                case .detection: DetectionPage(scheme: scheme)
                case .alerts:    AlertsPage(scheme: scheme)
                case .data:      DataPage(scheme: scheme)
                case .about:     AboutPage(scheme: scheme)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(FallGuardBackground(scheme: scheme))
        }
        .frame(minWidth: 640, minHeight: 460)
        .onAppear {
            Task {
                if store.settings == nil { await store.refreshSettings() }
            }
        }
    }
}

// MARK: - Pages

struct GeneralPage: View {
    @EnvironmentObject var store: AppStore
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var languageManager: LanguageManager
    let scheme: ColorScheme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s20) {
                PageHeader(title: "settings.tab.general", icon: "gearshape", scheme: scheme)

                SettingGroup(label: "settings.language", scheme: scheme) {
                    HStack(spacing: FallGuardSpacing.s12) {
                        ForEach(["en", "zh"], id: \.self) { lang in
                            Button { changeLanguage(to: lang) } label: {
                                Text(lang == "en" ? "English" : "中文")
                                    .font(FallGuardFont.callout)
                                    .fontWeight(.medium)
                                    .padding(.horizontal, FallGuardSpacing.s16)
                                    .padding(.vertical, FallGuardSpacing.s8)
                                    .background(
                                        languageManager.language == lang
                                            ? FallGuardColors.primary(for: scheme)
                                            : FallGuardColors.line(for: scheme).opacity(0.3)
                                    )
                                    .foregroundColor(
                                        languageManager.language == lang ? .white : FallGuardColors.textPrimary(for: scheme)
                                    )
                                    .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                SettingGroup(label: "settings.theme", scheme: scheme) {
                    HStack(spacing: FallGuardSpacing.s12) {
                        ForEach(ThemeMode.allCases, id: \.self) { mode in
                            Button { themeManager.mode = mode; saveTheme(mode) } label: {
                                Text(mode.displayName)
                                    .font(FallGuardFont.callout)
                                    .fontWeight(.medium)
                                    .padding(.horizontal, FallGuardSpacing.s16)
                                    .padding(.vertical, FallGuardSpacing.s8)
                                    .background(
                                        themeManager.mode == mode
                                            ? FallGuardColors.primary(for: scheme)
                                            : FallGuardColors.line(for: scheme).opacity(0.3)
                                    )
                                    .foregroundColor(
                                        themeManager.mode == mode ? .white : FallGuardColors.textPrimary(for: scheme)
                                    )
                                    .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding(FallGuardSpacing.s24)
        }
    }

    private func changeLanguage(to lang: String) {
        guard languageManager.language != lang else { return }
        languageManager.setLanguage(lang)
        Task { await store.updateSettings(["lang": lang]) }
    }

    private func saveTheme(_ mode: ThemeMode) {
        Task { await store.updateSettings(["theme": mode.rawValue]) }
    }
}

struct DetectionPage: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme
    @State private var sensitivity: String = "medium"

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s20) {
                PageHeader(title: "settings.tab.detection", icon: "slider.horizontal.3", scheme: scheme)

                SettingGroup(label: "settings.sensitivity_level", scheme: scheme) {
                    VStack(alignment: .leading, spacing: FallGuardSpacing.s12) {
                        HStack(spacing: FallGuardSpacing.s12) {
                            ForEach(["low", "medium", "high"], id: \.self) { s in
                                Button { sensitivity = s; saveSensitivity(s) } label: {
                                    Text(NSLocalizedString("sensitivity.\(s)", comment: ""))
                                        .font(FallGuardFont.callout)
                                        .fontWeight(.medium)
                                        .padding(.horizontal, FallGuardSpacing.s16)
                                        .padding(.vertical, FallGuardSpacing.s8)
                                        .background(
                                            sensitivity == s
                                                ? FallGuardColors.primary(for: scheme)
                                                : FallGuardColors.line(for: scheme).opacity(0.3)
                                        )
                                        .foregroundColor(
                                            sensitivity == s ? .white : FallGuardColors.textPrimary(for: scheme)
                                        )
                                        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                                }
                                .buttonStyle(.plain)
                            }
                        }

                        if let thresholds = store.settings?.thresholds, !thresholds.isEmpty {
                            HStack(spacing: FallGuardSpacing.s24) {
                                ForEach(thresholds.sorted(by: { $0.key < $1.key }), id: \.key) { key, val in
                                    VStack(spacing: 2) {
                                        Text(String(format: "%.2f", val))
                                            .font(.system(.callout, design: .rounded).bold())
                                            .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                                        Text(key.replacingOccurrences(of: "_", with: " "))
                                            .font(FallGuardFont.caption2)
                                            .foregroundColor(FallGuardColors.muted(for: scheme))
                                    }
                                }
                            }
                            .padding(.top, FallGuardSpacing.s4)
                        }
                    }
                }
            }
            .padding(FallGuardSpacing.s24)
        }
        .onAppear {
            sensitivity = store.settings?.sensitivity ?? "medium"
        }
    }

    private func saveSensitivity(_ s: String) {
        Task { await store.updateSettings(["sensitivity": s]) }
    }
}

struct AlertsPage: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme
    @State private var soundAlert: Bool = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s20) {
                PageHeader(title: "settings.tab.alerts", icon: "bell", scheme: scheme)

                SettingGroup(label: "settings.notifications", scheme: scheme) {
                    Toggle("settings.sound_alert", isOn: $soundAlert)
                        .onChange(of: soundAlert) { newVal in
                            Task { await store.updateSettings(["sound_alert": newVal]) }
                        }
                    Text("settings.sound_note")
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
            }
            .padding(FallGuardSpacing.s24)
        }
        .onAppear {
            soundAlert = store.settings?.soundAlert ?? true
        }
    }
}

struct DataPage: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s20) {
                PageHeader(title: "settings.tab.data", icon: "folder", scheme: scheme)

                SettingGroup(label: "settings.data_management", scheme: scheme) {
                    VStack(spacing: FallGuardSpacing.s12) {
                        SettingActionRow(
                            icon: "square.and.arrow.up",
                            title: "settings.export_logs",
                            detail: "settings.export_logs_detail",
                            scheme: scheme,
                            action: { exportLogs() }
                        )
                        Divider()
                        SettingActionRow(
                            icon: "trash",
                            title: "settings.clear_history",
                            detail: "settings.clear_history_detail",
                            scheme: scheme,
                            action: { /* TODO */ }
                        )
                        Divider()
                        SettingActionRow(
                            icon: "externaldrive",
                            title: "settings.open_data",
                            detail: "settings.open_data_detail",
                            scheme: scheme,
                            action: { openDataFolder() }
                        )
                    }
                }
            }
            .padding(FallGuardSpacing.s24)
        }
    }

    private func exportLogs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "fallguard_logs.json"
        panel.begin { _ in }
    }

    private func openDataFolder() {
        NSWorkspace.shared.open(FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!)
    }
}

struct AboutPage: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        VStack(spacing: FallGuardSpacing.s24) {
            Spacer()

            ZStack {
                Circle()
                    .fill(FallGuardColors.primary(for: scheme).opacity(0.1))
                    .frame(width: 80, height: 80)
                Image(systemName: "shield.checkered")
                    .font(.system(size: 36))
                    .foregroundColor(FallGuardColors.primary(for: scheme))
            }

            Text("FallGuard")
                .font(FallGuardFont.title)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))

            Text("settings.version 0.3.3")
                .font(FallGuardFont.body)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))

            VStack(alignment: .leading, spacing: FallGuardSpacing.s10) {
                infoLine("settings.api_version", "v1")
                infoLine("settings.service", store.serviceManager.state.displayText)
                if let s = store.settings {
                    infoLine("settings.sensitivity", NSLocalizedString("sensitivity.\(s.sensitivity)", comment: ""))
                    infoLine("settings.camera", "\(s.cameraIndex)")
                }
            }
            .padding(FallGuardSpacing.s20)
            .glassSurface(cornerRadius: FallGuardRadius.lg)
            .overlay(
                RoundedRectangle(cornerRadius: FallGuardRadius.lg)
                    .stroke(scheme == .dark
                        ? Color.white.opacity(0.1)
                        : Color.black.opacity(0.08),
                        lineWidth: 0.5)
                    .allowsHitTesting(false)
            )
            .frame(width: 300)

            Text("settings.copyright")
                .font(FallGuardFont.caption2)
                .foregroundColor(FallGuardColors.muted(for: scheme))

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func infoLine(_ key: LocalizedStringKey, _ value: String) -> some View {
        HStack {
            Text(key)
                .font(FallGuardFont.body)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            Spacer()
            Text(value)
                .font(FallGuardFont.callout)
                .fontWeight(.medium)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
        }
    }
}

// MARK: - Shared Components

struct PageHeader: View {
    let title: LocalizedStringKey
    let icon: String
    let scheme: ColorScheme

    var body: some View {
        HStack(spacing: FallGuardSpacing.s8) {
            Image(systemName: icon)
                .foregroundColor(FallGuardColors.primary(for: scheme))
            Text(title)
                .font(FallGuardFont.title2)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
        }
    }
}

struct SettingGroup<Content: View>: View {
    let label: LocalizedStringKey
    let scheme: ColorScheme
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
            Text(label)
                .font(FallGuardFont.caption)
                .fontWeight(.semibold)
                .foregroundColor(FallGuardColors.muted(for: scheme))
                .textCase(.uppercase)

            content()
                .padding(FallGuardSpacing.s16)
                .glassSurface(cornerRadius: FallGuardRadius.lg)
                .overlay(
                    RoundedRectangle(cornerRadius: FallGuardRadius.lg)
                        .stroke(scheme == .dark
                            ? Color.white.opacity(0.1)
                            : Color.black.opacity(0.08),
                            lineWidth: 0.5)
                        .allowsHitTesting(false)
                )
        }
    }
}

struct SettingActionRow: View {
    let icon: String
    let title: LocalizedStringKey
    let detail: LocalizedStringKey
    let scheme: ColorScheme
    let action: () -> Void

    var body: some View {
        Button { action() } label: {
            HStack(spacing: FallGuardSpacing.s12) {
                Image(systemName: icon)
                    .font(.callout)
                    .foregroundColor(FallGuardColors.primary(for: scheme))
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(FallGuardFont.callout)
                        .fontWeight(.medium)
                        .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    Text(detail)
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(FallGuardColors.muted(for: scheme))
            }
            .padding(.vertical, FallGuardSpacing.s4)
        }
        .buttonStyle(.plain)
    }
}
