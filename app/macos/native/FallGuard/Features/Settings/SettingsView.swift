import SwiftUI
import ServiceManagement
import UniformTypeIdentifiers

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
            DispatchQueue.main.async {
                let title = String(
                    format: "%@ %@",
                    "FallGuard",
                    NSLocalizedString("settings.title", comment: "")
                )
                NSApp.windows.first {
                    $0.identifier?.rawValue == "com_apple_SwiftUI_Settings_window"
                }?.title = title
            }
        }
    }
}

struct GeneralPage: View {
    @EnvironmentObject var store: AppStore
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var languageManager: LanguageManager
    let scheme: ColorScheme
    @State private var launchAtLogin = false

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
                    .frame(maxWidth: .infinity, alignment: .center)
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
                    .frame(maxWidth: .infinity, alignment: .center)
                }

                if #available(macOS 13.0, *) {
                    SettingGroup(label: "settings.startup", scheme: scheme) {
                        VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
                            Toggle("settings.launch_at_login", isOn: Binding(
                                get: { launchAtLogin },
                                set: { updateLaunchAtLogin($0) }
                            ))
                            Text("settings.launch_at_login_note")
                                .font(FallGuardFont.caption2)
                                .foregroundColor(FallGuardColors.muted(for: scheme))
                        }
                    }
                }
            }
            .frame(maxWidth: 560, alignment: .leading)
            .padding(.vertical, FallGuardSpacing.s24)
            .frame(maxWidth: .infinity, alignment: .top)
        }
        .onAppear {
            if #available(macOS 13.0, *) {
                launchAtLogin = SMAppService.mainApp.status == .enabled
            }
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

    @available(macOS 13.0, *)
    private func updateLaunchAtLogin(_ enabled: Bool) {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            launchAtLogin = SMAppService.mainApp.status == .enabled
        } catch {
            launchAtLogin = SMAppService.mainApp.status == .enabled
            store.connectionError = error.localizedDescription
        }
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
                        .frame(maxWidth: .infinity, alignment: .center)

                        if let thresholds = store.settings?.thresholds, !thresholds.isEmpty {
                            HStack(spacing: FallGuardSpacing.s24) {
                                ForEach(thresholds.sorted(by: { $0.key < $1.key }), id: \.key) { key, val in
                                    VStack(spacing: 2) {
                                        Text(String(format: "%.2f", val))
                                            .font(.system(.callout, design: .rounded).bold())
                                            .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                                        Text(NSLocalizedString("threshold.\(key)", comment: ""))
                                            .font(FallGuardFont.caption2)
                                            .foregroundColor(FallGuardColors.muted(for: scheme))
                                    }
                                }
                            }
                            .padding(.top, FallGuardSpacing.s4)
                            .frame(maxWidth: .infinity, alignment: .center)
                        }
                    }
                }
            }
            .frame(maxWidth: 560, alignment: .leading)
            .padding(.vertical, FallGuardSpacing.s24)
            .frame(maxWidth: .infinity, alignment: .top)
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
                    VStack(alignment: .leading, spacing: FallGuardSpacing.s12) {
                        HStack {
                            Label(
                                store.notificationsAuthorized
                                    ? "settings.system_notifications_enabled"
                                    : "settings.system_notifications_disabled",
                                systemImage: store.notificationsAuthorized
                                    ? "checkmark.circle.fill" : "exclamationmark.circle"
                            )
                            Spacer()
                            if !store.notificationsAuthorized {
                                Button("settings.open_notification_settings") {
                                    PermissionService.openNotificationSettings()
                                }
                            }
                        }

                        Toggle("settings.sound_alert", isOn: $soundAlert)
                            .onChange(of: soundAlert) { newVal in
                                Task { await store.updateSettings(["sound_alert": newVal]) }
                            }
                        Text("settings.sound_note")
                            .font(FallGuardFont.caption2)
                            .foregroundColor(FallGuardColors.muted(for: scheme))
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
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
    @State private var showingClearConfirmation = false
    @State private var clearSucceeded = false
    @State private var showingExportResult = false
    @State private var exportMessage = ""
    @State private var exportedDirectory: URL?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s20) {
                PageHeader(title: "settings.tab.data", icon: "folder", scheme: scheme)

                SettingGroup(label: "settings.data_management", scheme: scheme) {
                    VStack(spacing: FallGuardSpacing.s12) {
                        SettingActionRow(
                            icon: "shippingbox.and.arrow.backward",
                            title: "settings.export_dataset",
                            detail: "settings.export_dataset_detail",
                            scheme: scheme,
                            action: { exportTrainingDataset() }
                        )
                        Divider()
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
                            action: { showingClearConfirmation = true }
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
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(FallGuardSpacing.s24)
        }
        .alert("settings.clear_history_confirm_title",
               isPresented: $showingClearConfirmation) {
            Button("cancel", role: .cancel) {}
            Button("settings.clear_history", role: .destructive) {
                Task {
                    clearSucceeded = await store.clearHistory()
                }
            }
        } message: {
            Text("settings.clear_history_confirm_message")
        }
        .alert("settings.clear_history_complete",
               isPresented: $clearSucceeded) {
            Button("ok") {}
        }
        .alert("settings.export_dataset_complete",
               isPresented: $showingExportResult) {
            Button("ok") {}
            if let exportedDirectory {
                Button("settings.export_dataset_open") {
                    NSWorkspace.shared.open(exportedDirectory)
                }
            }
        } message: {
            Text(exportMessage)
        }
    }

    private func exportLogs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "fallguard_logs.json"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            let settingsPayload: [String: Any] = [
                "sensitivity": store.settings?.sensitivity ?? "",
                "camera_index": store.settings?.cameraIndex ?? 0,
                "theme": store.settings?.theme ?? "",
                "language": store.settings?.lang ?? "",
            ]
            let eventPayload = store.recentEvents.map { event in
                [
                    "id": event.id,
                    "type": event.eventType,
                    "status": event.status,
                    "peak_risk": event.peakRisk,
                    "started_at": event.startedAt,
                    "ended_at": event.endedAt as Any? ?? NSNull(),
                ] as [String: Any]
            }
            let payload: [String: Any] = [
                "exported_at": ISO8601DateFormatter().string(from: Date()),
                "app_version": Bundle.main.object(
                    forInfoDictionaryKey: "CFBundleShortVersionString"
                ) as? String ?? "unknown",
                "service": store.serviceManager.state.displayText,
                "settings": settingsPayload,
                "recent_events": eventPayload,
            ]
            do {
                let data = try JSONSerialization.data(
                    withJSONObject: payload,
                    options: [.prettyPrinted, .sortedKeys]
                )
                try data.write(to: url, options: .atomic)
            } catch {
                Task { @MainActor in
                    store.connectionError = error.localizedDescription
                }
            }
        }
    }

    private func exportTrainingDataset() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = NSLocalizedString("settings.export_dataset_choose", comment: "")
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            Task {
                guard let result = await store.exportTrainingDataset(
                    to: destination.path
                ) else { return }
                await MainActor.run {
                    exportedDirectory = URL(
                        fileURLWithPath: result.outputDirectory,
                        isDirectory: true
                    )
                    exportMessage = String(
                        format: NSLocalizedString(
                            "settings.export_dataset_result", comment: ""
                        ),
                        result.videoCount,
                        result.annotationCount
                    )
                    showingExportResult = true
                }
            }
        }
    }

    private func openDataFolder() {
        let root = FileManager.default.urls(
            for: .moviesDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("FallGuard", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(root)
    }
}

struct AboutPage: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme
    @State private var showingSafetyNotice = false

    var body: some View {
        VStack(spacing: FallGuardSpacing.s24) {
            Spacer()

            ZStack {
                Circle()
                    .fill(FallGuardColors.primary(for: scheme).opacity(0.1))
                    .frame(width: 80, height: 80)
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(width: 54, height: 54)
            }

            Text("FallGuard")
                .font(FallGuardFont.title)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))

            Text(String(
                format: NSLocalizedString("settings.version", comment: ""),
                Bundle.main.object(
                    forInfoDictionaryKey: "CFBundleShortVersionString"
                ) as? String ?? "0.3.3"
            ))
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

            Button {
                showingSafetyNotice = true
            } label: {
                Label(
                    NSLocalizedString("settings.view_safety_notice", comment: ""),
                    systemImage: "shield.lefthalf.filled"
                )
                .font(.system(size: 14, weight: .semibold))
                .padding(.horizontal, FallGuardSpacing.s16)
                .frame(minHeight: 34)
            }
            .buttonStyle(FallGuardSecondaryButtonStyle(scheme: scheme))

            Text("settings.copyright")
                .font(FallGuardFont.caption2)
                .foregroundColor(FallGuardColors.muted(for: scheme))

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(isPresented: $showingSafetyNotice) {
            SafetyNoticeView(
                scheme: scheme,
                onCancel: { showingSafetyNotice = false },
                onAcknowledge: { showingSafetyNotice = false }
            )
        }
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
                .frame(maxWidth: .infinity, alignment: .leading)
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
        .frame(maxWidth: .infinity, alignment: .leading)
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
