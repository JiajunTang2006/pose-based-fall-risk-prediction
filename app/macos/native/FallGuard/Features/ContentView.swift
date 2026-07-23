import SwiftUI
import AppKit

// 文案修改位置：Resources/*/Localizable.strings 中的 Tab Bar、Brand、Buttons、Service 分组。
// 品牌名称与应用图标都直接取自 FallGuard 应用本身。
/// Root view with branded sidebar navigation.
///
/// Uses a manual split layout (instead of ``NavigationSplitView``)
/// for macOS 11 compatibility.  The sidebar matches the old MD3 design:
/// - Brand logo + title at top
/// - Navigation items with icons
/// - Profile + settings at bottom
/// - System status indicator
struct ContentView: View {
    @EnvironmentObject var store: AppStore
    @EnvironmentObject var themeManager: ThemeManager
    @Environment(\.colorScheme) private var colorScheme
    @State private var selectedTab: Tab = .dashboard

    private var scheme: ColorScheme {
        themeManager.resolve(osScheme: colorScheme)
    }

    enum Tab: String, CaseIterable {
        case dashboard, events, importMedia, profiles

        var label: LocalizedStringKey {
            switch self {
            case .dashboard:   return "tab.dashboard"
            case .events:      return "tab.events"
            case .importMedia: return "tab.import_media"
            case .profiles:    return "tab.profiles"
            }
        }

        var icon: String {
            switch self {
            case .dashboard:   return "video.fill"
            case .events:      return "list.bullet.rectangle"
            case .importMedia: return "square.and.arrow.down"
            case .profiles:    return "person.2.fill"
            }
        }
    }

    var body: some View {
        GeometryReader { geometry in
            HSplitView {
                // Sidebar
                sidebar
                    .frame(minWidth: 200, idealWidth: 220, maxWidth: 260)
                    .frame(height: geometry.size.height, alignment: .top)

                // Content
                VStack(spacing: 0) {
                    // Toolbar — transparent so the shared ambient gradient below
                    // shows straight through.  This keeps the top strip the exact
                    // same colour as the page content beneath it (no grey seam).
                    toolbarContent
                        .padding(.horizontal, FallGuardSpacing.s16)
                        .padding(.vertical, FallGuardSpacing.s8)

                    GlassDivider()

                    // Page content
                    pageContent
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(maxWidth: .infinity, alignment: .top)
                .frame(height: geometry.size.height, alignment: .top)
                .background(FallGuardBackground(scheme: scheme))
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .top)
            .background(FallGuardBackground(scheme: scheme))
        }
        .background(FallGuardBackground(scheme: scheme))
        .onReceive(NotificationCenter.default.publisher(for: .fallGuardNavigateToImport)) { _ in
            selectedTab = .importMedia
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            // Brand
            HStack(spacing: FallGuardSpacing.s12) {
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 44, height: 44)
                    .shadow(color: FallGuardColors.green.opacity(0.25), radius: 8, y: 4)

                Text("FallGuard")
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundColor(FallGuardColors.textPrimary(for: scheme))
            }
            .padding(.top, FallGuardSpacing.s40)  // room for traffic lights
            .padding(.bottom, FallGuardSpacing.s16)
            .padding(.horizontal, FallGuardSpacing.s16)

            GlassDivider()
                .padding(.horizontal, FallGuardSpacing.s12)

            // Navigation
            VStack(spacing: FallGuardSpacing.s4) {
                ForEach(Tab.allCases, id: \.self) { tab in
                    SidebarNavItem(
                        tab: tab,
                        isSelected: selectedTab == tab,
                        scheme: scheme,
                        action: { selectedTab = tab }
                    )
                }
            }
            .padding(.vertical, FallGuardSpacing.s12)
            .padding(.horizontal, FallGuardSpacing.s12)

            Spacer()

            // Bottom section
            VStack(spacing: FallGuardSpacing.s8) {
                GlassDivider().padding(.horizontal, FallGuardSpacing.s4)

                // Profile pill
                Button {
                    selectedTab = .profiles
                } label: {
                    HStack(spacing: FallGuardSpacing.s10) {
                        Image(systemName: "person.circle.fill")
                            .font(.title3)
                            .foregroundColor(FallGuardColors.muted(for: scheme))
                        Text(store.activeProfile?.name ?? NSLocalizedString("profile.default", comment: ""))
                            .font(FallGuardFont.body)
                            .fontWeight(.medium)
                            .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                            .lineLimit(1)
                        Spacer()
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                            .foregroundColor(FallGuardColors.muted(for: scheme))
                    }
                    .padding(.horizontal, FallGuardSpacing.s14)
                    .padding(.vertical, FallGuardSpacing.s10)
                    .liquidGlass(cornerRadius: FallGuardRadius.md, interactive: true)
                    .overlay(
                        RoundedRectangle(cornerRadius: FallGuardRadius.md)
                            .stroke(FallGuardColors.greenDark.opacity(0.1), lineWidth: 0.5)
                            .allowsHitTesting(false)
                    )
                    .contentShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                }
                .buttonStyle(.plain)

                settingsControl

                // Status indicator
                HStack(spacing: 6) {
                    Circle()
                        .fill(connectionColor)
                        .frame(width: 8, height: 8)
                    Text(connectionText)
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.textSecondary(for: scheme))
                    Spacer()
                }
                .padding(.horizontal, FallGuardSpacing.s14)
                .padding(.top, FallGuardSpacing.s4)
            }
            .padding(.horizontal, FallGuardSpacing.s12)
            .padding(.bottom, FallGuardSpacing.s16)
        }
        .frame(minWidth: 200)
        .glassSidebar()
        .overlay(
            // Subtle tint so text remains readable on glass
            FallGuardColors.sidebarTint(for: scheme)
                .opacity(scheme == .dark ? 0.34 : 0.28)
                .allowsHitTesting(false)
        )
        .overlay(alignment: .trailing) {
            // Restore the visual boundary between the sidebar and content.
            Rectangle()
                .fill(FallGuardColors.line(for: scheme))
                .frame(width: 1)
                .allowsHitTesting(false)
        }
    }

    // MARK: Toolbar

    @ViewBuilder
    private var settingsControl: some View {
        if #available(macOS 14.0, *) {
            SettingsLink { settingsLabel }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity)
        } else {
            Button(action: openSettings) { settingsLabel }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity)
        }
    }

    private var settingsLabel: some View {
        HStack(spacing: FallGuardSpacing.s10) {
            Image(systemName: "gearshape")
                .font(.callout)
                .frame(width: 20)
            Text("settings.title")
                .font(FallGuardFont.callout)
                .fontWeight(.medium)
            Spacer()
        }
        .padding(.horizontal, FallGuardSpacing.s14)
        .padding(.vertical, FallGuardSpacing.s10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .foregroundColor(FallGuardColors.textSecondary(for: scheme))
        .background(FallGuardColors.glassTint(for: scheme).opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.lg))
        .contentShape(RoundedRectangle(cornerRadius: FallGuardRadius.lg))
    }

    private var toolbarContent: some View {
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(connectionColor)
                    .frame(width: 8, height: 8)
                Text(connectionText)
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            }
            .padding(.horizontal, FallGuardSpacing.s14)
            .padding(.vertical, 7)
            .liquidGlass(cornerRadius: FallGuardRadius.full, interactive: true)
            .clipShape(Capsule())

            Spacer()

            HStack(spacing: FallGuardSpacing.s8) {
                if store.isMonitoring {
                    Button {
                        Task { await store.stopMonitoring() }
                    } label: {
                        Label(NSLocalizedString("button.stop", comment: ""),
                              systemImage: "stop.fill")
                            .font(FallGuardFont.callout)
                            .padding(.horizontal, FallGuardSpacing.s14)
                            .padding(.vertical, 7)
                    }
                    .buttonStyle(FallGuardDangerButtonStyle(scheme: scheme))
                } else if !store.isImporting {
                    Button {
                        Task { await store.startMonitoring() }
                    } label: {
                        Label(NSLocalizedString("button.start", comment: ""),
                              systemImage: "play.fill")
                            .font(.system(size: 14, weight: .semibold))
                            .padding(.horizontal, FallGuardSpacing.s14)
                            .padding(.vertical, 7)
                    }
                    .buttonStyle(FallGuardButtonStyle(scheme: scheme))
                    .disabled(!store.serviceManager.state.isReady)
                }
            }
        }
    }

    // MARK: Page Content

    @ViewBuilder
    private var pageContent: some View {
        switch selectedTab {
        case .dashboard:
            DashboardView()
                .environmentObject(themeManager)
        case .events:
            EventsView()
        case .importMedia:
            ImportMediaView()
        case .profiles:
            ProfilesView()
        }
    }

    // MARK: Helpers

    private func openSettings() {
        if #available(macOS 14, *) {
            NSApplication.shared.sendAction(
                Selector(("showSettingsWindow:")), to: nil, from: nil)
        } else {
            NSApplication.shared.sendAction(
                Selector(("showPreferencesWindow:")), to: nil, from: nil)
        }
    }

    private var connectionColor: Color {
        if store.connectionError != nil { return FallGuardColors.red }
        if store.serviceManager.state.isReady { return FallGuardColors.green }
        return FallGuardColors.amber
    }

    private var connectionText: String {
        if let err = store.connectionError { return err }
        return store.serviceManager.state.displayText
    }

    private var connectionBg: Color {
        if store.connectionError != nil { return FallGuardColors.redLight }
        if store.serviceManager.state.isReady { return FallGuardColors.greenLight }
        return FallGuardColors.amberLight
    }
}

extension Notification.Name {
    static let fallGuardNavigateToImport = Notification.Name("FallGuardNavigateToImport")
}

// MARK: - Sidebar Nav Item

struct SidebarNavItem: View {
    let tab: ContentView.Tab
    let isSelected: Bool
    let scheme: ColorScheme
    let action: () -> Void

    var body: some View {
        Button {
            action()
        } label: {
            HStack(spacing: FallGuardSpacing.s12) {
                Image(systemName: tab.icon)
                    .font(.callout)
                    .frame(width: 20)
                Text(tab.label)
                    .font(FallGuardFont.callout)
                    .fontWeight(.medium)
                    .lineLimit(1)
                    .minimumScaleFactor(0.84)
                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s14)
            .padding(.vertical, FallGuardSpacing.s10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                isSelected
                    ? FallGuardColors.navActiveBg(for: scheme).opacity(scheme == .dark ? 0.78 : 0.88)
                    : Color.clear
            )
            .overlay(alignment: .leading) {
                if isSelected {
                    Capsule()
                        .fill(FallGuardColors.primary(for: scheme))
                        .frame(width: 3, height: 22)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            .foregroundColor(isSelected
                ? FallGuardColors.primary(for: scheme)
                : FallGuardColors.textSecondary(for: scheme))
            .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.lg))
            .contentShape(RoundedRectangle(cornerRadius: FallGuardRadius.lg))
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity)
    }
}
