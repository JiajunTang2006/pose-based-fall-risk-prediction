import SwiftUI
import AppKit

// 文案修改位置：Resources/*/Localizable.strings 中的 Dashboard、Dashboard Cards、
// Risk Levels、Status、Metrics、States 分组；布局代码无需修改。
// MARK: - Dashboard View

/// Main monitoring dashboard with 6-card layout inspired by MD3.
///
/// Layout (monitoring state):
/// ```
/// ┌──────────────────────┐ ┌──────────┐ ┌──────────┐
/// │   Monitor Card        │ │ RiskRing │ │ Status   │
/// │   (video preview)    │ │  (gauge) │ │  Card    │
/// └──────────────────────┘ └──────────┘ └──────────┘
/// ┌──────────────┐ ┌──────────┐
/// │ Risk Trend   │ │ Events   │
/// │ Chart        │ │ MiniList │
/// └──────────────┘ └──────────┘
/// ┌──────────────────────────────────────────────┐
/// │              Metrics Bar                      │
/// └──────────────────────────────────────────────┘
/// ```
struct DashboardView: View {
    @EnvironmentObject var store: AppStore
    @EnvironmentObject var themeManager: ThemeManager
    @Environment(\.colorScheme) private var colorScheme

    private var scheme: ColorScheme {
        themeManager.resolve(osScheme: colorScheme)
    }

    var body: some View {
        ZStack {
            switch store.dashboardState {
            case .launchingService:
                launchingView
            case .serviceFailed(let msg):
                errorView(message: msg, canRetry: true)
            case .serviceReadyIdle:
                idleView
            case .requestingCamera:
                loadingView(message: NSLocalizedString("dashboard.loading.camera", comment: ""))
            case .monitoringNormal, .monitoringPreFall, .monitoringFall, .personUnknown:
                monitoringGridView
            case .stopping:
                loadingView(message: NSLocalizedString("dashboard.stopping", comment: ""))
            case .importingMedia:
                importingView
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Launching

    private var launchingView: some View {
        VStack(spacing: FallGuardSpacing.s20) {
            ProgressView()
                .scaleEffect(1.5)
            Text("dashboard.launching")
                .font(FallGuardFont.title2)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Error

    private func errorView(message: String, canRetry: Bool) -> some View {
        VStack(spacing: FallGuardSpacing.s24) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(FallGuardColors.red)
            Text("dashboard.error.title")
                .font(FallGuardFont.title2)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
            Text(message)
                .font(FallGuardFont.body)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
            if canRetry {
                Button(action: { Task { await store.bootstrap() } }) {
                    Label("dashboard.retry", systemImage: "arrow.clockwise")
                }
                .buttonStyle(FallGuardButtonStyle(scheme: scheme))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Idle

    private var idleView: some View {
        VStack {
            Spacer(minLength: FallGuardSpacing.s24)

            FallGuardCard(scheme: scheme) {
                VStack(spacing: FallGuardSpacing.s20) {
                    HStack(spacing: FallGuardSpacing.s32) {
                        ZStack {
                            Circle()
                                .fill(FallGuardColors.primary(for: scheme).opacity(0.07))
                                .frame(width: 170, height: 170)
                            Circle()
                                .stroke(FallGuardColors.primary(for: scheme).opacity(0.14), lineWidth: 1)
                                .frame(width: 142, height: 142)
                            RoundedRectangle(cornerRadius: 28, style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            FallGuardColors.primary(for: scheme),
                                            Color(hex: "#15803D")
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 94, height: 94)
                                .shadow(color: FallGuardColors.green.opacity(0.3), radius: 18, y: 10)
                            Image(systemName: "shield.checkered")
                                .font(.system(size: 44, weight: .semibold))
                                .foregroundColor(.white)
                        }
                        .allowsHitTesting(false)

                        VStack(alignment: .leading, spacing: FallGuardSpacing.s16) {
                            Text("dashboard.ready")
                                .font(.system(size: 32, weight: .bold, design: .rounded))
                                .foregroundColor(FallGuardColors.textPrimary(for: scheme))

                            Button(action: { Task { await store.startMonitoring() } }) {
                                Label("dashboard.start_monitoring", systemImage: "play.fill")
                                    .font(FallGuardFont.headline)
                                    .padding(.horizontal, FallGuardSpacing.s24)
                                    .padding(.vertical, FallGuardSpacing.s12)
                            }
                            .buttonStyle(FallGuardButtonStyle(scheme: scheme))
                            .controlSize(.large)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(FallGuardSpacing.s32)
            }
            .frame(maxWidth: 780)
            .padding(.horizontal, FallGuardSpacing.s32)

            Spacer(minLength: FallGuardSpacing.s24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Loading

    private func loadingView(message: String) -> some View {
        VStack(spacing: FallGuardSpacing.s20) {
            ProgressView()
                .scaleEffect(1.5)
            Text(message)
                .font(FallGuardFont.title2)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Monitoring Grid

    private var monitoringGridView: some View {
        ScrollView {
            VStack(spacing: FallGuardSpacing.s16) {
                // Top row: Monitor | RiskRing | Status
                HStack(alignment: .top, spacing: FallGuardSpacing.s16) {
                    MonitorCard(scheme: scheme)
                        .layoutPriority(2)

                    VStack(spacing: FallGuardSpacing.s16) {
                        RiskRingView(scheme: scheme)
                        StatusCardView(scheme: scheme)
                    }
                    .frame(width: 280)
                }

                // Middle row: RiskTrend | EventsMini
                HStack(alignment: .top, spacing: FallGuardSpacing.s16) {
                    RiskTrendChart(scheme: scheme)
                        .layoutPriority(1)
                    EventsMiniList(scheme: scheme)
                        .frame(width: 280)
                }

                // Bottom row: Metrics
                MetricsBar(scheme: scheme)
            }
            .padding(FallGuardSpacing.s16)
        }
        .background(FallGuardBackground(scheme: scheme))
    }

    // MARK: - Importing

    private var importingView: some View {
        VStack(spacing: FallGuardSpacing.s20) {
            ProgressView()
                .scaleEffect(1.5)
            Text("dashboard.importing")
                .font(FallGuardFont.title2)
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
            if let job = store.importJob {
                VStack(spacing: FallGuardSpacing.s8) {
                    ProgressView(value: job.progress)
                        .frame(width: 300)
                    Text("Frame \(job.currentFrame) / \(job.totalFrames)")
                        .font(FallGuardFont.caption)
                        .foregroundColor(FallGuardColors.textSecondary(for: scheme))
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(FallGuardBackground(scheme: scheme))
    }
}

// MARK: - Card Container

/// Reusable card with frosted-glass MD3 styling.
///
/// When ``glass`` is `true` (default), the card background uses
/// `.regularMaterial` to blur content behind it, creating the
/// classic macOS layered-depth look.
struct FallGuardCard<Content: View>: View {
    let scheme: ColorScheme
    var glass: Bool = true
    @ViewBuilder let content: () -> Content

    var body: some View {
        content()
            .background(
                Group {
                    if glass {
                        ZStack {
                            RoundedRectangle(cornerRadius: FallGuardRadius.xl, style: .continuous)
                                .fill(.regularMaterial)
                            RoundedRectangle(cornerRadius: FallGuardRadius.xl, style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            Color.white.opacity(scheme == .dark ? 0.03 : 0.34),
                                            FallGuardColors.glassTint(for: scheme)
                                                .opacity(scheme == .dark ? 0.12 : 0.2)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        }
                    } else {
                        RoundedRectangle(cornerRadius: FallGuardRadius.xl)
                            .fill(FallGuardColors.surface(for: scheme))
                    }
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.xl))
            .overlay(
                RoundedRectangle(cornerRadius: FallGuardRadius.xl)
                    .stroke(
                        glass
                            ? (scheme == .dark
                                ? FallGuardColors.green.opacity(0.2)
                                : FallGuardColors.greenDark.opacity(0.14))
                            : FallGuardColors.line(for: scheme),
                        lineWidth: 0.5
                    )
                    .allowsHitTesting(false)
            )
            .fallGuardCardShadow(scheme: scheme)
    }
}

// MARK: - Monitor Card

struct MonitorCard: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        FallGuardCard(scheme: scheme) {
            VStack(spacing: 0) {
                // Header
                HStack {
                    Text("dashboard.live_monitor")
                        .font(FallGuardFont.headline)
                        .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    Spacer()
                    SensitivityBadge(scheme: scheme)
                }
                .padding(.horizontal, FallGuardSpacing.s16)
                .padding(.vertical, FallGuardSpacing.s12)

                // Video Preview
                videoShell
                    .padding(.horizontal, FallGuardSpacing.s12)
                    .padding(.bottom, FallGuardSpacing.s12)

                // Action buttons
                HStack(spacing: FallGuardSpacing.s12) {
                    if store.isMonitoring {
                        Button(action: { Task { await store.stopMonitoring() } }) {
                            Label(NSLocalizedString("button.stop", comment: ""),
                                  systemImage: "stop.fill")
                                .font(FallGuardFont.callout)
                                .padding(.horizontal, FallGuardSpacing.s20)
                                .padding(.vertical, FallGuardSpacing.s8)
                        }
                        .buttonStyle(FallGuardDangerButtonStyle(scheme: scheme))
                    } else {
                        Button(action: { Task { await store.startMonitoring() } }) {
                            Label(NSLocalizedString("button.start", comment: ""),
                                  systemImage: "play.fill")
                                .font(FallGuardFont.callout)
                                .padding(.horizontal, FallGuardSpacing.s20)
                                .padding(.vertical, FallGuardSpacing.s8)
                        }
                        .buttonStyle(FallGuardButtonStyle(scheme: scheme))
                    }
                    Spacer()
                    if !store.isMonitoring {
                        Button(action: {
                            NotificationCenter.default.post(
                                name: .fallGuardNavigateToImport,
                                object: nil
                            )
                        }) {
                            Label(NSLocalizedString("tab.import_media", comment: ""),
                                  systemImage: "square.and.arrow.down")
                                .font(FallGuardFont.callout)
                                .padding(.horizontal, FallGuardSpacing.s20)
                                .padding(.vertical, FallGuardSpacing.s8)
                        }
                        .buttonStyle(FallGuardSecondaryButtonStyle(scheme: scheme))
                    }
                }
                .padding(.horizontal, FallGuardSpacing.s16)
                .padding(.bottom, FallGuardSpacing.s12)
            }
        }
    }

    // MARK: Video Shell

    private var videoShell: some View {
        ZStack {
            // Dark background
            RoundedRectangle(cornerRadius: FallGuardRadius.lg)
                .fill(FallGuardColors.videoBg)

            // Preview image or placeholder
            if let img = store.previewImage {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.lg))
            } else {
                VStack(spacing: FallGuardSpacing.s8) {
                    RoundedRectangle(cornerRadius: FallGuardRadius.sm)
                        .stroke(FallGuardColors.videoText.opacity(0.3), lineWidth: 2)
                        .frame(width: 48, height: 36)
                        .overlay(
                            Image(systemName: "play.fill")
                                .font(.caption)
                                .foregroundColor(FallGuardColors.videoText.opacity(0.3))
                        )
                    Text("dashboard.placeholder")
                        .font(FallGuardFont.caption)
                        .foregroundColor(FallGuardColors.videoText)
                    Text("dashboard.placeholder_sub")
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
            }

            // Overlays
            VStack {
                HStack {
                    // REC badge
                    if store.isMonitoring {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(FallGuardColors.red)
                                .frame(width: 8, height: 8)
                                .modifier(PulseAnimation())
                            Text("REC")
                                .font(.system(size: 11, weight: .bold))
                            if let start = store.monitoringStartTime {
                                Text(elapsedTime(from: start))
                                    .font(.system(size: 12, weight: .medium))
                            }
                        }
                        .padding(.horizontal, FallGuardSpacing.s12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.sm))
                        .padding(FallGuardSpacing.s12)
                    }
                    Spacer()
                }
                Spacer()
                HStack {
                    // FPS
                    FpsChip()
                    Spacer()
                }
                .padding(.bottom, FallGuardSpacing.s4)
            }
        }
        .frame(minHeight: 320)
        .aspectRatio(4/3, contentMode: .fit)
    }

    private func elapsedTime(from start: Date) -> String {
        let interval = Int(Date().timeIntervalSince(start))
        let h = interval / 3600
        let m = (interval % 3600) / 60
        let s = interval % 60
        if h > 0 {
            return String(format: "%d:%02d:%02d", h, m, s)
        }
        return String(format: "%02d:%02d", m, s)
    }
}

// MARK: - FPS Chip

struct FpsChip: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        HStack(spacing: 4) {
            Text("FPS")
                .font(.system(size: 10, weight: .semibold))
            Text(String(format: "%.0f", store.fps))
                .font(.system(size: 12, weight: .bold, design: .rounded))
        }
        .padding(.horizontal, FallGuardSpacing.s8)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.sm))
        .padding(.leading, FallGuardSpacing.s12)
    }
}

// MARK: - Pulse Animation

struct PulseAnimation: ViewModifier {
    @State private var opacity: Double = 1.0

    func body(content: Content) -> some View {
        content
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever()) {
                    opacity = 0.3
                }
            }
    }
}

// MARK: - Sensitivity Badge

struct SensitivityBadge: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        if let s = store.settings?.sensitivity {
            Text(NSLocalizedString("sensitivity.\(s)", comment: ""))
                .font(FallGuardFont.caption2)
                .fontWeight(.semibold)
                .padding(.horizontal, FallGuardSpacing.s8)
                .padding(.vertical, 3)
                .background(FallGuardColors.primary(for: scheme).opacity(0.1))
                .foregroundColor(FallGuardColors.primary(for: scheme))
                .clipShape(Capsule())
        }
    }
}

// MARK: - Risk Ring View

struct RiskRingView: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    private var riskPercent: Double { Double(store.riskPercent) / 100.0 }
    private var riskColor: Color {
        if store.riskPercent < RiskDisplayThresholds.warningPercent {
            return FallGuardColors.green
        }
        if store.riskPercent < RiskDisplayThresholds.dangerPercent {
            return FallGuardColors.amber
        }
        return FallGuardColors.red
    }

    private var riskLevelText: String {
        if store.riskPercent < RiskDisplayThresholds.warningPercent { return "risk.low" }
        if store.riskPercent < RiskDisplayThresholds.dangerPercent { return "risk.medium" }
        return "risk.high"
    }

    var body: some View {
        FallGuardCard(scheme: scheme) {
            VStack(spacing: FallGuardSpacing.s8) {
                Text("dashboard.risk_prediction")
                    .font(FallGuardFont.headline)
                    .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, FallGuardSpacing.s16)
                    .padding(.top, FallGuardSpacing.s12)

                // Risk Ring
                ZStack {
                    // Track
                    Circle()
                        .stroke(FallGuardColors.line(for: scheme), lineWidth: 12)
                        .frame(width: 130, height: 130)

                    // Colored arc
                    Circle()
                        .trim(from: 0, to: riskPercent)
                        .stroke(riskColor, style: StrokeStyle(lineWidth: 12, lineCap: .round))
                        .frame(width: 130, height: 130)
                        .rotationEffect(.degrees(-90))
                        .animation(.easeInOut(duration: FallGuardAnimation.normal), value: riskPercent)

                    // Center text
                    VStack(spacing: 2) {
                        Text("\(store.riskPercent)%")
                            .font(FallGuardFont.hero)
                            .foregroundColor(riskColor)
                            .animation(.easeInOut(duration: FallGuardAnimation.fast), value: store.riskPercent)
                        Text(LocalizedStringKey(riskLevelText))
                            .font(FallGuardFont.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(FallGuardColors.textSecondary(for: scheme))
                    }
                }
                .padding(.vertical, FallGuardSpacing.s4)

                // Risk level text
                Text(store.modelState.displayName)
                    .font(FallGuardFont.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, FallGuardSpacing.s12)
                    .padding(.vertical, 4)
                    .background(riskColor.opacity(0.12))
                    .foregroundColor(riskColor)
                    .clipShape(Capsule())
                    .padding(.bottom, FallGuardSpacing.s12)
            }
        }
    }
}

// MARK: - Status Card

struct StatusCardView: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    private var heroColor: Color {
        if !store.isMonitoring { return FallGuardColors.muted(for: scheme) }
        switch store.businessState {
        case .safe: return FallGuardColors.green
        case .warning: return FallGuardColors.amber
        case .danger: return FallGuardColors.red
        case .unknown: return FallGuardColors.muted(for: scheme)
        }
    }

    private var heroBg: Color {
        if !store.isMonitoring { return FallGuardColors.line(for: scheme) }
        switch store.businessState {
        case .safe: return FallGuardColors.greenLight
        case .warning: return FallGuardColors.amberLight
        case .danger: return FallGuardColors.redLight
        case .unknown: return FallGuardColors.line(for: scheme)
        }
    }

    var body: some View {
        FallGuardCard(scheme: scheme) {
            VStack(spacing: FallGuardSpacing.s12) {
                Text("dashboard.system_status")
                    .font(FallGuardFont.headline)
                    .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, FallGuardSpacing.s16)
                    .padding(.top, FallGuardSpacing.s12)

                // Hero banner
                HStack(spacing: FallGuardSpacing.s12) {
                    Image(systemName: heroIcon)
                        .font(.title2)
                        .foregroundColor(heroColor)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(LocalizedStringKey(heroTitle))
                            .font(FallGuardFont.callout)
                            .fontWeight(.bold)
                            .foregroundColor(heroColor)
                        Text(heroDetail)
                            .font(FallGuardFont.caption)
                            .foregroundColor(FallGuardColors.textSecondary(for: scheme))
                    }
                    Spacer()
                }
                .padding(FallGuardSpacing.s12)
                .background(heroBg.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
                .padding(.horizontal, FallGuardSpacing.s12)

                // Status rows
                VStack(spacing: 0) {
                    StatusRow(icon: "web.camera", label: "status.camera",
                              value: store.isMonitoring ? "status.connected" : "status.idle",
                              ok: store.isMonitoring, scheme: scheme)
                    Divider().padding(.leading, 36)
                    StatusRow(icon: "cpu", label: "status.model",
                              value: store.isLoading ? "status.loading" : "status.active",
                              ok: !store.isLoading, scheme: scheme)
                    Divider().padding(.leading, 36)
                    StatusRow(icon: "eye", label: "status.confidence",
                              value: "\(store.confidencePercent)%",
                              ok: store.personVisible && store.confidencePercent >= 45,
                              scheme: scheme)
                    Divider().padding(.leading, 36)
                    StatusRow(icon: "antenna.radiowaves.left.and.right", label: "status.environment",
                              value: store.isMonitoring ? "status.monitoring" : "status.standby",
                              ok: true, scheme: scheme)
                }
                .padding(.horizontal, FallGuardSpacing.s12)
                .padding(.bottom, FallGuardSpacing.s12)
            }
        }
    }

    private var heroIcon: String {
        if !store.isMonitoring { return "pause.circle" }
        switch store.businessState {
        case .safe: return "shield.checkered"
        case .warning: return "exclamationmark.shield"
        case .danger: return "exclamationmark.triangle.fill"
        case .unknown: return "questionmark.shield"
        }
    }

    private var heroTitle: String {
        if !store.isMonitoring { return "status.unknown" }
        switch store.businessState {
        case .safe: return "status.safe"
        case .warning: return "status.warning"
        case .danger: return "status.danger"
        case .unknown: return "status.unknown"
        }
    }

    private var heroDetail: String {
        if !store.isMonitoring { return "status.idle_detail" }
        return store.modelState.displayName
    }
}

// MARK: - Status Row

struct StatusRow: View {
    let icon: String
    let label: LocalizedStringKey
    let value: String
    let ok: Bool
    let scheme: ColorScheme

    var body: some View {
        HStack(spacing: FallGuardSpacing.s10) {
            Image(systemName: icon)
                .font(.callout)
                .foregroundColor(FallGuardColors.muted(for: scheme))
                .frame(width: 20)
            Text(label)
                .font(FallGuardFont.body)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            Spacer()
            HStack(spacing: 4) {
                Circle()
                    .fill(ok ? FallGuardColors.green : FallGuardColors.amber)
                    .frame(width: 6, height: 6)
                Text(LocalizedStringKey(value))
                    .font(FallGuardFont.caption)
                    .foregroundColor(ok ? FallGuardColors.greenDark : FallGuardColors.amberDark)
            }
        }
        .padding(.vertical, FallGuardSpacing.s8)
    }
}

// MARK: - Risk Trend Chart (Canvas)

struct RiskTrendChart: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    private var riskColor: Color {
        switch store.businessState {
        case .safe: return FallGuardColors.green
        case .warning: return FallGuardColors.amber
        case .danger: return FallGuardColors.red
        case .unknown: return FallGuardColors.muted(for: scheme)
        }
    }

    var body: some View {
        FallGuardCard(scheme: scheme) {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s4) {
                HStack {
                    Text("dashboard.risk_trend")
                        .font(FallGuardFont.headline)
                        .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    Spacer()
                    Text("dashboard.last_60s")
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
                .padding(.horizontal, FallGuardSpacing.s16)
                .padding(.top, FallGuardSpacing.s12)

                if store.riskHistory.isEmpty {
                    Text("dashboard.no_data")
                        .font(FallGuardFont.caption)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 30)
                } else {
                    Canvas { context, size in
                        let values = store.riskHistory
                        guard values.count > 1 else { return }

                        let chartW = size.width - 32
                        let chartH = size.height - 24
                        let xStep = chartW / CGFloat(max(values.count - 1, 1))
                        let yMax = chartH / 100.0  // 0–100 scale

                        // Grid lines
                        for i in 0...4 {
                            let y = 12 + chartH * CGFloat(i) / 4
                            var path = Path()
                            path.move(to: CGPoint(x: 16, y: y))
                            path.addLine(to: CGPoint(x: size.width - 16, y: y))
                            context.stroke(path,
                                with: .color(FallGuardColors.line(for: scheme)),
                                lineWidth: 0.5)
                        }

                        // Build points
                        let points = values.enumerated().map { i, v in
                            CGPoint(
                                x: 16 + CGFloat(i) * xStep,
                                y: 12 + chartH - CGFloat(v) * yMax
                            )
                        }

                        // Fill area
                        var fillPath = Path()
                        fillPath.move(to: CGPoint(x: 16, y: 12 + chartH))
                        for pt in points {
                            fillPath.addLine(to: pt)
                        }
                        fillPath.addLine(to: CGPoint(x: points.last!.x, y: 12 + chartH))
                        fillPath.closeSubpath()
                        context.fill(fillPath,
                            with: .color(riskColor.opacity(0.14)))

                        // Line
                        var linePath = Path()
                        linePath.move(to: points[0])
                        for pt in points.dropFirst() {
                            linePath.addLine(to: pt)
                        }
                        context.stroke(linePath,
                            with: .color(riskColor),
                            style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))

                        // End dot
                        if let last = points.last {
                            context.fill(
                                Path(ellipseIn: CGRect(x: last.x - 3, y: last.y - 3, width: 6, height: 6)),
                                with: .color(riskColor))
                        }

                        // Labels
                        context.draw(
                            Text("-60s")
                                .font(.system(size: 9))
                                .foregroundColor(FallGuardColors.muted(for: scheme)),
                            at: CGPoint(x: 16, y: size.height - 10))
                        context.draw(
                            Text("Now")
                                .font(.system(size: 9))
                                .foregroundColor(FallGuardColors.muted(for: scheme)),
                            at: CGPoint(x: size.width - 40, y: size.height - 10))
                    }
                    .frame(height: 150)
                    .padding(.horizontal, FallGuardSpacing.s8)
                }
            }
            .padding(.bottom, FallGuardSpacing.s12)
        }
    }
}

// MARK: - Events Mini List

struct EventsMiniList: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        FallGuardCard(scheme: scheme) {
            VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
                HStack {
                    Text("dashboard.recent_events")
                        .font(FallGuardFont.headline)
                        .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    Spacer()
                }
                .padding(.horizontal, FallGuardSpacing.s16)
                .padding(.top, FallGuardSpacing.s12)

                if store.recentEvents.isEmpty {
                    Text("dashboard.no_events")
                        .font(FallGuardFont.caption)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 20)
                } else {
                    VStack(spacing: 0) {
                        ForEach(Array(store.recentEvents.prefix(6).enumerated()), id: \.element.id) { i, event in
                            if i > 0 {
                                Divider().padding(.leading, 32)
                            }
                            MiniEventRow(event: event, scheme: scheme)
                        }
                    }
                }
            }
            .padding(.bottom, FallGuardSpacing.s12)
        }
    }
}

struct MiniEventRow: View {
    let event: EventDTO
    let scheme: ColorScheme

    private var dotColor: Color {
        event.eventType == "fall" ? FallGuardColors.red : FallGuardColors.amber
    }

    var body: some View {
        HStack(spacing: FallGuardSpacing.s8) {
            Circle()
                .fill(dotColor)
                .frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 2) {
                Text(event.eventType == "fall"
                     ? NSLocalizedString("event.type.fall", comment: "")
                     : NSLocalizedString("event.type.prefall", comment: ""))
                    .font(FallGuardFont.body)
                    .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                Text(event.startedAt)
                    .font(FallGuardFont.caption2)
                    .foregroundColor(FallGuardColors.muted(for: scheme))
            }
            Spacer()
            Text("\(Int(round(event.peakRisk * 100)))%")
                .font(FallGuardFont.caption)
                .fontWeight(.bold)
                .foregroundColor(dotColor)
        }
        .padding(.horizontal, FallGuardSpacing.s16)
        .padding(.vertical, FallGuardSpacing.s12)
    }
}

// MARK: - Metrics Bar

struct MetricsBar: View {
    @EnvironmentObject var store: AppStore
    let scheme: ColorScheme

    var body: some View {
        FallGuardCard(scheme: scheme) {
            HStack(spacing: 0) {
                MetricItem(
                    icon: "clock",
                    label: "metric.monitor_time",
                    value: monitoringElapsed,
                    scheme: scheme
                )
                Divider().frame(height: 40)
                MetricItem(
                    icon: "bell.badge",
                    label: "metric.total_alerts",
                    value: "\(store.totalAlerts)",
                    scheme: scheme
                )
                Divider().frame(height: 40)
                MetricItem(
                    icon: "exclamationmark.triangle",
                    label: "metric.high_risk",
                    value: "\(store.highRiskEvents)",
                    scheme: scheme
                )
                Divider().frame(height: 40)
                MetricItem(
                    icon: "chart.bar",
                    label: "metric.avg_risk",
                    value: String(format: "%.1f%%", store.riskScore * 100),
                    scheme: scheme
                )
                Divider().frame(height: 40)
                MetricItem(
                    icon: "info.circle",
                    label: "metric.version",
                    value: "0.3.3",
                    scheme: scheme
                )
            }
            .padding(.vertical, FallGuardSpacing.s12)
        }
    }

    private var monitoringElapsed: String {
        guard let start = store.monitoringStartTime else { return "--" }
        let s = Int(Date().timeIntervalSince(start))
        let h = s / 3600, m = (s % 3600) / 60
        return h > 0 ? "\(h)h \(m)m" : "\(m)m"
    }
}

struct MetricItem: View {
    let icon: String
    let label: LocalizedStringKey
    let value: String
    let scheme: ColorScheme

    var body: some View {
        VStack(spacing: FallGuardSpacing.s4) {
            Image(systemName: icon)
                .font(.callout)
                .foregroundColor(FallGuardColors.muted(for: scheme))
            Text(value)
                .font(.system(.callout, design: .rounded).bold())
                .foregroundColor(FallGuardColors.textPrimary(for: scheme))
            Text(label)
                .font(FallGuardFont.caption2)
                .foregroundColor(FallGuardColors.muted(for: scheme))
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Button Styles

struct FallGuardButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    let scheme: ColorScheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white)
            .background(
                LinearGradient(
                    colors: [
                        FallGuardColors.primary(for: scheme),
                        scheme == .dark ? Color(hex: "#16A34A") : Color(hex: "#15803D")
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
                    .stroke(Color.white.opacity(0.18), lineWidth: 0.5)
                    .allowsHitTesting(false)
            )
            .shadow(
                color: FallGuardColors.primary(for: scheme).opacity(isEnabled ? 0.24 : 0),
                radius: 8,
                y: 4
            )
            .opacity(isEnabled ? 1 : 0.42)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.easeInOut(duration: FallGuardAnimation.fast), value: configuration.isPressed)
    }
}

struct FallGuardDangerButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    let scheme: ColorScheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(FallGuardColors.red)
            .background(
                FallGuardColors.redLight.opacity(scheme == .dark ? 0.2 : 0.9),
                in: RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
                    .stroke(FallGuardColors.redBorder, lineWidth: 1)
                    .allowsHitTesting(false)
            )
            .opacity(isEnabled ? 1 : 0.42)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.easeInOut(duration: FallGuardAnimation.fast), value: configuration.isPressed)
    }
}

struct FallGuardSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    let scheme: ColorScheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(FallGuardColors.primary(for: scheme))
            .background(
                .thinMaterial,
                in: RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: FallGuardRadius.md, style: .continuous)
                    .stroke(FallGuardColors.primary(for: scheme).opacity(0.26), lineWidth: 0.8)
                    .allowsHitTesting(false)
            )
            .opacity(isEnabled ? 1 : 0.42)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.easeInOut(duration: FallGuardAnimation.fast), value: configuration.isPressed)
    }
}

// (no helpers needed currently)
