import SwiftUI

// 文案修改位置：Resources/*/Localizable.strings 中的 Events 分组；布局代码无需修改。
/// Events list with type/status filtering.
struct EventsView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.colorScheme) private var colorScheme
    @State private var searchText: String = ""
    @State private var filterType: EventFilter = .all
    @State private var filterStatus: StatusFilter = .all
    @State private var selectedEvent: EventDTO?

    enum EventFilter: String, CaseIterable {
        case all, fall, prefall
        var label: LocalizedStringKey {
            switch self {
            case .all: return "events.filter.all"
            case .fall: return "events.filter.fall"
            case .prefall: return "events.filter.prefall"
            }
        }
    }

    enum StatusFilter: String, CaseIterable {
        case all, open, resolved
        var label: LocalizedStringKey {
            switch self {
            case .all: return "events.filter.all_status"
            case .open: return "events.filter.open"
            case .resolved: return "events.filter.resolved"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("events.title")
                    .font(FallGuardFont.title)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                Spacer()
                Button(action: { Task { await store.loadRecentEvents() } }) {
                    Label("events.refresh", systemImage: "arrow.clockwise")
                        .font(.system(size: 14, weight: .semibold))
                        .padding(.horizontal, FallGuardSpacing.s14)
                        .padding(.vertical, 7)
                }
                .buttonStyle(FallGuardButtonStyle(scheme: colorScheme))
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.top, FallGuardSpacing.s20)
            .padding(.bottom, FallGuardSpacing.s12)

            // Filter bar — glass header
            HStack(spacing: FallGuardSpacing.s12) {
                Picker("", selection: $filterType) {
                    ForEach(EventFilter.allCases, id: \.self) { f in
                        Text(f.label).tag(f)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 260)

                Picker("", selection: $filterStatus) {
                    ForEach(StatusFilter.allCases, id: \.self) { s in
                        Text(s.label).tag(s)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)

                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.vertical, FallGuardSpacing.s10)
            .glassHeader()

            GlassDivider()
                .padding(.horizontal, FallGuardSpacing.s24)

            // List
            if store.recentEvents.isEmpty {
                Spacer()
                VStack(spacing: FallGuardSpacing.s12) {
                    Image(systemName: "tray")
                        .font(.system(size: 40))
                        .foregroundColor(FallGuardColors.muted(for: colorScheme))
                    Text("events.empty")
                        .font(FallGuardFont.body)
                        .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))
                }
                Spacer()
            } else {
                List {
                    ForEach(filteredEvents) { event in
                        Button {
                            selectedEvent = event
                        } label: {
                            EventListRow(event: event, scheme: colorScheme)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                            .contextMenu {
                                Button("events.copy_id") {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(event.id, forType: .string)
                                }
                            }
                    }
                }
                .listStyle(.plain)
            }

            // Status bar
            HStack {
                Text(String(format: NSLocalizedString("events.count", comment: ""),
                           filteredEvents.count))
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.muted(for: colorScheme))
                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.vertical, FallGuardSpacing.s8)
        }
        .background(FallGuardBackground(scheme: colorScheme))
        .onAppear { Task { await store.loadRecentEvents() } }
        .sheet(item: $selectedEvent) { event in
            EventDetailView(event: event, scheme: colorScheme)
                .environmentObject(store)
        }
    }

    private var filteredEvents: [EventDTO] {
        var events = store.recentEvents
        if !searchText.isEmpty {
            events = events.filter {
                $0.eventType.localizedCaseInsensitiveContains(searchText) ||
                $0.id.localizedCaseInsensitiveContains(searchText)
            }
        }
        switch filterType {
        case .fall: events = events.filter { $0.eventType == "fall" }
        case .prefall: events = events.filter { $0.eventType == "pre-fall" }
        case .all: break
        }
        switch filterStatus {
        case .open: events = events.filter { $0.status == "open" }
        case .resolved:
            events = events.filter {
                $0.status == "ended" || $0.status == "reviewed" ||
                $0.status == "resolved"
            }
        case .all: break
        }
        return events
    }
}

// MARK: - Event List Row

struct EventListRow: View {
    let event: EventDTO
    let scheme: ColorScheme

    private var dotColor: Color {
        event.eventType == "fall" ? FallGuardColors.red : FallGuardColors.amber
    }

    var body: some View {
        HStack(spacing: FallGuardSpacing.s12) {
            // Colored status bar
            RoundedRectangle(cornerRadius: 2)
                .fill(dotColor)
                .frame(width: 4, height: 40)

            // Icon
            Image(systemName: event.eventType == "fall"
                  ? "exclamationmark.triangle.fill"
                  : "exclamationmark.circle.fill")
                .font(.callout)
                .foregroundColor(dotColor)
                .frame(width: 28)

            // Info
            VStack(alignment: .leading, spacing: 3) {
                Text(event.eventType == "fall"
                     ? NSLocalizedString("event.type.fall", comment: "")
                     : NSLocalizedString("event.type.prefall", comment: ""))
                    .font(FallGuardFont.callout)
                    .fontWeight(.semibold)
                    .foregroundColor(FallGuardColors.textPrimary(for: scheme))

                HStack(spacing: FallGuardSpacing.s8) {
                    Text(EventFormatting.displayDate(event.startedAt))
                        .font(FallGuardFont.caption)
                    if let ended = event.endedAt {
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                        Text(EventFormatting.displayDate(ended))
                            .font(FallGuardFont.caption)
                    }
                    if event.videoClipPath != nil {
                        Image(systemName: "video.fill")
                            .font(.caption2)
                            .foregroundColor(FallGuardColors.primary(for: scheme))
                            .help("events.video_available")
                    }
                }
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            }

            Spacer()

            // Risk badge
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(Int(round(event.peakRisk * 100)))%")
                    .font(.system(.callout, design: .rounded).bold())
                    .foregroundColor(dotColor)
                Text(EventFormatting.statusLabel(event.status))
                    .font(FallGuardFont.caption2)
                    .padding(.horizontal, FallGuardSpacing.s8)
                    .padding(.vertical, 2)
                    .background(
                        event.status == "open"
                            ? FallGuardColors.amberLight
                            : FallGuardColors.greenLight
                    )
                    .foregroundColor(
                        event.status == "open"
                            ? FallGuardColors.amberDark
                            : FallGuardColors.greenDark
                    )
                    .clipShape(Capsule())
            }
        }
        .padding(.vertical, FallGuardSpacing.s8)
        .padding(.horizontal, FallGuardSpacing.s4)
    }
}

// MARK: - Event Detail

enum EventFormatting {
    private static let isoWithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let iso: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static func displayDate(_ value: String) -> String {
        guard let date = isoWithFractional.date(from: value) ??
                iso.date(from: value) else {
            return value
        }
        let formatter = DateFormatter()
        formatter.locale = LanguageManager.formattingLocale
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        return formatter.string(from: date)
    }

    static func statusLabel(_ status: String) -> LocalizedStringKey {
        switch status {
        case "open": return "event.status.open"
        case "reviewed": return "event.status.reviewed"
        case "ended", "resolved": return "event.status.ended"
        default: return LocalizedStringKey(status)
        }
    }
}

private struct EventDetailView: View {
    @EnvironmentObject private var store: AppStore
    @Environment(\.dismiss) private var dismiss

    let event: EventDTO
    let scheme: ColorScheme

    @State private var annotationLabel: String
    @State private var notes: String
    @State private var prefallStartSeconds: Double
    @State private var fallStartSeconds: Double
    @State private var isSaving = false

    init(event: EventDTO, scheme: ColorScheme) {
        self.event = event
        self.scheme = scheme
        let clipDuration = event.clipDurationSeconds ?? max(event.durationSeconds + 15, 1)
        let defaultPrefall = min(5, max(0, clipDuration * 0.25))
        let defaultFall = min(
            max(defaultPrefall + 0.5, defaultPrefall + event.durationSeconds * 0.5),
            max(defaultPrefall + 0.1, clipDuration - 0.1)
        )
        _annotationLabel = State(
            initialValue: event.annotationLabel ??
                (event.eventType == "fall" ? "Fall" : "Pre-fall")
        )
        _notes = State(initialValue: event.notes ?? "")
        _prefallStartSeconds = State(
            initialValue: event.prefallStartSeconds ?? defaultPrefall
        )
        _fallStartSeconds = State(
            initialValue: event.fallStartSeconds ?? defaultFall
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: FallGuardSpacing.s16) {
            HStack {
                Label(
                    NSLocalizedString(
                        event.eventType == "fall"
                            ? "event.type.fall" : "event.type.prefall",
                        comment: ""
                    ),
                    systemImage: event.eventType == "fall"
                        ? "exclamationmark.triangle.fill"
                        : "exclamationmark.circle.fill"
                )
                .font(FallGuardFont.title2)
                Spacer()
                Button(NSLocalizedString("event.detail.close", comment: "")) { dismiss() }
            }

            if let thumbnailPath = event.thumbnailPath,
               let image = NSImage(contentsOfFile: thumbnailPath) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 220)
                    .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.md))
            }

            VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
                detailRow("event.detail.started", EventFormatting.displayDate(event.startedAt))
                if let endedAt = event.endedAt {
                    detailRow("event.detail.ended", EventFormatting.displayDate(endedAt))
                }
                detailRow("event.detail.duration",
                          String(format: NSLocalizedString("event.duration.format", comment: ""),
                                 event.durationSeconds))
                detailRow("event.detail.peak_risk",
                          "\(Int(round(event.peakRisk * 100)))%")
                detailRow("event.detail.avg_risk",
                          "\(Int(round(event.avgRisk * 100)))%")
                detailRow("event.detail.status",
                          NSLocalizedString("event.status.\(normalizedStatus)", comment: ""))
                detailRow("event.detail.id", event.id)
            }

            if let videoPath = event.videoClipPath {
                Button {
                    NSWorkspace.shared.open(URL(fileURLWithPath: videoPath))
                } label: {
                    Label(
                        NSLocalizedString("event.detail.open_video", comment: ""),
                        systemImage: "play.rectangle"
                    )
                }
            }

            Text(NSLocalizedString("event.annotation.title", comment: ""))
                .font(FallGuardFont.headline)

            Picker("", selection: $annotationLabel) {
                ForEach(["Normal", "Pre-fall", "Fall"], id: \.self) { value in
                    Text(NSLocalizedString("event.annotation.\(value)", comment: ""))
                        .tag(value)
                }
            }
            .pickerStyle(.segmented)

            if annotationLabel != "Normal" {
                VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
                    Text(
                        String(
                            format: NSLocalizedString(
                                "event.annotation.clip_duration", comment: ""
                            ),
                            clipDuration
                        )
                    )
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.textSecondary(for: scheme))

                    HStack {
                        Text(NSLocalizedString(
                            "event.annotation.prefall_start", comment: ""
                        ))
                            .frame(width: 170, alignment: .leading)
                        TextField("", value: $prefallStartSeconds, format: .number)
                            .frame(width: 90)
                        Text(NSLocalizedString(
                            "event.annotation.seconds", comment: ""
                        ))
                    }
                    if annotationLabel == "Fall" {
                        HStack {
                            Text(NSLocalizedString(
                                "event.annotation.fall_start", comment: ""
                            ))
                                .frame(width: 170, alignment: .leading)
                            TextField("", value: $fallStartSeconds, format: .number)
                                .frame(width: 90)
                            Text(NSLocalizedString(
                                "event.annotation.seconds", comment: ""
                            ))
                        }
                    }
                    Text(NSLocalizedString("event.annotation.hint", comment: ""))
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
            } else {
                Text(NSLocalizedString(
                    "event.annotation.normal_excluded", comment: ""
                ))
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            }

            TextEditor(text: $notes)
                .frame(minHeight: 72)
                .overlay(
                    RoundedRectangle(cornerRadius: FallGuardRadius.sm)
                        .stroke(FallGuardColors.line(for: scheme), lineWidth: 1)
                )
                .accessibilityLabel(
                    Text(NSLocalizedString("event.feedback.notes", comment: ""))
                )

            HStack {
                Button(NSLocalizedString("events.copy_id", comment: "")) {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(event.id, forType: .string)
                }
                Spacer()
                Button {
                    isSaving = true
                    Task {
                        let saved = await store.updateEventFeedback(
                            id: event.id,
                            feedback: feedbackValue,
                            notes: notes,
                            annotationLabel: annotationLabel,
                            prefallStartSeconds: annotationLabel == "Normal"
                                ? nil : prefallStartSeconds,
                            fallStartSeconds: annotationLabel == "Fall"
                                ? fallStartSeconds : nil,
                            clipDurationSeconds: clipDuration
                        )
                        isSaving = false
                        if saved { dismiss() }
                    }
                } label: {
                    HStack(spacing: FallGuardSpacing.s8) {
                        if isSaving {
                            ProgressView()
                                .controlSize(.small)
                                .tint(.white)
                        } else {
                            Image(systemName: "checkmark")
                        }
                        Text(NSLocalizedString("event.feedback.save", comment: ""))
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .padding(.horizontal, FallGuardSpacing.s16)
                    .frame(minWidth: 142, minHeight: 36)
                }
                .buttonStyle(FallGuardButtonStyle(scheme: scheme))
                .disabled(isSaving || !annotationIsValid)
            }
        }
        .padding(FallGuardSpacing.s24)
        .frame(minWidth: 580, minHeight: 610)
        .background(FallGuardBackground(scheme: scheme))
    }

    private var clipDuration: Double {
        event.clipDurationSeconds ?? max(event.durationSeconds + 15, 1)
    }

    private var feedbackValue: String {
        switch annotationLabel {
        case "Fall": return "confirmed"
        case "Pre-fall": return "near_fall"
        default: return "normal"
        }
    }

    private var annotationIsValid: Bool {
        guard event.videoClipPath != nil || annotationLabel == "Normal" else {
            return false
        }
        guard annotationLabel != "Normal" else { return true }
        guard prefallStartSeconds >= 0, prefallStartSeconds < clipDuration else {
            return false
        }
        if annotationLabel == "Fall" {
            return fallStartSeconds > prefallStartSeconds &&
                fallStartSeconds < clipDuration
        }
        return true
    }

    private var normalizedStatus: String {
        switch event.status {
        case "open": return "open"
        case "reviewed": return "reviewed"
        default: return "ended"
        }
    }

    private func detailRow(_ key: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(NSLocalizedString(key, comment: ""))
                .frame(width: 120, alignment: .leading)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))
            Text(value)
                .textSelection(.enabled)
            Spacer()
        }
    }
}
