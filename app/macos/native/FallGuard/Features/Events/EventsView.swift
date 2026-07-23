import SwiftUI

// 文案修改位置：Resources/*/Localizable.strings 中的 Events 分组；布局代码无需修改。
/// Events list with type/status filtering.
struct EventsView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.colorScheme) private var colorScheme
    @State private var searchText: String = ""
    @State private var filterType: EventFilter = .all
    @State private var filterStatus: StatusFilter = .all

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
                        EventListRow(event: event, scheme: colorScheme)
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
        case .resolved: events = events.filter { $0.status == "resolved" }
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
                    Text(event.startedAt)
                        .font(FallGuardFont.caption)
                    if let ended = event.endedAt {
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                        Text(ended)
                            .font(FallGuardFont.caption)
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
                Text(event.status)
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
