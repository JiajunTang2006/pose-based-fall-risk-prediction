import SwiftUI

// 文案修改位置：Resources/*/Localizable.strings 中的 Profiles 分组；布局代码无需修改。
/// Profile management with card-style layout.
struct ProfilesView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.colorScheme) private var colorScheme
    @State private var newProfileName: String = ""
    @State private var showingCreateAlert: Bool = false
    @State private var showingDeleteAlert: Bool = false
    @State private var profileToDelete: ProfileDTO?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("profiles.title")
                    .font(FallGuardFont.title)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                Spacer()
                Button(action: { showingCreateAlert = true }) {
                    Label("profiles.create", systemImage: "plus")
                        .font(.system(size: 14, weight: .semibold))
                        .padding(.horizontal, FallGuardSpacing.s14)
                        .padding(.vertical, 7)
                }
                .buttonStyle(FallGuardButtonStyle(scheme: colorScheme))
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.top, FallGuardSpacing.s20)
            .padding(.bottom, FallGuardSpacing.s12)

            GlassDivider()
                .padding(.horizontal, FallGuardSpacing.s24)

            if store.profiles.isEmpty {
                Spacer()
                VStack(spacing: FallGuardSpacing.s12) {
                    Image(systemName: "person.crop.circle.badge.questionmark")
                        .font(.system(size: 40))
                        .foregroundColor(FallGuardColors.muted(for: colorScheme))
                    Text("profiles.empty")
                        .font(FallGuardFont.body)
                        .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))
                }
                Spacer()
            } else {
                ScrollView {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 280), spacing: FallGuardSpacing.s16)],
                        spacing: FallGuardSpacing.s16
                    ) {
                        ForEach(store.profiles) { profile in
                            ProfileCard(
                                profile: profile,
                                isActive: profile.id == store.activeProfileId,
                                scheme: colorScheme,
                                onActivate: { Task { await store.activateProfile(id: profile.id) } },
                                onDelete: {
                                    profileToDelete = profile
                                    showingDeleteAlert = true
                                }
                            )
                        }
                    }
                    .padding(FallGuardSpacing.s24)
                }
            }

            // Status bar
            HStack {
                Text(String(format: NSLocalizedString("profiles.count", comment: ""),
                           store.profiles.count))
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.muted(for: colorScheme))
                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.vertical, FallGuardSpacing.s8)
        }
        .background(FallGuardBackground(scheme: colorScheme))
        .onAppear { Task { await store.loadProfiles() } }
        .alert("profiles.create", isPresented: $showingCreateAlert) {
            TextField("profiles.name_placeholder", text: $newProfileName)
            Button("profiles.create_button") {
                let name = newProfileName.trimmingCharacters(in: .whitespaces)
                guard !name.isEmpty else { return }
                Task { await store.createProfile(name: name) }
                newProfileName = ""
            }
            Button("cancel", role: .cancel) { newProfileName = "" }
        } message: {
            Text("profiles.create_message")
        }
        .alert("profiles.delete_title", isPresented: $showingDeleteAlert) {
            Button("delete", role: .destructive) {
                guard let p = profileToDelete else { return }
                Task { await store.deleteProfile(id: p.id) }
                profileToDelete = nil
            }
            Button("cancel", role: .cancel) { profileToDelete = nil }
        } message: {
            Text(String(format: NSLocalizedString("profiles.delete_message", comment: ""),
                       profileToDelete?.name ?? ""))
        }
    }
}

// MARK: - Profile Card

struct ProfileCard: View {
    let profile: ProfileDTO
    let isActive: Bool
    let scheme: ColorScheme
    let onActivate: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(spacing: FallGuardSpacing.s12) {
            // Top
            HStack {
                ZStack {
                    Circle()
                        .fill(
                            isActive
                                ? FallGuardColors.greenLight
                                : FallGuardColors.line(for: scheme)
                        )
                        .frame(width: 40, height: 40)
                    Image(systemName: isActive ? "checkmark" : "person.fill")
                        .font(.callout)
                        .foregroundColor(
                            isActive
                                ? FallGuardColors.greenDark
                                : FallGuardColors.muted(for: scheme)
                        )
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(profile.name)
                        .font(FallGuardFont.headline)
                        .foregroundColor(FallGuardColors.textPrimary(for: scheme))
                    Text(formattedCreatedAt)
                        .font(FallGuardFont.caption2)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }

                Spacer()

                if isActive {
                    Text("profiles.active_label")
                        .font(FallGuardFont.caption2)
                        .fontWeight(.semibold)
                        .padding(.horizontal, FallGuardSpacing.s8)
                        .padding(.vertical, 3)
                        .background(FallGuardColors.greenLight)
                        .foregroundColor(FallGuardColors.greenDark)
                        .clipShape(Capsule())
                }
            }

            Divider()

            // Bottom
            HStack {
                Label(
                    String(format: NSLocalizedString("profiles.fall_count", comment: ""), profile.fallCount),
                    systemImage: "exclamationmark.triangle"
                )
                .font(FallGuardFont.caption)
                .foregroundColor(FallGuardColors.textSecondary(for: scheme))

                Spacer()

                if !isActive {
                    Button(action: onActivate) {
                        Text("profiles.activate")
                            .font(.system(size: 13, weight: .semibold))
                            .padding(.horizontal, FallGuardSpacing.s12)
                            .padding(.vertical, 6)
                    }
                    .buttonStyle(FallGuardButtonStyle(scheme: scheme))
                }

                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundColor(FallGuardColors.muted(for: scheme))
                }
                .buttonStyle(.borderless)
                .help("profiles.delete")
            }
        }
        .padding(FallGuardSpacing.s16)
        .glassSurface(cornerRadius: FallGuardRadius.lg)
        .overlay(
            RoundedRectangle(cornerRadius: FallGuardRadius.lg)
                .stroke(
                    isActive ? FallGuardColors.green : Color.clear,
                    lineWidth: isActive ? 2 : 0
                )
                .allowsHitTesting(false)
        )
        .overlay(
            RoundedRectangle(cornerRadius: FallGuardRadius.lg)
                .stroke(
                    isActive ? Color.clear
                        : (scheme == .dark ? Color.white.opacity(0.1) : Color.black.opacity(0.08)),
                    lineWidth: 0.5
                )
                .allowsHitTesting(false)
        )
    }

    private var formattedCreatedAt: String {
        let parser = ISO8601DateFormatter()
        parser.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = parser.date(from: profile.createdAt) else {
            return profile.createdAt
        }
        return DateFormatter.localizedString(from: date, dateStyle: .medium, timeStyle: .none)
    }
}
