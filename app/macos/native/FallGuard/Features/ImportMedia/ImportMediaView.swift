import SwiftUI
import UniformTypeIdentifiers

// 文案修改位置：Resources/*/Localizable.strings 中的 Import 分组；布局代码无需修改。
/// Import Media view with drag-and-drop support.
struct ImportMediaView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.colorScheme) private var colorScheme
    @State private var selectedPaths: [URL] = []
    @State private var sensitivity: String = "medium"
    @State private var isDropTargeted: Bool = false

    private let sensitivities = ["low", "medium", "high"]

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("import.title")
                    .font(FallGuardFont.title)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s24)
            .padding(.top, FallGuardSpacing.s20)
            .padding(.bottom, FallGuardSpacing.s12)

            GlassDivider()
                .padding(.horizontal, FallGuardSpacing.s24)

            if let job = store.importJob, store.isImporting {
                importProgressView(job: job)
            } else {
                importSetupView
            }
        }
        .background(FallGuardBackground(scheme: colorScheme))
    }

    // MARK: Setup

    private var importSetupView: some View {
        VStack(spacing: FallGuardSpacing.s24) {
            Spacer()

            // Drop zone
            dropZone
                .padding(.horizontal, FallGuardSpacing.s40)

            // Selected files
            if !selectedPaths.isEmpty {
                VStack(alignment: .leading, spacing: FallGuardSpacing.s8) {
                    HStack {
                        Text("import.selected")
                            .font(FallGuardFont.headline)
                            .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                        Spacer()
                        Button(action: { selectedPaths = [] }) {
                            Label("import.clear", systemImage: "xmark.circle.fill")
                                .font(FallGuardFont.caption)
                                .foregroundColor(FallGuardColors.muted(for: colorScheme))
                        }
                        .buttonStyle(.borderless)
                    }

                    ForEach(selectedPaths, id: \.self) { url in
                        HStack(spacing: FallGuardSpacing.s8) {
                            Image(systemName: fileIcon(for: url))
                                .foregroundColor(FallGuardColors.primary(for: colorScheme))
                                .frame(width: 24)
                            Text(url.lastPathComponent)
                                .font(FallGuardFont.body)
                                .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                                .lineLimit(1)
                            Spacer()
                            Text(formattedSize(for: url))
                                .font(FallGuardFont.caption)
                                .foregroundColor(FallGuardColors.muted(for: colorScheme))
                        }
                        .padding(.horizontal, FallGuardSpacing.s12)
                        .padding(.vertical, FallGuardSpacing.s8)
                        .background(FallGuardColors.hoverBg(for: colorScheme))
                        .clipShape(RoundedRectangle(cornerRadius: FallGuardRadius.sm))
                    }
                }
                .padding(.horizontal, FallGuardSpacing.s40)
            }

            // Options
            HStack(spacing: FallGuardSpacing.s20) {
                Picker(selection: $sensitivity) {
                    ForEach(sensitivities, id: \.self) { s in
                        Text(NSLocalizedString("sensitivity.\(s)", comment: "")).tag(s)
                    }
                } label: {
                    Text("settings.sensitivity")
                        .font(FallGuardFont.body)
                        .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))
                }
                .frame(width: 180)

                Spacer()
            }
            .padding(.horizontal, FallGuardSpacing.s40)

            // Import button
            Button(action: startImport) {
                Label("import.start", systemImage: "play.fill")
                    .font(FallGuardFont.headline)
                    .padding(.horizontal, FallGuardSpacing.s32)
                    .padding(.vertical, FallGuardSpacing.s12)
            }
            .buttonStyle(FallGuardButtonStyle(scheme: colorScheme))
            .controlSize(.large)
            .disabled(selectedPaths.isEmpty)

            if let err = store.connectionError {
                Text(err)
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.red)
            }

            Spacer()
        }
    }

    // MARK: Drop Zone

    private var dropZone: some View {
        VStack(spacing: FallGuardSpacing.s12) {
            Image(systemName: "square.and.arrow.down.on.square")
                .font(.system(size: 48))
                .foregroundColor(
                    isDropTargeted
                        ? FallGuardColors.primary(for: colorScheme)
                        : FallGuardColors.primary(for: colorScheme).opacity(0.6)
                )

            Text("import.select_files")
                .font(FallGuardFont.title3)
                .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))

            Text("import.supported_formats")
                .font(FallGuardFont.body)
                .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))

            HStack(spacing: FallGuardSpacing.s16) {
                Button(action: selectFiles) {
                    Label("import.choose_video", systemImage: "film")
                }
                .buttonStyle(FallGuardSecondaryButtonStyle(scheme: colorScheme))

                Button(action: selectFolder) {
                    Label("import.choose_folder", systemImage: "folder")
                }
                .buttonStyle(FallGuardSecondaryButtonStyle(scheme: colorScheme))
            }
            .padding(.top, FallGuardSpacing.s4)
        }
        .padding(FallGuardSpacing.s40)
        .frame(maxWidth: .infinity)
        .glassCard(cornerRadius: FallGuardRadius.xl)
        .background(
            RoundedRectangle(cornerRadius: FallGuardRadius.xl)
                .strokeBorder(
                    style: StrokeStyle(
                        lineWidth: isDropTargeted ? 3 : 2,
                        dash: [8, 4]
                    )
                )
                .foregroundColor(
                    isDropTargeted
                        ? FallGuardColors.primary(for: colorScheme)
                        : FallGuardColors.greenDark.opacity(0.28)
                )
        )
        .background(
            RoundedRectangle(cornerRadius: FallGuardRadius.xl)
                .fill(isDropTargeted
                    ? FallGuardColors.primary(for: colorScheme).opacity(0.04)
                    : Color.clear)
        )
        .animation(.easeInOut(duration: FallGuardAnimation.fast), value: isDropTargeted)
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers: providers)
            return true
        }
    }

    // MARK: Progress

    private func importProgressView(job: ImportJobDTO) -> some View {
        VStack(spacing: FallGuardSpacing.s24) {
            Spacer()

            if job.state == .complete {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(FallGuardColors.green)
                Text("import.complete")
                    .font(FallGuardFont.title2)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))

                if let output = job.outputVideo {
                    HStack(spacing: FallGuardSpacing.s8) {
                        Text("import.output")
                            .font(FallGuardFont.body)
                            .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))
                        Text(URL(fileURLWithPath: output).lastPathComponent)
                            .font(FallGuardFont.body)
                        Button(action: {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: output)])
                        }) {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.borderless)
                    }
                }

                Button("import.new") {
                    selectedPaths = []
                    store.importJob = nil
                }
                .buttonStyle(FallGuardButtonStyle(scheme: colorScheme))
            } else if job.state == .error {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(FallGuardColors.red)
                Text("import.failed")
                    .font(FallGuardFont.title2)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))
                if let err = job.error {
                    Text(err.messageKey)
                        .font(FallGuardFont.body)
                        .foregroundColor(FallGuardColors.textSecondary(for: colorScheme))
                }
                Button("import.retry") { store.importJob = nil }
                    .buttonStyle(FallGuardSecondaryButtonStyle(scheme: colorScheme))
            } else {
                ProgressView()
                    .scaleEffect(1.5)
                Text("import.processing")
                    .font(FallGuardFont.title3)
                    .foregroundColor(FallGuardColors.textPrimary(for: colorScheme))

                VStack(spacing: FallGuardSpacing.s8) {
                    ProgressView(value: job.progress)
                        .frame(width: 350)
                    HStack {
                        Text("\(Int(job.progress * 100))%")
                        Spacer()
                        Text("Frame \(job.currentFrame) / \(job.totalFrames)")
                    }
                    .font(FallGuardFont.caption)
                    .foregroundColor(FallGuardColors.muted(for: colorScheme))
                    .frame(width: 350)
                }

                Button(role: .destructive, action: {
                    store.importJob = nil
                    store.isImporting = false
                }) {
                    Label("import.cancel", systemImage: "xmark")
                }
                .buttonStyle(.borderless)
            }

            Spacer()
        }
    }

    // MARK: Actions

    private func selectFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie, .png, .jpeg, .bmp, .tiff]
        panel.message = NSLocalizedString("import.panel.message", comment: "")
        panel.prompt = NSLocalizedString("import.panel.select", comment: "")
        if panel.runModal() == .OK { selectedPaths = panel.urls }
    }

    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = NSLocalizedString("import.panel.folder.message", comment: "")
        panel.prompt = NSLocalizedString("import.panel.select", comment: "")
        if panel.runModal() == .OK, let url = panel.url { selectedPaths = [url] }
    }

    private func startImport() {
        guard !selectedPaths.isEmpty else { return }
        Task {
            await store.startImport(paths: selectedPaths.map { $0.path })
        }
    }

    private func handleDrop(providers: [NSItemProvider]) {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
                if let urlData = data as? Data,
                   let path = String(data: urlData, encoding: .utf8),
                   let url = URL(string: path) {
                    DispatchQueue.main.async {
                        if !selectedPaths.contains(url) {
                            selectedPaths.append(url)
                        }
                    }
                }
            }
        }
    }

    private func fileIcon(for url: URL) -> String {
        let videoExts: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]
        return videoExts.contains(url.pathExtension.lowercased()) ? "film" : "photo"
    }

    private func formattedSize(for url: URL) -> String {
        guard let attrs = try? url.resourceValues(forKeys: [.fileSizeKey]),
              let size = attrs.fileSize else { return "" }
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }
}
