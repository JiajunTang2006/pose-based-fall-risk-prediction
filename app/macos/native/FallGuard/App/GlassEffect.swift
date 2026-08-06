import SwiftUI
import AppKit

struct GlassEffect: NSViewRepresentable {

    var material: NSVisualEffectView.Material = .sidebar

    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow

    var state: NSVisualEffectView.State = .active

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = state
        view.isEmphasized = true
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
        nsView.state = state
    }
}

extension View {

    func glassSidebar() -> some View {
        self.background(GlassEffect(
            material: .sidebar,
            blendingMode: .behindWindow
        ).allowsHitTesting(false))
    }

    @ViewBuilder
    func liquidGlass(
        cornerRadius: CGFloat = FallGuardRadius.lg,
        tint: Color? = nil,
        interactive: Bool = false
    ) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        if #available(macOS 26.0, *) {
            self.glassEffect(
                Self.makeGlass(tint: tint, interactive: interactive),
                in: shape
            )
        } else {
            self
                .glassSurface(cornerRadius: cornerRadius)
                .overlay(
                    shape
                        .fill((tint ?? .clear).opacity(tint == nil ? 0 : 0.14))
                        .allowsHitTesting(false)
                )
        }
    }

    @available(macOS 26.0, *)
    private static func makeGlass(tint: Color?, interactive: Bool) -> Glass {
        var glass: Glass = .regular
        if let tint { glass = glass.tint(tint) }
        if interactive { glass = glass.interactive() }
        return glass
    }

    func glassCard(cornerRadius: CGFloat = FallGuardRadius.xl) -> some View {
        self.background(
            GlassPanelBackground(cornerRadius: cornerRadius, material: .regular)
                .allowsHitTesting(false)
        )
    }

    func glassHeader() -> some View {
        self.background(
            GlassHeaderBackground()
                .allowsHitTesting(false)
        )
    }

    func glassSurface(cornerRadius: CGFloat = FallGuardRadius.lg) -> some View {
        self.background(
            GlassPanelBackground(cornerRadius: cornerRadius, material: .thin)
                .allowsHitTesting(false)
        )
    }
}

private struct GlassPanelBackground: View {
    enum MaterialWeight { case thin, regular }

    @Environment(\.colorScheme) private var scheme
    let cornerRadius: CGFloat
    let material: MaterialWeight

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        ZStack {
            switch material {
            case .regular:
                shape.fill(.regularMaterial)
            case .thin:
                shape.fill(.thinMaterial)
            }
            shape.fill(
                LinearGradient(
                    colors: [
                        Color.white.opacity(scheme == .dark ? 0.035 : 0.36),
                        FallGuardColors.glassTint(for: scheme)
                            .opacity(scheme == .dark ? 0.11 : 0.19)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        }
    }
}

private struct GlassHeaderBackground: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ZStack {
            Rectangle().fill(.ultraThinMaterial)
            LinearGradient(
                colors: [
                    Color.white.opacity(scheme == .dark ? 0.02 : 0.28),
                    FallGuardColors.glassTint(for: scheme)
                        .opacity(scheme == .dark ? 0.08 : 0.14)
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        }
    }
}

struct GlassDivider: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Rectangle()
            .fill(colorScheme == .dark
                ? FallGuardColors.green.opacity(0.16)
                : FallGuardColors.greenDark.opacity(0.13))
            .frame(height: 1)
    }
}

struct GlassVerticalDivider: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Rectangle()
            .fill(colorScheme == .dark
                ? FallGuardColors.green.opacity(0.16)
                : FallGuardColors.greenDark.opacity(0.13))
            .frame(width: 1)
            .allowsHitTesting(false)
    }
}

struct BrandFigureShape: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let w = rect.width
        let h = rect.height

        let headD = w * 0.30
        let headRect = CGRect(
            x: rect.midX - headD / 2,
            y: h * 0.06,
            width: headD,
            height: headD
        )
        p.addEllipse(in: headRect)

        let baseX = rect.midX
        let baseY = h * 0.98
        let tipY  = h * 0.30            // arms reach up to about head height

        p.move(to: CGPoint(x: baseX, y: baseY))
        p.addQuadCurve(
            to: CGPoint(x: w * 0.14, y: tipY),
            control: CGPoint(x: w * 0.10, y: h * 0.74)
        )
        p.addQuadCurve(
            to: CGPoint(x: baseX, y: baseY),
            control: CGPoint(x: w * 0.44, y: h * 0.60)
        )

        p.move(to: CGPoint(x: baseX, y: baseY))
        p.addQuadCurve(
            to: CGPoint(x: w * 0.86, y: tipY),
            control: CGPoint(x: w * 0.90, y: h * 0.74)
        )
        p.addQuadCurve(
            to: CGPoint(x: baseX, y: baseY),
            control: CGPoint(x: w * 0.56, y: h * 0.60)
        )

        return p
    }
}

struct BrandMark: View {
    let scheme: ColorScheme
    var size: CGFloat = 44

    private var figureGradient: LinearGradient {
        LinearGradient(
            colors: [
                Color(hex: "#4ADE80"),   // light leaf green (top)
                Color(hex: "#22C55E"),
                Color(hex: "#15803D")    // deep green (bottom)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
    }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.30, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: scheme == .dark
                            ? [Color(hex: "#F4FBF6"), Color(hex: "#E4F5EA")]
                            : [Color.white, Color(hex: "#EAF7EF")],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: size * 0.30, style: .continuous)
                        .stroke(FallGuardColors.green.opacity(0.35), lineWidth: 0.75)
                )

            BrandFigureShape()
                .fill(figureGradient)
                .frame(width: size * 0.62, height: size * 0.62)
        }
        .frame(width: size, height: size)
    }
}
