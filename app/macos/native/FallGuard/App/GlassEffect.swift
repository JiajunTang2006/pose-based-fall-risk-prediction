import SwiftUI
import AppKit

// MARK: - NSVisualEffectView Wrapper

/// Authentic macOS frosted glass backed by ``NSVisualEffectView``.
///
/// Use ``GlassEffect`` when you need the material to blur content
/// *behind the window* (sidebar glass) — something SwiftUI's native
/// `.material` backgrounds cannot do.
///
/// For in-window blur (cards, panels), prefer the native
/// `.background(.regularMaterial, in: shape)` modifier.
struct GlassEffect: NSViewRepresentable {

    /// The material style (`.sidebar`, `.menu`, `.hudWindow`, etc.).
    var material: NSVisualEffectView.Material = .sidebar

    /// `.behindWindow` blurs the desktop / other windows.
    /// `.withinWindow` blurs content behind this view inside the window.
    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow

    /// `.active` (default), `.inactive`, or `.followsWindowActiveState`.
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

// MARK: - Convenience View Modifiers

extension View {

    /// Authentic macOS sidebar glass — blurs the desktop wallpaper
    /// and other windows behind the sidebar.
    ///
    /// Use this for the navigation sidebar and settings sidebar.
    func glassSidebar() -> some View {
        self.background(GlassEffect(
            material: .sidebar,
            blendingMode: .behindWindow
        ).allowsHitTesting(false))
    }

    /// Apple **Liquid Glass** surface (macOS 26+).
    ///
    /// On macOS 26 and later this uses the native `.glassEffect(_:in:)`
    /// modifier — the real Liquid Glass material that refracts and
    /// reflects the content behind it.  On earlier systems it gracefully
    /// falls back to the existing frosted-glass surface so the app still
    /// builds and looks right on the macOS 12 deployment target.
    ///
    /// - Parameters:
    ///   - cornerRadius: corner radius of the glass shape.
    ///   - tint: optional accent tint blended into the glass.
    ///   - interactive: whether the glass reacts to press/hover (buttons).
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
            // Fallback: existing frosted surface with an optional tint wash.
            self
                .glassSurface(cornerRadius: cornerRadius)
                .overlay(
                    shape
                        .fill((tint ?? .clear).opacity(tint == nil ? 0 : 0.14))
                        .allowsHitTesting(false)
                )
        }
    }

    /// Builds the configured `Glass` value outside of any `@ViewBuilder`
    /// context (where plain statements would be mis-parsed as views).
    @available(macOS 26.0, *)
    private static func makeGlass(tint: Color?, interactive: Bool) -> Glass {
        var glass: Glass = .regular
        if let tint { glass = glass.tint(tint) }
        if interactive { glass = glass.interactive() }
        return glass
    }

    /// In-window frosted glass for card panels.
    ///
    /// Uses SwiftUI's native `.regularMaterial` for within-window blur
    /// of the content behind the card.
    func glassCard(cornerRadius: CGFloat = FallGuardRadius.xl) -> some View {
        self.background(
            GlassPanelBackground(cornerRadius: cornerRadius, material: .regular)
                .allowsHitTesting(false)
        )
    }

    /// Subtle glass for toolbar and header areas.
    func glassHeader() -> some View {
        self.background(
            GlassHeaderBackground()
                .allowsHitTesting(false)
        )
    }

    /// Lightweight frosted surface — thinner than glassCard.
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

// MARK: - Glass-Friendly Separator

/// A subtle divider designed to sit on top of glass backgrounds.
/// Slightly more opaque than the standard divider so it remains
/// visible against the blurred material.
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

/// Vertical counterpart used between sidebars and content columns.
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

// MARK: - Brand Mark (green figure)

/// Vector rendering of the FallGuard mascot — the green "person" from the
/// app icon: a round head above two upraised leaf-shaped arms.  Drawn as a
/// resolution-independent `Shape` so it stays crisp at any badge size and
/// needs no bundled image asset.
struct BrandFigureShape: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let w = rect.width
        let h = rect.height

        // Head — a circle in the upper-middle.
        let headD = w * 0.30
        let headRect = CGRect(
            x: rect.midX - headD / 2,
            y: h * 0.06,
            width: headD,
            height: headD
        )
        p.addEllipse(in: headRect)

        // Body base where the two leaf-arms meet.
        let baseX = rect.midX
        let baseY = h * 0.98
        let tipY  = h * 0.30            // arms reach up to about head height

        // Left leaf-arm: base → up-left tip → back to base, two curves
        // forming a pointed-oval leaf.
        p.move(to: CGPoint(x: baseX, y: baseY))
        p.addQuadCurve(
            to: CGPoint(x: w * 0.14, y: tipY),
            control: CGPoint(x: w * 0.10, y: h * 0.74)
        )
        p.addQuadCurve(
            to: CGPoint(x: baseX, y: baseY),
            control: CGPoint(x: w * 0.44, y: h * 0.60)
        )

        // Right leaf-arm — mirror of the left.
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

/// Legacy vector badge retained for compatibility with older views.
/// Replaces the old `shield.checkered` SF Symbol with the brand's green figure.
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
            // Light badge background, matching the icon's white shield card.
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

            // Green figure.
            BrandFigureShape()
                .fill(figureGradient)
                .frame(width: size * 0.62, height: size * 0.62)
        }
        .frame(width: size, height: size)
    }
}
