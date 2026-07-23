import SwiftUI

// MARK: - Design Tokens

/// FallGuard Material Design 3 color and spacing tokens.
///
/// All colors adapt to light/dark mode via `@Environment(\.colorScheme)`.
enum FallGuardColors {

    // MARK: Primary

    // Kept as compatibility aliases for older call sites.  The product accent
    // follows the green palette used by the original desktop application.
    static let blue        = Color(hex: "#22C55E")
    static let blueDark    = Color(hex: "#16A34A")
    static let blueActive  = Color(hex: "#15803D")
    static let blueLight   = Color(hex: "#EAF7EF")
    static let blueBorder  = Color(hex: "#BBF7D0")

    // Dark mode primary
    static let blueDarkMode    = Color(hex: "#4ADE80")
    static let blueLightDark   = Color(hex: "#123D24")

    // MARK: Semantic

    static let green       = Color(hex: "#22C55E")
    static let greenLight  = Color(hex: "#EAF7EF")
    static let greenDark   = Color(hex: "#15803D")
    static let greenBorder = Color(hex: "#BBF7D0")

    static let amber       = Color(hex: "#FBBC04")
    static let amberLight  = Color(hex: "#FEF7E0")
    static let amberDark   = Color(hex: "#B06000")
    static let amberBorder = Color(hex: "#FDE293")

    static let red         = Color(hex: "#EA4335")
    static let redLight    = Color(hex: "#FCE8E6")
    static let redDark     = Color(hex: "#C5221F")
    static let redBorder   = Color(hex: "#FAD2CF")

    // MARK: Surface (light)

    static let bgLight        = Color(hex: "#F4FAF6")
    static let panelLight     = Color(hex: "#FFFFFF")
    static let sidebarBgLight = Color(hex: "#F5FBF7")
    static let sidebarBottomLight = Color(hex: "#F2FAF5")
    static let hoverLight     = Color(hex: "#EEF8F1")
    static let navActiveLight = Color(hex: "#EAF7EF")

    // MARK: Surface (dark)

    static let bgDark         = Color(hex: "#08150E")
    static let panelDark      = Color(hex: "#0E1C14")
    static let sidebarBgDark  = Color(hex: "#0A1810")
    static let hoverDark      = Color(hex: "#14291B")
    static let navActiveDark  = Color(hex: "#123D24")

    // MARK: Text

    static let textLight       = Color(hex: "#1F2937")
    static let textSecondaryLight = Color(hex: "#6B7280")
    static let mutedLight      = Color(hex: "#9CA3AF")

    static let textDark        = Color(hex: "#F9FAFB")
    static let textSecondaryDark = Color(hex: "#CBD5E1")
    static let mutedDark       = Color(hex: "#94A3B8")

    // MARK: Borders & Lines

    static let lineLight = Color(hex: "#D9E9DE")
    static let lineDark  = Color(hex: "#284331")

    // MARK: Video

    static let videoBg   = Color(hex: "#111827")
    static let videoText = Color(hex: "#D1D5DB")

    // MARK: Context-aware helpers

    /// Returns the primary accent color for the given color scheme.
    static func primary(for scheme: ColorScheme) -> Color {
        scheme == .dark ? blueDarkMode : blue
    }

    static func surface(for scheme: ColorScheme) -> Color {
        scheme == .dark ? panelDark : panelLight
    }

    static func background(for scheme: ColorScheme) -> Color {
        scheme == .dark ? bgDark : bgLight
    }

    static func sidebarBg(for scheme: ColorScheme) -> Color {
        scheme == .dark ? sidebarBgDark : sidebarBgLight
    }

    static func textPrimary(for scheme: ColorScheme) -> Color {
        scheme == .dark ? textDark : textLight
    }

    static func textSecondary(for scheme: ColorScheme) -> Color {
        scheme == .dark ? textSecondaryDark : textSecondaryLight
    }

    static func muted(for scheme: ColorScheme) -> Color {
        scheme == .dark ? mutedDark : mutedLight
    }

    static func line(for scheme: ColorScheme) -> Color {
        scheme == .dark ? lineDark : lineLight
    }

    static func hoverBg(for scheme: ColorScheme) -> Color {
        scheme == .dark ? hoverDark : hoverLight
    }

    static func navActiveBg(for scheme: ColorScheme) -> Color {
        scheme == .dark ? navActiveDark : navActiveLight
    }

    static func sidebarTint(for scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: "#0B2615") : Color(hex: "#E9F8EE")
    }

    static func glassTint(for scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(hex: "#123D24") : Color(hex: "#DDF8E7")
    }
}

// MARK: - Ambient App Background

/// A softly coloured backdrop that gives translucent panels something to blur.
/// Decorative layers never participate in hit testing.
struct FallGuardBackground: View {
    let scheme: ColorScheme

    var body: some View {
        ZStack {
            LinearGradient(
                colors: scheme == .dark
                    ? [Color(hex: "#07120C"), Color(hex: "#0B1A12"), Color(hex: "#0A1512")]
                    : [Color(hex: "#F8FCF9"), Color(hex: "#EEF8F1"), Color(hex: "#E8F6ED")],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            RadialGradient(
                colors: [
                    FallGuardColors.primary(for: scheme).opacity(scheme == .dark ? 0.18 : 0.13),
                    .clear
                ],
                center: .topTrailing,
                startRadius: 12,
                endRadius: 520
            )

            RadialGradient(
                colors: [
                    Color(hex: scheme == .dark ? "#0EA5A0" : "#A7F3D0")
                        .opacity(scheme == .dark ? 0.08 : 0.16),
                    .clear
                ],
                center: .bottomLeading,
                startRadius: 20,
                endRadius: 460
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

// MARK: - Color Hex Initializer

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 6:
            (a, r, g, b) = (255, (int >> 16) & 0xFF, (int >> 8) & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = ((int >> 24) & 0xFF, (int >> 16) & 0xFF, (int >> 8) & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - Spacing (8px grid)

enum FallGuardSpacing {
    static let s4: CGFloat  = 4
    static let s8: CGFloat  = 8
    static let s10: CGFloat = 10
    static let s12: CGFloat = 12
    static let s14: CGFloat = 14
    static let s16: CGFloat = 16
    static let s20: CGFloat = 20
    static let s24: CGFloat = 24
    static let s32: CGFloat = 32
    static let s40: CGFloat = 40
}

// MARK: - Corner Radius

enum FallGuardRadius {
    static let sm: CGFloat    = 8
    static let md: CGFloat    = 12
    static let lg: CGFloat    = 16
    static let xl: CGFloat    = 20
    static let full: CGFloat  = 9999
}

// MARK: - Typography

enum FallGuardFont {
    static let caption2: Font = .system(size: 11)
    static let caption: Font  = .system(size: 12)
    static let body: Font     = .system(size: 13)
    static let callout: Font  = .system(size: 14)
    static let headline: Font = .system(size: 16, weight: .semibold)
    static let title3: Font   = .system(size: 18, weight: .semibold, design: .rounded)
    static let title2: Font   = .system(size: 22, weight: .semibold, design: .rounded)
    static let title: Font    = .system(size: 28, weight: .bold, design: .rounded)
    static let hero: Font     = .system(size: 40, weight: .bold, design: .rounded)
}

// MARK: - Animation Durations

enum FallGuardAnimation {
    static let fast: Double    = 0.15
    static let normal: Double  = 0.20
    static let slow: Double    = 0.30
    static let sidebar: Double = 0.55
}

// MARK: - Shadows

/// Shadow color for card elevation — light in light mode, stronger in dark.
func fallGuardShadowColor(for scheme: ColorScheme) -> Color {
    scheme == .dark ? .black.opacity(0.36) : Color(hex: "#14532D").opacity(0.1)
}

extension View {
    /// Apply FallGuard card shadow.
    func fallGuardCardShadow(scheme: ColorScheme) -> some View {
        self.shadow(color: fallGuardShadowColor(for: scheme), radius: 14, y: 6)
    }
}
