/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
        sans: ["Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      colors: {
        // AcoustiGuard Signature Audio Palette
        navy: {
          DEFAULT: "#101820",
          deep: "#0a0f15",
          surface: "#141e28",
          light: "#182430",
        },
        charcoal: {
          DEFAULT: "#202A33",
          surface: "#25313c",
          border: "#2c3b47",
          hover: "#344553",
        },
        ivory: {
          DEFAULT: "#F4F0E8",
          muted: "#ded7c8",
          dim: "#a39d91",
        },
        lime: {
          DEFAULT: "#C7F464",
          hover: "#b6e94e",
          dim: "rgba(199, 244, 100, 0.15)",
          border: "rgba(199, 244, 100, 0.35)",
        },
        orange: {
          DEFAULT: "#FF6B35",
          hover: "#f35a22",
          dim: "rgba(255, 107, 53, 0.15)",
          border: "rgba(255, 107, 53, 0.35)",
        },
        steel: {
          DEFAULT: "#64748B",
          dark: "#475569",
          light: "#94a3b8",
        },
        // Unique Audio Hardware & Spectral Accent Colors
        cyan: {
          DEFAULT: "#00F0FF",
          hover: "#00d4e0",
          dim: "rgba(0, 240, 255, 0.12)",
          border: "rgba(0, 240, 255, 0.35)",
        },
        violet: {
          DEFAULT: "#BD00FF",
          light: "#D946EF",
          surface: "#A855F7",
          dim: "rgba(189, 0, 255, 0.12)",
          border: "rgba(189, 0, 255, 0.35)",
        },
        amber: {
          DEFAULT: "#F59E0B",
          hover: "#d97706",
          dim: "rgba(245, 158, 11, 0.12)",
          border: "rgba(245, 158, 11, 0.35)",
        },
        emerald: {
          DEFAULT: "#10B981",
          hover: "#059669",
          dim: "rgba(16, 185, 129, 0.12)",
          border: "rgba(16, 185, 129, 0.35)",
        },
        // Backward-compatibility brand alias mapped to electric lime
        brand: {
          50: "#f7fee7",
          100: "#ecfccb",
          200: "#d9f99d",
          300: "#C7F464",
          400: "#b6e94e",
          500: "#C7F464",
          600: "#a3d936",
          700: "#84b325",
          800: "#658a1d",
          900: "#496316",
        },
      },
      borderRadius: {
        none: "0px",
        sm: "2px",
        DEFAULT: "0px",
        md: "4px",
      },
      boxShadow: {
        sharp: "4px 4px 0px 0px #202A33",
        "sharp-lime": "4px 4px 0px 0px #C7F464",
        "sharp-orange": "4px 4px 0px 0px #FF6B35",
        "sharp-cyan": "4px 4px 0px 0px #00F0FF",
        "sharp-violet": "4px 4px 0px 0px #BD00FF",
        "sharp-amber": "4px 4px 0px 0px #F59E0B",
      },
    },
  },
  plugins: [],
};
