/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#dbe6fe",
          200: "#bfd3fe",
          300: "#93b4fd",
          400: "#608cfa",
          500: "#3b6bf6",
          600: "#2554eb",
          700: "#1d40d8",
          800: "#1e37af",
          900: "#1e3389",
        },
        // Care-path accent colors (mirror the backend's red/yellow/green).
        critical: "#dc2626",
        preventive: "#f59e0b",
        wellness: "#16a34a",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      backgroundImage: {
        "brand-gradient": "linear-gradient(135deg, #2554eb 0%, #6d28d9 100%)",
        "hero-radial":
          "radial-gradient(900px 500px at 12% -10%, rgba(59,107,246,0.14), transparent 60%), radial-gradient(700px 500px at 100% 0%, rgba(124,58,237,0.12), transparent 55%)",
      },
      boxShadow: {
        card: "0 1px 3px 0 rgb(0 0 0 / 0.07), 0 1px 2px -1px rgb(0 0 0 / 0.05)",
        "card-hover":
          "0 12px 28px -8px rgb(37 84 235 / 0.18), 0 6px 12px -8px rgb(0 0 0 / 0.12)",
        glow: "0 8px 30px -6px rgb(37 84 235 / 0.45)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(0.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.3s ease-out both",
        "scale-in": "scale-in 0.25s ease-out both",
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
