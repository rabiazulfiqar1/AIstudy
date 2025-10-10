import LightRays from "@/components/LightRays"
import { FeatureCards } from "@/components/feature-cards"
import Link from "next/link"

export default function Home() {
  return (
    <div className="relative w-full min-h-screen overflow-hidden flex flex-col items-center justify-center">
      {/* 🩵 Background with Light Rays */}
      <div className="absolute inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        <LightRays
          raysOrigin="top-center"
          raysColor="#ffffff"
          raysSpeed={1}
          lightSpread={0.8}
          rayLength={3.0}
          followMouse
          mouseInfluence={0.1}
          noiseAmount={0.0}
          distortion={0.0}
        />
        {/* subtle radial vignette */}
        <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      {/* 🌟 Navbar */}
      <header className="w-full">
        <nav className="absolute top-6 left-1/2 -translate-x-1/2 flex items-center justify-between w-[92%] max-w-5xl bg-white/10 backdrop-blur-md border border-white/10 rounded-full px-6 py-3">
          <span className="font-semibold tracking-wide text-background">React Bits</span>
          <div className="flex gap-6 text-background/80">
            <a
              href="#"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Home
              <span className="sr-only">{"current page"}</span>
            </a>
            <a
              href="#"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Features
            </a>
            <Link
              href="/signup"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Sign Up
            </Link>
            <Link
              href="/signout"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Sign Out
            </Link>
          </div>
        </nav>
      </header>

      {/* ✨ Center Content */}
      <main className="w-full text-center px-6 mt-28 md:mt-32">
        <h1 className="text-white/80 text-4xl md:text-6xl font-bold mb-5 leading-tight">
          May these lights guide you
          <br />
          on your learning path.
        </h1>
        <p className="mx-auto max-w-2xl text-background/80 mb-8 leading-relaxed">
          Process videos into knowledge, plan your journey, track your progress, and build real projects alongside a
          helpful community.
        </p>

        <div className="flex justify-center gap-4">
          {/* 🌟 Primary Button */}
          <button
            className="px-6 py-3 rounded-full bg-foreground text-background font-semibold transition-transform duration-200 hover:scale-105 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Get Started"
          >
            Get Started
          </button>

          {/* ✨ Secondary Button */}
          <button
            className="px-6 py-3 rounded-full border border-foreground/20 text-foreground font-semibold bg-background/40 backdrop-blur-sm transition-all duration-200 hover:bg-background/60 hover:border-foreground/30 hover:scale-105 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Learn More"
          >
            Learn More
          </button>
        </div>

        {/* 🔹 Feature Cards */}
        <section className="mt-14 md:mt-16">
          <FeatureCards />
        </section>
      </main>
    </div>
  )
}
