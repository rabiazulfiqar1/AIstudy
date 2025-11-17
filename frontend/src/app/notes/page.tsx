import LightRays from "@/components/LightRays"
import { VideoUploadForm } from "@/components/video-upload-form"
import Link from "next/link"

export default function UploadPage() {
  return (
    <div className="relative w-full min-h-screen overflow-hidden flex flex-col">
      {/* Background with Light Rays */}
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
        <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      {/* Navbar */}
      <header className="w-full">
        <nav className="absolute top-6 left-1/2 -translate-x-1/2 flex items-center justify-between w-[92%] max-w-5xl bg-white/10 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 z-50">
          <Link href="/" className="font-semibold tracking-wide text-background hover:text-cyan-600 transition-colors">
            React Bits
          </Link>
          <div className="flex gap-6 text-background/80">
            <Link
              href="/"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Home
            </Link>
            <Link
              href="/upload"
              className="hover:text-cyan-600 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md px-1"
            >
              Upload
              <span className="sr-only">current page</span>
            </Link>
          </div>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full px-6 py-20 flex flex-col items-center justify-center">
        <div className="w-full max-w-3xl">
          <h1 className="text-white/80 text-4xl md:text-5xl font-bold mb-3 leading-tight text-center">
            Transform Your Videos
          </h1>
          <p className="mx-auto max-w-2xl text-background/80 mb-12 leading-relaxed text-center">
            Upload a video or paste a link to instantly get transcripts, summaries, and notes powered by AI.
          </p>

          {/* Video Upload Form */}
          <VideoUploadForm />
        </div>
      </main>
    </div>
  )
}
