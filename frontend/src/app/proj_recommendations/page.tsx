"use client"

import { Navbar } from "@/components/navbar"
import Link from "next/link"
import { ArrowRight, Zap, Target, Users, Sparkles, BookOpen, Rocket } from "lucide-react"
import LightRays from "@/components/LightRays"

export default function ProjectRecommendations() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* 🩵 Background with Light Rays - Consistent with Home Page */}
      <div className="fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
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

      <Navbar />

      {/* Hero Section */}
      <section className="relative px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-white mb-6">
            Find Your Perfect Project
          </h1>
          <p className="text-xl text-white/70 mb-8">
            Discover projects tailored to your skills and interests with AI-powered matching. Connect with challenges
            that help you grow.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/projects/search"
              className="inline-flex items-center justify-center gap-2 px-8 py-3 bg-cyan-600 text-white rounded-lg font-semibold hover:bg-cyan-700 transition-colors"
            >
              Search Projects
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/recommendations"
              className="inline-flex items-center justify-center gap-2 px-8 py-3 border border-cyan-600 text-cyan-400 rounded-lg font-semibold hover:bg-cyan-950/30 transition-colors"
            >
              Get Recommendations
              <Zap className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Target,
                title: "Smart Matching",
                description: "AI analyzes your skills to find perfectly aligned projects",
              },
              {
                icon: Users,
                title: "Diverse Sources",
                description: "Access projects from GitHub, Kaggle, and more",
              },
              {
                icon: Zap,
                title: "Personalized",
                description: "Get recommendations based on your interests and experience",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="p-6 rounded-lg bg-white/5 backdrop-blur-sm border border-white/10 hover:border-cyan-500/50 transition-colors"
              >
                <feature.icon className="h-8 w-8 text-cyan-400 mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-white/60">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="px-4 py-20 sm:px-6 lg:px-8 border-t border-white/10">
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
            <Sparkles className="h-4 w-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-400">AI-Powered System</span>
          </div>

          <h2 className="text-4xl sm:text-5xl font-bold tracking-tight text-white mb-4">
            Discover Your Next Challenge
          </h2>

          <p className="text-lg text-white/70 mb-8 max-w-2xl mx-auto">
            Our intelligent recommender system learns from your profile, analyzes project requirements, and matches you
            with opportunities that align with your growth goals.
          </p>

          <div className="grid md:grid-cols-3 gap-6 mb-8">
            {[
              {
                icon: BookOpen,
                title: "Smart Analysis",
                description: "Advanced algorithms match your skills with project needs",
              },
              {
                icon: Rocket,
                title: "Growth Focused",
                description: "Find projects that challenge and develop your abilities",
              },
              {
                icon: Target,
                title: "Result Driven",
                description: "Track your progress and celebrate your achievements",
              },
            ].map((item) => (
              <div key={item.title} className="p-4">
                <item.icon className="h-6 w-6 text-cyan-400 mb-3 mx-auto" />
                <h3 className="font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-sm text-white/60">{item.description}</p>
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/recommendations"
              className="inline-flex items-center justify-center gap-2 px-8 py-3 bg-cyan-600 text-white rounded-lg font-semibold hover:bg-cyan-700 transition-colors"
            >
              Get Personalized Recommendations
              <Sparkles className="h-4 w-4" />
            </Link>
            <Link
              href="/projects/search"
              className="inline-flex items-center justify-center gap-2 px-8 py-3 border border-white/20 text-white rounded-lg font-semibold hover:bg-white/5 transition-colors"
            >
              Browse All Projects
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}