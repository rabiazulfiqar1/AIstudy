"use client"

import { Navbar } from "@/components/navbar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export default function InteractionsPage() {
  // Mock interaction data
  const interactions = [
    {
      id: 1,
      projectTitle: "Build a E-commerce Platform",
      type: "viewed",
      timestamp: "2 hours ago",
      projectId: "123",
    },
    {
      id: 2,
      projectTitle: "Machine Learning Model Optimization",
      type: "saved",
      timestamp: "5 hours ago",
      projectId: "124",
    },
    {
      id: 3,
      projectTitle: "Real-time Chat Application",
      type: "rated",
      timestamp: "1 day ago",
      projectId: "125",
      rating: 4,
    },
  ]

  const getInteractionBadgeColor = (type: string) => {
    switch (type) {
      case "viewed":
        return "bg-blue-900/30 text-blue-300 border-blue-900/50"
      case "saved":
        return "bg-green-900/30 text-green-300 border-green-900/50"
      case "rated":
        return "bg-purple-900/30 text-purple-300 border-purple-900/50"
      default:
        return "bg-gray-900/30 text-gray-300 border-gray-900/50"
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* 🩵 Background with Light Rays - Consistent with Home Page */}
      <div className="fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        {/* subtle radial vignette */}
        <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      <Navbar />

      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Interaction History</h1>
          <p className="text-white/70">Track all your project interactions and activity</p>
        </div>

        <Card className="bg-white/5 backdrop-blur-sm border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Recent Interactions</CardTitle>
            <CardDescription className="text-white/60">
              {interactions.length} interactions in total
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {interactions.length === 0 ? (
              <p className="text-center text-white/60 py-8">No interactions yet. Start exploring projects!</p>
            ) : (
              interactions.map((interaction) => (
                <div
                  key={interaction.id}
                  className="flex items-center justify-between p-4 rounded-lg border border-white/10 hover:border-cyan-500/50 transition-colors bg-white/5"
                >
                  <div className="flex-1">
                    <p className="text-white font-medium">{interaction.projectTitle}</p>
                    <p className="text-sm text-white/60 mt-1">{interaction.timestamp}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {interaction.rating && (
                      <span className="text-sm text-cyan-400 font-medium">⭐ {interaction.rating}/5</span>
                    )}
                    <Badge className={`${getInteractionBadgeColor(interaction.type)} border`}>{interaction.type}</Badge>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}