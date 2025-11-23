"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Eye, Bookmark } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

interface InteractionWidgetProps {
  projectId: string
}

export function InteractionWidget({ projectId }: InteractionWidgetProps) {
  const [interactions, setInteractions] = useState({
    viewed: false,
    saved: false,
    rating: 0,
  })

  const handleInteraction = (type: "viewed" | "saved" | "rating", value?: number) => {
    console.log("[v0] Logging interaction:", { projectId, type, value })

    if (type === "rating") {
      setInteractions((prev) => ({
        ...prev,
        rating: prev.rating === value ? 0 : (value ?? 0),
      }))
    } else {
      setInteractions((prev) => ({
        ...prev,
        [type]: !prev[type],
      }))
    }
  }

  return (
    <Card>
      <CardContent className="pt-6 space-y-4">
        {/* Viewed */}
        <Button
          onClick={() => handleInteraction("viewed")}
          variant="outline"
          className={`w-full justify-start gap-2 ${
            interactions.viewed
              ? "bg-cyan-900/30 border-cyan-700/50 text-cyan-300 hover:bg-cyan-900/40"
              : "border-cyan-700/30 text-foreground hover:bg-cyan-900/20 hover:border-cyan-700/50 hover:text-cyan-300"
          }`}
        >
          <Eye className="h-4 w-4" />
          {interactions.viewed ? "Marked as Viewed" : "Mark as Viewed"}
        </Button>

        {/* Saved */}
        <Button
          onClick={() => handleInteraction("saved")}
          variant="outline"
          className={`w-full justify-start gap-2 ${
            interactions.saved
              ? "bg-emerald-900/30 border-emerald-700/50 text-emerald-300 hover:bg-emerald-900/40"
              : "border-emerald-700/30 text-foreground hover:bg-emerald-900/20 hover:border-emerald-700/50 hover:text-emerald-300"
          }`}
        >
          <Bookmark className="h-4 w-4" />
          {interactions.saved ? "Saved" : "Save Project"}
        </Button>

        {/* Rating */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-white">Rate This Project</p>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => handleInteraction("rating", star)}
                className={`text-2xl transition-all ${
                  star <= interactions.rating ? "text-yellow-400" : "text-muted-foreground"
                }`}
              >
                ⭐
              </button>
            ))}
          </div>
          {interactions.rating > 0 && (
            <p className="text-xs text-muted-foreground">You rated this {interactions.rating}/5</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
