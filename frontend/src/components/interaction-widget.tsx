"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Eye, Bookmark, Play, CheckCircle } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { supabase } from "@/lib/supabaseClient"

interface InteractionWidgetProps {
  projectId: number
}

interface InteractionData {
  id: number
  project_id: number
  interaction_type: string
  rating: number | null
}

export function InteractionWidget({ projectId }: InteractionWidgetProps) {
  const [userId, setUserId] = useState<string | null>(null)
  const [interactions, setInteractions] = useState({
    viewed: { active: false, id: null as number | null },
    bookmarked: { active: false, id: null as number | null },
    started: { active: false, id: null as number | null },
    completed: { active: false, id: null as number | null },
    rating: 0,
    ratingInteractionId: null as number | null,
  })
  const [loading, setLoading] = useState(false)

  // Simple toast function (you can replace with a proper toast library later)
  const showToast = (title: string, description?: string) => {
    console.log(`Toast: ${title}`, description)
    // You can implement a custom toast notification here or install a toast library
  }

  // Get authenticated user
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser()
        if (user) {
          setUserId(user.id)
        }
      } catch (error) {
        console.error("Error fetching user:", error)
      }
    }
    fetchUser()
  }, [])

  // Fetch existing interactions when component mounts
  const fetchExistingInteractions = useCallback(async () => {
    if (!userId) return

    try {
      const response = await fetch(`http://localhost:8000/api/interactions/${userId}`)
      const data = await response.json()
      
      // Filter interactions for this specific project
      const projectInteractions: InteractionData[] = data.interactions.filter(
        (int: InteractionData) => int.project_id === projectId
      )

      // Update state based on existing interactions
      const newState = {
        viewed: { active: false, id: null as number | null },
        bookmarked: { active: false, id: null as number | null },
        started: { active: false, id: null as number | null },
        completed: { active: false, id: null as number | null },
        rating: 0,
        ratingInteractionId: null as number | null,
      }

      projectInteractions.forEach((int: InteractionData) => {
        const type = int.interaction_type as keyof typeof newState
        if (type in newState && type !== 'rating' && type !== 'ratingInteractionId') {
          newState[type] = { active: true, id: int.id }
        }
        if (int.rating) {
          newState.rating = int.rating
          newState.ratingInteractionId = int.id
        }
      })

      setInteractions(newState)
    } catch (error) {
      console.error("Failed to fetch existing interactions:", error)
    }
  }, [userId, projectId])

  useEffect(() => {
    if (userId) {
      fetchExistingInteractions()
    }
  }, [userId, fetchExistingInteractions])

  const handleInteraction = async (type: "viewed" | "bookmarked" | "started" | "completed") => {
    if (!userId) {
      showToast("Authentication required", "Please log in to interact with projects")
      return
    }

    setLoading(true)

    try {
      const currentInteraction = interactions[type]

      if (currentInteraction.active && currentInteraction.id) {
        // Delete existing interaction
        const response = await fetch(
          `http://localhost:8000/api/interactions/${currentInteraction.id}`,
          { method: "DELETE" }
        )

        if (response.ok) {
          setInteractions((prev) => ({
            ...prev,
            [type]: { active: false, id: null },
          }))
          showToast("Removed", `Project ${type} status removed`)
        }
      } else {
        // Create new interaction
        const response = await fetch(
          `http://localhost:8000/api/interactions?user_id=${userId}&project_id=${projectId}&interaction_type=${type}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          }
        )

        if (response.ok) {
          const data = await response.json()
          setInteractions((prev) => ({
            ...prev,
            [type]: { active: true, id: data.id },
          }))
          showToast("Success", `Project marked as ${type}`)
        }
      }
    } catch (error) {
      console.error("Failed to update interaction:", error)
      showToast("Error", "Failed to update interaction")
    } finally {
      setLoading(false)
    }
  }

  const handleRating = async (star: number) => {
    if (!userId) {
      showToast("Authentication required", "Please log in to rate projects")
      return
    }

    setLoading(true)

    try {
      const newRating = interactions.rating === star ? 0 : star

      if (interactions.ratingInteractionId) {
        // Update existing rating
        if (newRating === 0) {
          // Remove rating if user clicks same star
          const response = await fetch(
            `http://localhost:8000/api/interactions/${interactions.ratingInteractionId}?rating=null`,
            { method: "PUT" }
          )
          if (response.ok) {
            setInteractions((prev) => ({ ...prev, rating: 0 }))
            showToast("Rating removed")
          }
        } else {
          const response = await fetch(
            `http://localhost:8000/api/interactions/${interactions.ratingInteractionId}?rating=${newRating}`,
            { method: "PUT" }
          )
          if (response.ok) {
            setInteractions((prev) => ({ ...prev, rating: newRating }))
            showToast("Rating updated", `You rated this ${newRating}/5`)
          }
        }
      } else if (newRating > 0) {
        // Create new interaction with rating
        const response = await fetch(
          `http://localhost:8000/api/interactions?user_id=${userId}&project_id=${projectId}&interaction_type=viewed&rating=${newRating}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          }
        )

        if (response.ok) {
          const data = await response.json()
          setInteractions((prev) => ({
            ...prev,
            rating: newRating,
            ratingInteractionId: data.id,
          }))
          showToast("Rating added", `You rated this ${newRating}/5`)
        }
      }
    } catch (error) {
      console.error("Failed to update rating:", error)
      showToast("Error", "Failed to update rating")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="bg-white/5 backdrop-blur-sm border-white/10">
      <CardContent className="pt-6 space-y-4">
        {/* Viewed */}
        <Button
          onClick={() => handleInteraction("viewed")}
          disabled={loading}
          variant="outline"
          className={`w-full justify-start gap-2 transition-all ${
            interactions.viewed.active
              ? "bg-blue-900/30 border-blue-700/50 text-blue-300 hover:bg-blue-900/40"
              : "border-blue-700/30 text-foreground hover:bg-blue-900/20 hover:border-blue-700/50 hover:text-blue-300"
          }`}
        >
          <Eye className="h-4 w-4" />
          {interactions.viewed.active ? "Marked as Viewed" : "Mark as Viewed"}
        </Button>

        {/* Bookmarked */}
        <Button
          onClick={() => handleInteraction("bookmarked")}
          disabled={loading}
          variant="outline"
          className={`w-full justify-start gap-2 transition-all ${
            interactions.bookmarked.active
              ? "bg-green-900/30 border-green-700/50 text-green-300 hover:bg-green-900/40"
              : "border-green-700/30 text-foreground hover:bg-green-900/20 hover:border-green-700/50 hover:text-green-300"
          }`}
        >
          <Bookmark className="h-4 w-4" />
          {interactions.bookmarked.active ? "Bookmarked" : "Bookmark Project"}
        </Button>

        {/* Started */}
        <Button
          onClick={() => handleInteraction("started")}
          disabled={loading}
          variant="outline"
          className={`w-full justify-start gap-2 transition-all ${
            interactions.started.active
              ? "bg-purple-900/30 border-purple-700/50 text-purple-300 hover:bg-purple-900/40"
              : "border-purple-700/30 text-foreground hover:bg-purple-900/20 hover:border-purple-700/50 hover:text-purple-300"
          }`}
        >
          <Play className="h-4 w-4" />
          {interactions.started.active ? "Started" : "Mark as Started"}
        </Button>

        {/* Completed */}
        <Button
          onClick={() => handleInteraction("completed")}
          disabled={loading}
          variant="outline"
          className={`w-full justify-start gap-2 transition-all ${
            interactions.completed.active
              ? "bg-cyan-900/30 border-cyan-700/50 text-cyan-300 hover:bg-cyan-900/40"
              : "border-cyan-700/30 text-foreground hover:bg-cyan-900/20 hover:border-cyan-700/50 hover:text-cyan-300"
          }`}
        >
          <CheckCircle className="h-4 w-4" />
          {interactions.completed.active ? "Completed" : "Mark as Completed"}
        </Button>

        {/* Rating */}
        <div className="space-y-2 pt-2 border-t border-white/10">
          <p className="text-sm font-medium text-white">Rate This Project</p>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => handleRating(star)}
                disabled={loading}
                className={`text-2xl transition-all hover:scale-110 disabled:opacity-50 ${
                  star <= interactions.rating ? "text-yellow-400" : "text-white/20 hover:text-yellow-400/50"
                }`}
              >
                ⭐
              </button>
            ))}
          </div>
          {interactions.rating > 0 && (
            <p className="text-xs text-white/60">You rated this {interactions.rating}/5</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}