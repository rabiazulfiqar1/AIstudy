"use client"

import { useState, useEffect } from "react"
import { supabase } from "@/lib/supabaseClient"
import { Navbar } from "@/components/navbar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Star, ExternalLink, Trash2, Clock, BookMarked, Eye, Play, CheckCircle } from "lucide-react"

interface Interaction {
  id: number
  project_id: number
  project_title: string
  project_description: string
  difficulty: string
  topics: string[]
  repo_url: string | null
  estimated_hours: number
  source: string
  interaction_type: string
  rating: number | null
  timestamp: string
}

interface Stats {
  total_interactions: number
  by_type: {
    viewed?: number
    bookmarked?: number
    started?: number
    completed?: number
  }
  average_rating: number | null
  recent_activity_30d: number
}

export default function InteractionsPage() {
  const [interactions, setInteractions] = useState<Interaction[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeFilter, setActiveFilter] = useState<string>("all")
  const [userId, setUserId] = useState<string | null>(null)

  // Get authenticated user
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser()
        
        if (!user) {
          console.error("No authenticated user found")
          setLoading(false)
          return
        }

        setUserId(user.id)
      } catch (error) {
        console.error("Error fetching user:", error)
        setLoading(false)
      }
    }

    fetchUser()
  }, [])

  // Fetch interactions when user or filter changes
  useEffect(() => {
    if (userId) {
      fetchInteractions()
      fetchStats()
    }
  }, [userId, activeFilter])

  const fetchInteractions = async () => {
    setLoading(true)
    try {
      const filterParam = activeFilter !== "all" ? `?interaction_type=${activeFilter}` : ""
      const response = await fetch(`http://localhost:8000/api/interactions/${userId}${filterParam}`)
      const data = await response.json()
      setInteractions(data.interactions)
    } catch (error) {
      console.error("Failed to fetch interactions:", error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/interactions/${userId}/stats`)
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error("Failed to fetch stats:", error)
    }
  }

  const deleteInteraction = async (interactionId: number) => {
    try {
      await fetch(`http://localhost:8000/api/interactions/${interactionId}`, {
        method: "DELETE",
      })
      // Refresh interactions after deletion
      fetchInteractions()
      fetchStats()
    } catch (error) {
      console.error("Failed to delete interaction:", error)
    }
  }

  const getInteractionIcon = (type: string) => {
    switch (type) {
      case "viewed":
        return <Eye className="h-4 w-4" />
      case "bookmarked":
        return <BookMarked className="h-4 w-4" />
      case "started":
        return <Play className="h-4 w-4" />
      case "completed":
        return <CheckCircle className="h-4 w-4" />
      default:
        return null
    }
  }

  const getInteractionBadgeColor = (type: string) => {
    switch (type) {
      case "viewed":
        return "bg-blue-900/30 text-blue-300 border-blue-900/50"
      case "bookmarked":
        return "bg-green-900/30 text-green-300 border-green-900/50"
      case "started":
        return "bg-purple-900/30 text-purple-300 border-purple-900/50"
      case "completed":
        return "bg-cyan-900/30 text-cyan-300 border-cyan-900/50"
      default:
        return "bg-gray-900/30 text-gray-300 border-gray-900/50"
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty?.toLowerCase()) {
      case "beginner":
        return "bg-green-900/30 text-green-300 border-green-900/50"
      case "intermediate":
        return "bg-yellow-900/30 text-yellow-300 border-yellow-900/50"
      case "advanced":
        return "bg-red-900/30 text-red-300 border-red-900/50"
      default:
        return "bg-gray-900/30 text-gray-300 border-gray-900/50"
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffInMs = now.getTime() - date.getTime()
    const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60))
    
    if (diffInHours < 1) return "Just now"
    if (diffInHours < 24) return `${diffInHours}h ago`
    if (diffInHours < 48) return "Yesterday"
    if (diffInHours < 168) return `${Math.floor(diffInHours / 24)}d ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      <Navbar />

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Interaction History</h1>
          <p className="text-white/70">Track all your project interactions and activity</p>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card className="bg-white/5 backdrop-blur-sm border-white/10">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-cyan-400">{stats.total_interactions}</div>
                <p className="text-white/60 text-sm mt-1">Total Interactions</p>
              </CardContent>
            </Card>
            <Card className="bg-white/5 backdrop-blur-sm border-white/10">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-green-400">{stats.by_type.bookmarked || 0}</div>
                <p className="text-white/60 text-sm mt-1">Bookmarked</p>
              </CardContent>
            </Card>
            <Card className="bg-white/5 backdrop-blur-sm border-white/10">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-purple-400">{stats.by_type.started || 0}</div>
                <p className="text-white/60 text-sm mt-1">Started</p>
              </CardContent>
            </Card>
            <Card className="bg-white/5 backdrop-blur-sm border-white/10">
              <CardContent className="pt-6">
                <div className="text-3xl font-bold text-cyan-400">{stats.by_type.completed || 0}</div>
                <p className="text-white/60 text-sm mt-1">Completed</p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Filters */}
        <Tabs value={activeFilter} onValueChange={setActiveFilter} className="mb-6">
          <TabsList className="bg-white/5 backdrop-blur-sm border-white/10">
            <TabsTrigger value="all" className="data-[state=active]:bg-cyan-500/20">All</TabsTrigger>
            <TabsTrigger value="viewed" className="data-[state=active]:bg-cyan-500/20">Viewed</TabsTrigger>
            <TabsTrigger value="bookmarked" className="data-[state=active]:bg-cyan-500/20">Bookmarked</TabsTrigger>
            <TabsTrigger value="started" className="data-[state=active]:bg-cyan-500/20">Started</TabsTrigger>
            <TabsTrigger value="completed" className="data-[state=active]:bg-cyan-500/20">Completed</TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Interactions List */}
        <Card className="bg-white/5 backdrop-blur-sm border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Recent Interactions</CardTitle>
            <CardDescription className="text-white/60">
              {interactions.length} interaction{interactions.length !== 1 ? 's' : ''} found
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {loading ? (
              <div className="text-center text-white/60 py-8">Loading interactions...</div>
            ) : interactions.length === 0 ? (
              <div className="text-center text-white/60 py-8">
                No interactions yet. Start exploring projects!
              </div>
            ) : (
              interactions.map((interaction) => (
                <div
                  key={interaction.id}
                  className="p-4 rounded-lg border border-white/10 hover:border-cyan-500/50 transition-all bg-white/5 hover:bg-white/10"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <h3 className="text-white font-semibold text-lg">{interaction.project_title}</h3>
                        {interaction.repo_url && (
                          <a
                            href={interaction.repo_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-cyan-400 hover:text-cyan-300"
                          >
                            <ExternalLink className="h-4 w-4" />
                          </a>
                        )}
                      </div>
                      
                      <p className="text-white/70 text-sm line-clamp-2">{interaction.project_description}</p>
                      
                      <div className="flex flex-wrap gap-2">
                        <Badge className={getDifficultyColor(interaction.difficulty)}>{interaction.difficulty}</Badge>
                        <Badge className="bg-white/10 text-white/80 border-white/20">
                          <Clock className="h-3 w-3 mr-1" />
                          {interaction.estimated_hours}h
                        </Badge>
                        {interaction.topics.slice(0, 3).map((topic) => (
                          <Badge key={topic} variant="outline" className="bg-white/5 text-white/70 border-white/20">
                            {topic}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="flex flex-col items-end gap-2">
                      <Badge className={`${getInteractionBadgeColor(interaction.interaction_type)} border flex items-center gap-1`}>
                        {getInteractionIcon(interaction.interaction_type)}
                        {interaction.interaction_type}
                      </Badge>
                      {interaction.rating && (
                        <div className="flex items-center gap-1 text-yellow-400">
                          <Star className="h-4 w-4 fill-current" />
                          <span className="text-sm font-medium">{interaction.rating}/5</span>
                        </div>
                      )}
                      <span className="text-xs text-white/50">{formatTimestamp(interaction.timestamp)}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteInteraction(interaction.id)}
                        className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
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