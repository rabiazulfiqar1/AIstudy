"use client"

import { useEffect, useState } from "react"
import { Navbar } from "@/components/navbar"
import { RecommendationFilters } from "@/components/recommendation-filters"
import { RecommendationCard } from "@/components/recommendation-card"
import { Card, CardContent } from "@/components/ui/card"
import { getRecommendations } from "@/lib/api"

export default function RecommendationsPage() {
  const [algorithm, setAlgorithm] = useState("hybrid")
  const [sourceFilter, setSourceFilter] = useState("All")
  const [recommendations, setRecommendations] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [userId] = useState("default-user") // In production, get from auth context

  const loadRecommendations = async () => {
    setIsLoading(true)
    setError("")
    console.log("[v0] Loading recommendations with:", { algorithm, sourceFilter })

    try {
      const recs = await getRecommendations(
        userId,
        algorithm as "hybrid" | "semantic" | "traditional",
        sourceFilter,
        10,
      )
      setRecommendations(recs)
    } catch (err) {
      console.error("[v0] Recommendations error:", err)
      setError("Failed to load recommendations. Please try again.")
      setRecommendations([])
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadRecommendations()
  }, [algorithm, sourceFilter])

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Personalized Recommendations</h1>
          <p className="text-muted-foreground">Projects curated specifically for your skills and interests</p>
        </div>

        <RecommendationFilters
          algorithm={algorithm}
          setAlgorithm={setAlgorithm}
          sourceFilter={sourceFilter}
          setSourceFilter={setSourceFilter}
        />

        <div className="mt-8 grid md:grid-cols-3 gap-4">
          <Card className="bg-card border-border">
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Algorithm</p>
              <p className="text-2xl font-bold text-primary capitalize mt-1">{algorithm}</p>
            </CardContent>
          </Card>
          <Card className="bg-card border-border">
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Total Recommendations</p>
              <p className="text-2xl font-bold text-primary mt-1">{recommendations.length}</p>
            </CardContent>
          </Card>
          <Card className="bg-card border-border">
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Avg Match Score</p>
              <p className="text-2xl font-bold text-primary mt-1">
                {recommendations.length > 0
                  ? Math.round(recommendations.reduce((sum, r) => sum + r.match_score, 0) / recommendations.length)
                  : 0}
                %
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8">
          {error && (
            <div className="mb-4 p-4 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
              {error}
            </div>
          )}
          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">Loading recommendations...</p>
            </div>
          ) : recommendations.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                No recommendations available. Complete your profile to get personalized suggestions.
              </p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((rec) => (
                <RecommendationCard
                  key={rec.project_id}
                  id={rec.project_id}
                  title={rec.title}
                  description={rec.description}
                  matchScore={rec.match_score}
                  reason={rec.reason}
                  difficulty={rec.difficulty}
                  source={rec.source}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
