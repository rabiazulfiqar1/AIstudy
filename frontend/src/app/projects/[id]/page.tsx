"use client"

import { use, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { MatchAnalysis } from "@/components/match-analysis"
import { InteractionWidget } from "@/components/interaction-widget"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ExternalLink, Loader2 } from "lucide-react"
import Link from "next/link"
import { getProjectDetail } from "@/lib/api"
import { supabase } from "@/lib/supabaseClient"

interface ProjectDetail {
  id: number
  title: string
  description: string
  repo_url?: string
  difficulty: string
  topics: string[]
  estimated_hours?: number
  source: string
  stars?: number
  language?: string
  skills: Array<{
    name: string
    category: string
    is_required: boolean
  }>
  match_analysis?: {
    score: number
    matching_skills: string[]
    missing_skills: string[]
    reason: string
    semantic_similarity: number
  }
}

export default function ProjectDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const actualParams = use(params)
  const projectId = actualParams.id
  const router = useRouter()
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [userId, setUserId] = useState<string | undefined>(undefined)

  // Get authenticated user
  useEffect(() => {
    async function getUser() {
      try {
        const {
          data: { user },
        } = await supabase.auth.getUser()
        setUserId(user?.id)
      } catch (err) {
        console.error("Error getting user:", err)
      }
    }
    getUser()
  }, [])

  useEffect(() => {
    async function fetchProject() {
      try {
        setLoading(true)
        setError("")

        const data = await getProjectDetail(projectId, userId)
        setProject(data)
      } catch (err) {
        console.error("Error fetching project:", err)
        setError(err instanceof Error ? err.message : "Failed to load project")
      } finally {
        setLoading(false)
      }
    }

    fetchProject()
  }, [projectId, userId])

  const getDifficultyColor = (diff: string) => {
    switch (diff.toLowerCase()) {
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

  const getSourceDisplay = (source: string) => {
    const sourceMap: Record<string, string> = {
      github: "GitHub",
      kaggle_competition: "Kaggle Competition",
      kaggle_dataset: "Kaggle Dataset",
      curated: "Curated",
      manual: "Manual",
    }
    return sourceMap[source] || source
  }

  if (loading) {
    return (
      <div className="min-h-screen">
        <div className="min-h-screen fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
          <div className="fixed inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
          <div className="fixed inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
        </div>
        <Navbar />
        <div className="flex items-center justify-center min-h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  if (error || !project) {
    return (
      <div className="min-h-screen">
        <div className="min-h-screen fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
          <div className="fixed inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
          <div className="fixed inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
        </div>
        <Navbar />
        <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
          <Card className="bg-card border-border">
            <CardContent className="pt-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-white mb-2">{error || "Project not found"}</h2>
                <p className="text-white mb-4">
                  The project you&apos;re looking for doesn&apos;t exist or couldn&apos;t be loaded.
                </p>
                <button
                  onClick={() => router.push("/projects/search")}
                  className="text-primary hover:text-primary/80 font-medium"
                >
                  ← Back to search
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const requiredSkills = project.skills.filter((s) => s.is_required).map((s) => s.name)
  const optionalSkills = project.skills.filter((s) => !s.is_required).map((s) => s.name)

  return (
    <div className="min-h-screen">
      <div className="min-h-screen fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="fixed inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        <div className="fixed inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      <Navbar />

      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div className="flex-1">
              <h1 className="text-4xl font-bold text-white mb-2">{project.title}</h1>
              <div className="flex items-center gap-3 flex-wrap">
                <Badge className={`${getDifficultyColor(project.difficulty)} border`}>{project.difficulty}</Badge>
                <span className="text-muted-foreground text-sm">{getSourceDisplay(project.source)}</span>
                {project.estimated_hours && (
                  <span className="text-muted-foreground text-sm">~{project.estimated_hours} hours</span>
                )}
                {project.stars && project.stars > 0 && (
                  <span className="text-muted-foreground text-sm">⭐ {project.stars}</span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Overview */}
            <Card className="bg-[#0f1419]/80 border-[#1a2332]/60 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Overview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-white text-base leading-relaxed whitespace-pre-wrap">{project.description}</p>

                {/* Topics */}
                {project.topics && project.topics.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-white mb-3">Topics</h4>
                    <div className="flex flex-wrap gap-2">
                      {project.topics.map((topic) => (
                        <Badge
                          key={topic}
                          variant="outline"
                          className="bg-[#1a2332]/40 text-cyan-300 border-[#1a2332]/60"
                        >
                          {topic}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Required Skills */}
                {requiredSkills.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-white mb-3">Required Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {requiredSkills.map((skill) => (
                        <Badge key={skill} className="bg-cyan-600 text-white border-0">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Optional Skills */}
                {optionalSkills.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-white mb-3">Optional Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {optionalSkills.map((skill) => (
                        <Badge
                          key={skill}
                          variant="outline"
                          className="bg-[#1a2332]/40 text-emerald-300 border-[#1a2332]/60"
                        >
                          {skill}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Source Link */}
                {project.repo_url && (
                  <Link
                    href={project.repo_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
                  >
                    View on {getSourceDisplay(project.source)}
                    <ExternalLink className="h-4 w-4" />
                  </Link>
                )}
              </CardContent>
            </Card>

            {/* Match Analysis */}
            {project.match_analysis && (
              <MatchAnalysis
                matchScore={project.match_analysis.score}
                matchingSkills={project.match_analysis.matching_skills}
                missingSkills={project.match_analysis.missing_skills}
              />
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <InteractionWidget projectId={project.id.toString()} />

            {/* Project Stats */}
            <Card className="bg-[#0f1419]/80 border-[#1a2332]/60 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white text-lg">Project Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-sm text-slate-400">Difficulty Level</p>
                  <p className="font-semibold text-white capitalize">{project.difficulty}</p>
                </div>
                {project.estimated_hours && (
                  <div>
                    <p className="text-sm text-slate-400">Estimated Time</p>
                    <p className="font-semibold text-white">{project.estimated_hours} hours</p>
                  </div>
                )}
                <div>
                  <p className="text-sm text-slate-400">Source Platform</p>
                  <p className="font-semibold text-white">{getSourceDisplay(project.source)}</p>
                </div>
                {project.language && (
                  <div>
                    <p className="text-sm text-slate-400">Primary Language</p>
                    <p className="font-semibold text-white">{project.language}</p>
                  </div>
                )}
                {project.stars && project.stars > 0 && (
                  <div>
                    <p className="text-sm text-slate-400">GitHub Stars</p>
                    <p className="font-semibold text-foreground">{project.stars.toLocaleString()}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
