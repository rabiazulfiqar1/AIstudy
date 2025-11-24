"use client"

import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowRight } from "lucide-react"

interface ProjectCardProps {
  id: string
  title: string
  description: string
  difficulty: string
  source: string
  matchScore?: number
}

export function ProjectCard({ id, title, description, difficulty, source, matchScore }: ProjectCardProps) {
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

  return (
    <Link href={`/projects/${id}`}>
      <Card className="bg-[#0b1c24]/60 border-white/10 backdrop-blur-sm
                 hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10
                 transition-all cursor-pointer h-full">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <CardTitle className="text-white line-clamp-2">{title}</CardTitle>
              <CardDescription className="text-white/60 text-xs mt-1">{source}</CardDescription>
            </div>
{/* 
            {matchScore !== undefined && (
              <div className="text-right">
                <div className="text-2xl font-bold text-cyan-400">{matchScore}%</div>
                <p className="text-xs text-white/60">Match</p>
              </div>
            )} */}
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <p className="text-sm text-white/60 line-clamp-2">{description}</p>

          <div className="flex items-center justify-between">
            <Badge
              variant="default"
              className={`${getDifficultyColor(difficulty)}`}
            >
              {difficulty}
            </Badge>

            <ArrowRight className="h-4 w-4 text-cyan-400" />
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}