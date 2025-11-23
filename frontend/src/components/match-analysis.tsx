"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, AlertCircle } from "lucide-react"

interface MatchAnalysisProps {
  matchScore: number
  matchingSkills: string[]
  missingSkills: string[]
}

export function MatchAnalysis({ matchScore, matchingSkills, missingSkills }: MatchAnalysisProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-white">Match Analysis</CardTitle>
        <CardDescription className="text-muted-foreground">How well this project fits your profile</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Match Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-white font-medium">Overall Match Score</p>
            <span className="text-3xl font-bold text-cyan-400">{matchScore}%</span>
          </div>
          <div className="w-full bg-[#1a2332]/60 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-cyan-500 to-cyan-400 h-2 rounded-full transition-all"
              style={{ width: `${matchScore}%` }}
            />
          </div>
        </div>

        {/* Matching Skills */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="h-5 w-5 text-emerald-400" />
            <h4 className="font-semibold text-white">Matching Skills</h4>
            <span className="text-sm text-muted-foreground">({matchingSkills.length})</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {matchingSkills.length === 0 ? (
              <p className="text-sm text-muted-foreground">No matching skills yet</p>
            ) : (
              matchingSkills.map((skill) => (
                <Badge key={skill} className="bg-emerald-900/30 text-emerald-300 border-emerald-700/50 border">
                  {skill}
                </Badge>
              ))
            )}
          </div>
        </div>

        {/* Missing Skills */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle className="h-5 w-5 text-amber-400" />
            <h4 className="font-semibold text-white">Skills to Learn</h4>
            <span className="text-sm text-muted-foreground">({missingSkills.length})</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {missingSkills.length === 0 ? (
              <p className="text-sm text-muted-foreground">You have all required skills!</p>
            ) : (
              missingSkills.map((skill) => (
                <Badge key={skill} className="bg-amber-900/30 text-amber-300 border-amber-700/50 border">
                  {skill}
                </Badge>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
