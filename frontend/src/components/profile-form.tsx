"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Plus, X } from "lucide-react"

interface Skill {
  id: string
  name: string
  proficiency: number
}

interface ProfileFormData {
  bio?: string
  skillLevel?: string
  interests?: string[]
  githubUsername?: string
  preferredProjectTypes?: string[]
  skills?: Skill[]
}

interface ProfileFormProps {
  initialData?: ProfileFormData
  onSubmit: (data: ProfileFormData) => void
}

export function ProfileForm({ initialData, onSubmit }: ProfileFormProps) {
  const [bio, setBio] = useState(initialData?.bio || "")
  const [skillLevel, setSkillLevel] = useState(initialData?.skillLevel || "intermediate")
  const [interests, setInterests] = useState<string[]>(initialData?.interests || [])
  const [newInterest, setNewInterest] = useState("")
  const [githubUsername, setGithubUsername] = useState(initialData?.githubUsername || "")
  const [skills, setSkills] = useState<Skill[]>(initialData?.skills || [])
  const [newSkill, setNewSkill] = useState({ name: "", proficiency: 3 })

  const handleAddInterest = () => {
    if (newInterest.trim() && !interests.includes(newInterest)) {
      setInterests([...interests, newInterest])
      setNewInterest("")
    }
  }

  const handleRemoveInterest = (interest: string) => {
    setInterests(interests.filter((i) => i !== interest))
  }

  const handleAddSkill = () => {
    if (newSkill.name.trim()) {
      setSkills([...skills, { ...newSkill, id: Date.now().toString() }])
      setNewSkill({ name: "", proficiency: 3 })
    }
  }

  const handleRemoveSkill = (id: string) => {
    setSkills(skills.filter((s) => s.id !== id))
  }

  const handleUpdateSkillProficiency = (id: string, proficiency: number) => {
    setSkills(skills.map((s) => (s.id === id ? { ...s, proficiency } : s)))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      bio,
      skillLevel,
      interests,
      githubUsername,
      skills,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Bio */}
      <div>
        <Label htmlFor="bio" className="text-white">
          Bio
        </Label>
        <Textarea
          id="bio"
          placeholder="Tell us about yourself..."
          value={bio}
          onChange={(e) => setBio(e.target.value)}
          className="mt-2 min-h-24 bg-white/5 border-white/10 text-white placeholder:text-white/40"
        />
      </div>

      {/* Skill Level */}
      <div>
        <Label htmlFor="skillLevel" className="text-white">
          Skill Level
        </Label>
        <select
          id="skillLevel"
          value={skillLevel}
          onChange={(e) => setSkillLevel(e.target.value)}
          className="mt-2 w-full px-3 py-2 bg-white/5 border border-white/10 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>

      {/* GitHub Username */}
      <div>
        <Label htmlFor="github" className="text-white">
          GitHub Username
        </Label>
        <Input
          id="github"
          placeholder="your-username"
          value={githubUsername}
          onChange={(e) => setGithubUsername(e.target.value)}
          className="mt-2 bg-white/5 border-white/10 text-white placeholder:text-white/40"
        />
      </div>

      {/* Interests */}
      <div>
        <Label className="text-white mb-2 block">Interests</Label>
        <div className="flex gap-2 mb-3">
          <Input
            placeholder="Add an interest..."
            value={newInterest}
            onChange={(e) => setNewInterest(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), handleAddInterest())}
            className="bg-white/5 border-white/10 text-white placeholder:text-white/40"
          />
          <Button
            type="button"
            onClick={handleAddInterest}
            className="bg-cyan-600 text-white hover:bg-cyan-700"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex flex-wrap gap-2">
          {interests.map((interest) => (
            <div
              key={interest}
              className="px-3 py-1 rounded-full bg-cyan-600 text-white flex items-center gap-2"
            >
              {interest}
              <button type="button" onClick={() => handleRemoveInterest(interest)} className="hover:opacity-80">
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Skills */}
      <div>
        <Label className="text-white mb-2 block">Skills</Label>
        <div className="space-y-2 mb-3 p-3 bg-white/5 rounded-lg border border-white/10">
          {skills.map((skill) => (
            <div key={skill.id} className="flex items-center gap-3 justify-between">
              <span className="text-sm text-white font-medium">{skill.name}</span>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={skill.proficiency}
                  onChange={(e) => handleUpdateSkillProficiency(skill.id, Number.parseInt(e.target.value))}
                  className="w-24"
                />
                <span className="text-xs text-white/60 w-8">{skill.proficiency}/5</span>
                <button
                  type="button"
                  onClick={() => handleRemoveSkill(skill.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <Input
            placeholder="Skill name..."
            value={newSkill.name}
            onChange={(e) => setNewSkill({ ...newSkill, name: e.target.value })}
            className="bg-white/5 border-white/10 text-white placeholder:text-white/40"
          />
          <Button
            type="button"
            onClick={handleAddSkill}
            className="bg-cyan-600 text-white hover:bg-cyan-700"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Button type="submit" className="w-full bg-cyan-600 text-white hover:bg-cyan-700">
        Save Profile
      </Button>
    </form>
  )
}