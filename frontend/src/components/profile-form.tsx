"use client"

import React, { useState } from "react"
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

// ✅ ACTUAL SKILLS FROM YOUR DATABASE (IDs 1-26)
const AVAILABLE_SKILLS = [
  // Languages (1-5)
  { id: "1", name: "Python", category: "language" },
  { id: "2", name: "JavaScript", category: "language" },
  { id: "3", name: "TypeScript", category: "language" },
  { id: "4", name: "Java", category: "language" },
  { id: "5", name: "Go", category: "language" },
  
  // Frontend (6-9)
  { id: "6", name: "React", category: "framework" },
  { id: "7", name: "Next.js", category: "framework" },
  { id: "8", name: "Vue.js", category: "framework" },
  { id: "9", name: "HTML/CSS", category: "framework" },
  
  // Backend (10-13)
  { id: "10", name: "Node.js", category: "framework" },
  { id: "11", name: "FastAPI", category: "framework" },
  { id: "12", name: "Django", category: "framework" },
  { id: "13", name: "Flask", category: "framework" },
  
  // Databases (14-16)
  { id: "14", name: "PostgreSQL", category: "tool" },
  { id: "15", name: "MongoDB", category: "tool" },
  { id: "16", name: "Redis", category: "tool" },
  
  // ML/AI (17-20)
  { id: "17", name: "Machine Learning", category: "domain" },
  { id: "18", name: "Deep Learning", category: "domain" },
  { id: "19", name: "NLP", category: "domain" },
  { id: "20", name: "Computer Vision", category: "domain" },
  
  // DevOps (21-24)
  { id: "21", name: "Docker", category: "tool" },
  { id: "22", name: "Kubernetes", category: "tool" },
  { id: "23", name: "AWS", category: "tool" },
  { id: "24", name: "Git", category: "tool" },
  
  // Other (25-26)
  { id: "25", name: "API Development", category: "domain" },
  { id: "26", name: "Web Development", category: "domain" },
]

export function ProfileForm({ initialData, onSubmit }: ProfileFormProps) {
  const [bio, setBio] = useState(initialData?.bio || "")
  const [skillLevel, setSkillLevel] = useState(initialData?.skillLevel || "intermediate")
  const [interests, setInterests] = useState<string[]>(initialData?.interests || [])
  const [newInterest, setNewInterest] = useState("")
  const [githubUsername, setGithubUsername] = useState(initialData?.githubUsername || "")
  const [skills, setSkills] = useState<Skill[]>(initialData?.skills || [])
  const [selectedSkillId, setSelectedSkillId] = useState("")
  const [newSkillProficiency, setNewSkillProficiency] = useState(3)

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
    if (selectedSkillId) {
      // Check if skill already added
      if (skills.some(s => s.id === selectedSkillId)) {
        alert("This skill is already added!")
        return
      }
      
      const selectedSkill = AVAILABLE_SKILLS.find(s => s.id === selectedSkillId)
      if (selectedSkill) {
        setSkills([...skills, { 
          id: selectedSkillId,  // ✅ Using proper skill ID (1-26)
          name: selectedSkill.name, 
          proficiency: newSkillProficiency 
        }])
        setSelectedSkillId("")
        setNewSkillProficiency(3)
      }
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

  const skillLevelOptions = [
    { value: "beginner", label: "Beginner", description: "Just getting started" },
    { value: "intermediate", label: "Intermediate", description: "Some experience" },
    { value: "advanced", label: "Advanced", description: "Expert level" },
  ]

  // Get available skills that haven't been added yet
  const availableSkillsToAdd = AVAILABLE_SKILLS.filter(
    skill => !skills.some(s => s.id === skill.id)
  )

  // Group skills by category for better UX
  const groupedSkills = availableSkillsToAdd.reduce((acc, skill) => {
    if (!acc[skill.category]) {
      acc[skill.category] = []
    }
    acc[skill.category].push(skill)
    return acc
  }, {} as Record<string, typeof AVAILABLE_SKILLS>)

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
        <Label className="text-white mb-3 block">Skill Level</Label>
        <div className="flex flex-wrap gap-3">
          {skillLevelOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setSkillLevel(option.value)}
              className={`px-5 py-3 rounded-full transition-all duration-200 border ${
                skillLevel === option.value
                  ? "bg-cyan-500 text-white border-cyan-500 shadow-lg shadow-cyan-500/30"
                  : "bg-white/5 text-white/70 border-white/10 hover:border-cyan-500/50 hover:bg-white/10"
              }`}
            >
              <div className="text-left">
                <div className="font-medium">{option.label}</div>
                <div className="text-xs text-white/60">{option.description}</div>
              </div>
            </button>
          ))}
        </div>
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

      {/* Skills - FIXED VERSION WITH GROUPED SELECT */}
      <div>
        <Label className="text-white mb-2 block">Skills</Label>
        
        {/* Display existing skills */}
        {skills.length > 0 && (
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
        )}

        {/* Add new skill */}
        <div className="space-y-3 p-3 bg-white/5 rounded-lg border border-white/10">
          <div className="flex gap-2">
            <select
              value={selectedSkillId}
              onChange={(e) => setSelectedSkillId(e.target.value)}
              className="flex-1 bg-white/5 border border-white/10 text-white rounded px-3 py-2"
            >
              <option value="">Select a skill...</option>
              {Object.entries(groupedSkills).map(([category, categorySkills]) => (
                <optgroup key={category} label={category.toUpperCase()} className="bg-slate-900">
                  {categorySkills.map((skill) => (
                    <option key={skill.id} value={skill.id} className="bg-slate-900">
                      {skill.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          
          {selectedSkillId && (
            <div className="flex items-center gap-3">
              <Label className="text-white text-sm">Proficiency:</Label>
              <input
                type="range"
                min="1"
                max="5"
                value={newSkillProficiency}
                onChange={(e) => setNewSkillProficiency(Number.parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-xs text-white/60 w-8">{newSkillProficiency}/5</span>
            </div>
          )}
          
          <Button
            type="button"
            onClick={handleAddSkill}
            disabled={!selectedSkillId}
            className="w-full bg-cyan-600 text-white hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Skill
          </Button>
        </div>
      </div>

      <Button type="submit" className="w-full bg-cyan-600 text-white hover:bg-cyan-700">
        Save Profile
      </Button>
    </form>
  )
}