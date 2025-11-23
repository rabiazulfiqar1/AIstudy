"use client"

import { useState, useEffect } from "react"
import { Navbar } from "@/components/navbar"
import { ProfileForm } from "@/components/profile-form"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { History, Loader2 } from "lucide-react"
import { getUserProfile, updateUserProfile, transformProfileFromBackend } from "@/lib/api"
import { supabase } from "@/lib/supabaseClient"
import LightRays from "@/components/LightRays"

interface ProfileFormData {
  bio?: string
  skillLevel?: string
  interests?: string[]
  githubUsername?: string
  preferredProjectTypes?: string[]
  skills?: {
    id: string
    name: string
    proficiency: number
  }[]
}

export default function ProfilePage() {
  const [saveStatus, setSaveStatus] = useState("")
  const [loading, setLoading] = useState(true)
  const [initialData, setInitialData] = useState<ProfileFormData | undefined>(undefined)
  const [userId, setUserId] = useState<string | null>(null)

  useEffect(() => {
    // Get user ID from Supabase auth and fetch profile
    const fetchUserProfile = async () => {
      try {
        // Get authenticated user from Supabase
        const { data: { user } } = await supabase.auth.getUser()
        
        if (!user) {
          console.error("No authenticated user found")
          setLoading(false)
          return
        }

        const currentUserId = user.id
        setUserId(currentUserId)

        const profile = await getUserProfile(currentUserId)
        
        if (profile) {
          const transformedData = transformProfileFromBackend(profile)
          setInitialData(transformedData)
        }
      } catch (error) {
        console.error("Failed to fetch profile:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchUserProfile()
  }, [])

  const handleProfileSubmit = async (data: ProfileFormData) => {
    if (!userId) {
      setSaveStatus("Error: User not authenticated")
      return
    }

    try {
      // Transform frontend data to match API requirements
      const apiData = {
        bio: data.bio,
        skillLevel: data.skillLevel,
        interests: data.interests,
        githubUsername: data.githubUsername,
        preferredProjectTypes: data.preferredProjectTypes,
        skills: data.skills?.map(skill => ({
          skill_id: skill.id,
          name: skill.name,
          proficiency: skill.proficiency.toString(),
        })),
      }
      
      await updateUserProfile(userId, apiData)
      setSaveStatus("Profile saved successfully!")
      setTimeout(() => setSaveStatus(""), 3000)
    } catch (error) {
      console.error("Failed to save profile:", error)
      setSaveStatus("Failed to save profile. Please try again.")
      setTimeout(() => setSaveStatus(""), 3000)
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* 🩵 Background with Light Rays - Consistent with Home Page */}
      <div className="fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        <LightRays
          raysOrigin="top-center"
          raysColor="#ffffff"
          raysSpeed={1}
          lightSpread={0.8}
          rayLength={3.0}
          followMouse
          mouseInfluence={0.1}
          noiseAmount={0.0}
          distortion={0.0}
        />
        {/* subtle radial vignette */}
        <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      <Navbar />

      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Your Profile</h1>
          <p className="text-white/70">
            Manage your profile and preferences to get better project recommendations
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
          </div>
        ) : (
          <Tabs defaultValue="profile" className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-white/5 border border-white/10">
              <TabsTrigger 
                value="profile"
                className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white text-white/70"
              >
                Edit Profile
              </TabsTrigger>
              <TabsTrigger 
                value="interactions"
                className="data-[state=active]:bg-cyan-600 data-[state=active]:text-white text-white/70"
              >
                Interaction History
              </TabsTrigger>
            </TabsList>

            <TabsContent value="profile" className="mt-6">
              <Card className="bg-white/5 backdrop-blur-sm border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Profile Information</CardTitle>
                  <CardDescription className="text-white/60">
                    Update your skills, interests, and preferences
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ProfileForm initialData={initialData} onSubmit={handleProfileSubmit} />
                  {saveStatus && (
                    <div
                      className={`mt-4 p-3 rounded-lg border text-sm ${
                        saveStatus.includes("successfully")
                          ? "bg-green-900/20 border-green-900/50 text-green-300"
                          : "bg-red-900/20 border-red-900/50 text-red-300"
                      }`}
                    >
                      {saveStatus}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="interactions" className="mt-6">
              <Card className="bg-white/5 backdrop-blur-sm border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">Interaction History</CardTitle>
                  <CardDescription className="text-white/60">
                    View your project interactions and activity
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-white/60 text-sm">
                    Your interactions (viewed, saved, rated projects) will appear here.
                  </p>
                  <Link
                    href="/profile/interactions"
                    className="inline-flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors"
                  >
                    <History className="h-4 w-4" />
                    View detailed history
                  </Link>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  )
}