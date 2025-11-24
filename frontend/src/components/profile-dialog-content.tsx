"use client"

import * as React from "react"
import { supabase } from "@/lib/supabaseClient"
import type { User } from "@supabase/supabase-js"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { UserIcon, BookUser, School, Mail, Phone, Loader2 } from "lucide-react"

interface ProfileDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  user: User | null
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api'

export function ProfileDialogContent({ open, onOpenChange, user }: ProfileDialogProps) {
  const [isEditing, setIsEditing] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [profile, setProfile] = React.useState({
    fullName: "",
    username: "",
    organization: "",
    fieldOfStudy: "",
    phone: "",
  })

  // Load profile from backend when dialog opens
  const loadProfile = React.useCallback(async () => {
    if (!user) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/users/${user.id}/basic-profile`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (response.ok) {
        const data = await response.json()
        setProfile({
          fullName: data.full_name || "",
          username: data.username || "",
          organization: data.organization || "",
          fieldOfStudy: data.field_of_study || "",
          phone: data.phone || "",
        })
      } else if (response.status === 404) {
        // User doesn't exist in backend yet - use email as default
        setProfile({
          fullName: "",
          username: user.email?.split('@')[0] || "",
          organization: "",
          fieldOfStudy: "",
          phone: "",
        })
      } else {
        throw new Error('Failed to load profile')
      }
    } catch (err) {
      console.error("Error loading profile:", err)
      setError("Failed to load profile")
    } finally {
      setIsLoading(false)
    }
  }, [user])

  React.useEffect(() => {
    if (user && open) {
      loadProfile()
    }
  }, [user, open, loadProfile])

  const handleSave = async () => {
    if (!user) return

    // Validate required fields
    if (!profile.fullName.trim()) {
      setError("Full name is required")
      return
    }

    if (!profile.username.trim()) {
      setError("Username is required")
      return
    }

    setIsSaving(true)
    setError(null)

    try {
      // Save to backend
      const response = await fetch(`${API_BASE_URL}/users/${user.id}/basic-profile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          full_name: profile.fullName,
          username: profile.username,
          organization: profile.organization || null,
          field_of_study: profile.fieldOfStudy || null,
          phone: profile.phone || null,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to save profile')
      }

      // Also update Supabase auth metadata for consistency (optional)
      await supabase.auth.updateUser({
        data: {
          full_name: profile.fullName,
          username: profile.username,
        },
      })

      setIsEditing(false)
    } catch (err) {
      // Fixed: Specify type for err
      console.error("Error saving profile:", err)
      setError(err instanceof Error ? err.message : "Failed to save profile")
    } finally {
      setIsSaving(false)
    }
  }

  const handleCancel = () => {
    setIsEditing(false)
    setError(null)
    // Reload profile to reset changes
    if (user) {
      loadProfile()
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md bg-white/10 backdrop-blur-md border-white/20">
        <DialogHeader>
          <DialogTitle className="text-white">Profile</DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
          </div>
        ) : user ? (
          <div className="grid gap-6">
            {/* Profile Avatar */}
            <div className="flex justify-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-400 to-cyan-600 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">
                  {profile.fullName?.charAt(0)?.toUpperCase() || user.email?.charAt(0)?.toUpperCase() || "U"}
                </span>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-900/20 border border-red-900/50 text-red-300 p-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            {/* Profile Fields */}
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="fullName" className="text-white">
                  Full Name <span className="text-red-400">*</span>
                </Label>
                <div className="relative">
                  <UserIcon
                    className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60"
                    aria-hidden
                  />
                  <Input
                    id="fullName"
                    name="fullName"
                    placeholder="Enter your full name"
                    value={profile.fullName}
                    onChange={(e) => setProfile({ ...profile, fullName: e.target.value })}
                    disabled={!isEditing}
                    required
                    className="pl-9 bg-white/5 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="username" className="text-white">
                  Username <span className="text-red-400">*</span>
                </Label>
                <div className="relative">
                  <UserIcon
                    className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60"
                    aria-hidden
                  />
                  <Input
                    id="username"
                    name="username"
                    placeholder="Enter your username"
                    value={profile.username}
                    onChange={(e) => setProfile({ ...profile, username: e.target.value })}
                    disabled={!isEditing}
                    required
                    className="pl-9 bg-white/5 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="organization" className="text-white">University / School</Label>
                <div className="relative">
                  <School className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60" aria-hidden />
                  <Input
                    id="organization"
                    placeholder="Enter your university or school"
                    value={profile.organization}
                    onChange={(e) => setProfile({ ...profile, organization: e.target.value })}
                    disabled={!isEditing}
                    className="pl-9 bg-white/5 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="fieldOfStudy" className="text-white">Degree / Field of Study</Label>
                <div className="relative">
                  <BookUser
                    className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60"
                    aria-hidden
                  />
                  <Input
                    id="fieldOfStudy"
                    placeholder="Enter your field of study"
                    value={profile.fieldOfStudy}
                    onChange={(e) => setProfile({ ...profile, fieldOfStudy: e.target.value })}
                    disabled={!isEditing}
                    className="pl-9 bg-white/5 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="email" className="text-white">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60" aria-hidden />
                  <Input 
                    id="email" 
                    value={user.email || ""} 
                    disabled 
                    className="pl-9 bg-white/5 border-white/10 text-white/50 opacity-50" 
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="phone" className="text-white">Phone (Optional)</Label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/60" aria-hidden />
                  <Input
                    id="phone"
                    placeholder="Enter your phone number"
                    value={profile.phone}
                    onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                    disabled={!isEditing}
                    className="pl-9 bg-white/5 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                  />
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 pt-4">
              {!isEditing ? (
                <Button
                  onClick={() => setIsEditing(true)}
                  className="flex-1 bg-cyan-600 text-white hover:bg-cyan-700"
                >
                  Edit Profile
                </Button>
              ) : (
                <>
                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    className="flex-1 bg-white/10 text-white hover:text-black"
                    disabled={isSaving}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSave}
                    disabled={isSaving}
                    className="flex-1 bg-cyan-600 text-white hover:bg-cyan-700"
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Save"
                    )}
                  </Button>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-white/70 mb-4">Please sign in to view your profile</p>
            <Button className="bg-cyan-600 text-white hover:bg-cyan-700">Sign In</Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}