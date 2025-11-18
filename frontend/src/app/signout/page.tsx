"use client"

import * as React from "react"
import { useRouter } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { supabase } from "@/lib/supabaseClient"

export default function SignOutPage() {
  const [showMessage, setShowMessage] = React.useState(false) // for toast-like message
  const router = useRouter()

  async function handleConfirm() {
    const { error } = await supabase.auth.signOut()  // ← this signs out the user
    if (error) {
      console.error("Sign out error:", error.message)
      alert("Failed to sign out. Please try again.")
      return
    }

    setShowMessage(true)

    // Hide after 2s and redirect home
    setTimeout(() => {
      setShowMessage(false)
      router.replace("/")
    }, 2000)
  }

  return (
    <div className="relative w-full min-h-screen overflow-hidden flex flex-col items-center justify-center px-4 py-12">
      {/* Background */}
      <div className="absolute inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
      </div>
      
      <main className="min-h-screen w-full flex items-center justify-center px-4">
        <div className="w-full max-w-md relative">
          <Card className="bg-background/65 backdrop-blur-md border border-foreground/10 shadow-xl">
            <CardHeader>
              <CardTitle className="text-balance text-lg md:text-xl">Sign out</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-foreground/80">Are you sure you want to sign out from this device?</p>
              <div className="flex items-center justify-end gap-2">
                <Button
                  variant="ghost"
                  className="border border-foreground/10 hover:border-foreground/20"
                  onClick={() => {
                    router.back()
                  }}
                >
                  Cancel
                </Button>
                <Button
                  className="bg-foreground text-background hover:opacity-90"
                  onClick={handleConfirm}
                >
                  Sign out
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* ✅ Custom Toast Message (manual, Tailwind only) */}
          {showMessage && (
            <div className="absolute left-1/2 -bottom-16 transform -translate-x-1/2">
              <div className="bg-foreground text-background px-4 py-2 rounded-md shadow-md text-sm animate-fadeIn">
                ✅ Signed out successfully!
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
