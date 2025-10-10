"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { SignOutDialog } from "@/components/signout-dialog"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function SignOutPage() {
  const [open, setOpen] = React.useState(true) // auto-open on visit
  const [showMessage, setShowMessage] = React.useState(false) // for toast-like message
  const router = useRouter()

  async function handleConfirm() {
    // e.g., Supabase logout can go here later
    setShowMessage(true)

    // Hide after 2s and redirect home
    setTimeout(() => {
      setShowMessage(false)
      router.replace("/")
    }, 2000)
  }

  return (
    <main className="min-h-screen w-full flex items-center justify-center px-4">
      {/* Main Sign-out Card */}
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
                  setOpen(false)
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

      {/* Dialog opens by default for a focused confirmation UX */}
      <SignOutDialog open={open} onOpenChange={setOpen} onConfirm={handleConfirm} />
    </main>
  )
}
