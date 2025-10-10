"use client"

import * as React from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Mail, Lock } from "lucide-react"

export function LoginDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const [submitting, setSubmitting] = React.useState(false)

  async function handleLogin(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setSubmitting(true)
    const form = new FormData(e.currentTarget)
    const payload = {
      email: String(form.get("email") || ""),
      password: String(form.get("password") || ""),
    }
    // TODO: Replace with real login action/integration
    console.log("[v0] Login payload:", payload)
    await new Promise((r) => setTimeout(r, 500))
    setSubmitting(false)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background/80 backdrop-blur-md border border-foreground/10">
        <DialogHeader>
          <DialogTitle>Log in</DialogTitle>
          <DialogDescription>Access your account with email and password.</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleLogin} className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="login-email">Email</Label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="login-email"
                name="email"
                type="email"
                placeholder="you@example.com"
                required
                className="pl-9"
                autoComplete="email"
              />
            </div>
          </div>

          <div className="grid gap-2">
            <Label htmlFor="login-password">Password</Label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="login-password"
                name="password"
                type="password"
                placeholder="••••••••"
                required
                className="pl-9"
                autoComplete="current-password"
              />
            </div>
          </div>

          <DialogFooter className="mt-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className="border-foreground/20"
            >
              Cancel
            </Button>
            <Button type="submit" className="bg-foreground text-background" disabled={submitting}>
              {submitting ? "Signing in..." : "Sign in"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
