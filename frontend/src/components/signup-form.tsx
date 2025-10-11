"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Github, Mail, Phone, School, User, Lock, BookUser } from "lucide-react"
import { LoginDialog } from "@/components/login-dialog"
import { useRouter } from "next/navigation"

import { supabase } from "@/lib/supabaseClient"

export function SignupForm() {
  const router = useRouter()
  const [loginOpen, setLoginOpen] = React.useState(false)
  const [submitting, setSubmitting] = React.useState(false)

  function getValue(form: FormData, key: string) {
    const val = form.get(key)
    if (!val || String(val).trim() === "") return null
    return String(val)
  }

  async function handleSignup(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setSubmitting(true)
    const form = new FormData(e.currentTarget)
    const payload = {
      fullName: getValue(form, "fullName"),
      university: getValue(form, "university"),
      degree: getValue(form, "degree"),
      username: getValue(form, "username"),
      email: String(form.get("email") || ""),
      password: String(form.get("password") || ""),
      phone: getValue(form, "phone"),
    }

    const { data, error } = await supabase.auth.signUp({
      email: payload.email,
      password: payload.password,
      options: {
        data: {
          full_name: payload.fullName,
          university: payload.university,
          degree: payload.degree,
          username: payload.username,
          phone: payload.phone,
        },
      },
    })

    if (error) {
      console.error("Signup failed:", error.message)
      alert("Signup failed: " + error.message)
      setSubmitting(false)
      return
    }

    // Step 2: Also insert into your custom public.users table
    const user = data.user
    if (user) {
      const { error: insertError } = await supabase.from("users").insert({
        id: user.id, // same as auth.users id (UUID)
        username: payload.username,
        full_name: payload.fullName,
        university: payload.university,
        degree: payload.degree,
        phone: payload.phone,
        profile_pic: null,
      })

      if (insertError) {
          console.error("Error saving user data:", insertError.message)
      } else {
          console.log("User data saved successfully in public.users")
      }
    }

    router.push("/")
    setSubmitting(false)
  }

  async function handleOauth(provider: "google" | "github") {
    const { data, error } = await supabase.auth.signInWithOAuth({ provider })
    if (error) console.error(error.message)
    console.log("[v0] OAuth clicked:", provider)
  }

  return (
    <div className="grid gap-6">
      {/* OAuth providers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Button
          type="button"
          variant="outline"
          onClick={() => handleOauth("google")}
          className="w-full border-foreground/20 hover:border-foreground/30 hover:bg-foreground/[0.06]"
          aria-label="Sign up with Google"
        >
          <Mail className="mr-2 h-4 w-4" />
          Continue with Google
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => handleOauth("github")}
          className="w-full border-foreground/20 hover:border-foreground/30 hover:bg-foreground/[0.06]"
          aria-label="Sign up with GitHub"
        >
          <Github className="mr-2 h-4 w-4" />
          Continue with GitHub
        </Button>
      </div>

      <div className="relative">
        <Separator className="bg-foreground/10" />
        <span className="absolute inset-0 -translate-y-1/2 flex items-center justify-center">
          <span className="px-3 text-xs text-foreground/60 bg-background/80 backdrop-blur-sm rounded-full border border-foreground/10">
            or continue with email
          </span>
        </span>
      </div>

      {/* Email sign up form */}
      <form onSubmit={handleSignup} className="grid gap-5">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor="fullName">Full name</Label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="fullName"
                name="fullName"
                placeholder="Ada Lovelace"
                required
                className="pl-9"
                autoComplete="name"
              />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="username">Username</Label>
            <div className="relative">
              <BookUser className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="username"
                name="username"
                placeholder="adalabs"
                required
                className="pl-9"
                autoComplete="username"
              />
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor="university">University / School</Label>
            <div className="relative">
              <School className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="university"
                name="university"
                placeholder="University of Example"
                className="pl-9"
                autoComplete="organization"
              />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="degree">Degree / Field of study</Label>
            <div className="relative">
              <BookUser className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="degree"
                name="degree"
                placeholder="Computer Science"
                className="pl-9"
                autoComplete="education"
              />
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="grid gap-2">
            <Label htmlFor="email">Email</Label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="email"
                name="email"
                type="email"
                placeholder="ada@example.com"
                required
                className="pl-9"
                autoComplete="email"
              />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="phone">Phone (optional)</Label>
            <div className="relative">
              <Phone className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="phone"
                name="phone"
                type="tel"
                placeholder="+1 555 555 5555"
                className="pl-9"
                autoComplete="tel"
              />
            </div>
          </div>
        </div>

        <div className="grid gap-2">
          <Label htmlFor="password">Password</Label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
            <Input
              id="password"
              name="password"
              type="password"
              placeholder="••••••••"
              required
              className="pl-9"
              autoComplete="new-password"
              minLength={8}
            />
          </div>
        </div>

        <div className="flex flex-col md:flex-row md:items-center gap-4 md:justify-between">
          <p className="text-sm text-foreground/70">
            By continuing you agree to our terms and acknowledge our privacy policy.
          </p>
          <Button
            type="submit"
            className="self-start md:self-auto bg-foreground text-background hover:opacity-95 focus-visible:ring-2 focus-visible:ring-cyan-400"
            disabled={submitting}
            aria-busy={submitting}
          >
            {submitting ? "Creating account..." : "Create account"}
          </Button>
        </div>
      </form>

      <div className="flex items-center justify-center gap-2 text-sm">
        <span className="text-foreground/70">Already have an account?</span>
        <button
        type="button"
        onClick={() => setLoginOpen(true)}
        className="text-cyan-400 hover:text-cyan-600 hover:brightness-90 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded transition-all"
        >
        Log in
        </button>
      </div>

      <LoginDialog open={loginOpen} onOpenChange={setLoginOpen} />
    </div>
  )
}
