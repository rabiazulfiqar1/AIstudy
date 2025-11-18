"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Github, Mail, Phone, School, User, Lock, BookUser, Eye, EyeOff } from 'lucide-react'
import { LoginDialog } from "@/components/login-dialog"
import { useRouter } from 'next/navigation'

import { supabase } from "@/lib/supabaseClient"

export function SignupForm() {
  const router = useRouter()
  const [loginOpen, setLoginOpen] = React.useState(false)
  const [submitting, setSubmitting] = React.useState(false)
  const [passwordError, setPasswordError] = React.useState("")
  const [showPassword, setShowPassword] = React.useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = React.useState(false)

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
      organization: getValue(form, "organization"),
      fieldOfStudy: getValue(form, "fieldOfStudy"),
      username: getValue(form, "username"),
      email: String(form.get("email") || ""),
      password: String(form.get("password") || ""),
      confirmPassword: String(form.get("confirmPassword") || ""),
      phone: getValue(form, "phone"),
    }

    if (payload.password !== payload.confirmPassword) {
      setPasswordError("Passwords do not match")
      setSubmitting(false)
      return
    }


    const { data, error } = await supabase.auth.signUp({
      email: payload.email,
      password: payload.password,
      options: {
        data: {
          full_name: payload.fullName,
          organization: payload.organization,
          field_of_study: payload.fieldOfStudy,
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
        user_id: user.id, // same as auth.users id (UUID)
        email: payload.email,
        username: payload.username,
        full_name: payload.fullName,
        organization: payload.organization,
        field_of_study: payload.fieldOfStudy,
        phone: payload.phone,
        profile_pic: null,
      })

      if (insertError) {
        console.error("Error saving user data:", insertError.message)
        alert("There was a problem saving your account. Please try again.")
        setSubmitting(false)
        return // stay on signup page
      }
    }

    console.log("User data saved successfully in public.users")
    router.push("/")
    setSubmitting(false)
  }

  async function handleOauth(provider: "google" | "github") {
    const { data, error } = await supabase.auth.signInWithOAuth({ 
      provider ,
      options: {redirectTo: window.location.origin }
    })
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
                placeholder="Rabia Zulfiqar"
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
            <Label htmlFor="organization">University / School</Label>
            <div className="relative">
              <School className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="organization"
                name="organization"
                placeholder="University of Example"
                className="pl-9"
                autoComplete="organization"
              />
            </div>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="fieldOfStudy">Degree / Field of study</Label>
            <div className="relative">
              <BookUser className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
              <Input
                id="fieldOfStudy"
                name="fieldOfStudy"
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
                placeholder="03316061234"
                className="pl-9"
                autoComplete="tel"
                inputMode="numeric"
                pattern="^0\\d{10}$"
                minLength={11}
                maxLength={11}
                title="Phone number must start with 0 and be exactly 11 digits (e.g. 03316061234)"
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
              type={showPassword ? "text" : "password"}
              placeholder="••••••••"
              className="pl-9 pr-9"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground/60 hover:text-foreground transition-colors"
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>

        <div className="grid gap-2">
          <Label htmlFor="confirmPassword">Confirm Password</Label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-foreground/60" aria-hidden />
            <Input
              id="confirmPassword"
              name="confirmPassword"
              type={showConfirmPassword ? "text" : "password"}
              placeholder="••••••••"
              required
              className="pl-9 pr-9"
              autoComplete="new-password"
              minLength={8}
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground/60 hover:text-foreground transition-colors"
              aria-label={showConfirmPassword ? "Hide password" : "Show password"}
            >
              {showConfirmPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>

        {passwordError && <p className="text-sm text-red-500">{passwordError}</p>}

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
