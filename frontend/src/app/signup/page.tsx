import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { SignupForm } from "@/components/signup-form"

export default function Page() {
  return (
    <div className="relative w-full min-h-screen overflow-hidden flex flex-col items-center justify-center px-4 py-12">
      {/* Background */}
      <div className="absolute inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
      </div>

      {/* 🪩 Signup Card */}
      <Card className="w-full max-w-2xl bg-background/70 backdrop-blur-md border border-foreground/10 shadow-lg hover:shadow-cyan-500/10 transition-shadow">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Create your account</CardTitle>
          <CardDescription className="text-balance">
            Sign up to plan your learning path, process videos, and chat with the community.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SignupForm />
        </CardContent>
      </Card>
    </div>
  )
}
