import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type React from "react" // Import React to declare JSX

type Feature = {
  title: string
  description: string
  points: string[]
  cta: string
  href: string
  Icon: (props: { className?: string }) => React.JSX.Element // Use React.JSX.Element instead of JSX.Element
}

const icons = {
  Video: ({ className = "w-5 h-5" }) => (
    <svg
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
    >
      <rect x="3" y="5" width="14" height="14" rx="2" />
      <path d="M21 7l-4 3v4l4 3V7z" />
    </svg>
  ),
  Plan: ({ className = "w-5 h-5" }) => (
    <svg
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
    >
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M8 2v4M16 2v4M3 10h18M7 14h5M7 18h5" />
    </svg>
  ),
  Rocket: ({ className = "w-5 h-5" }) => (
    <svg
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
    >
      <path d="M14 4l6 6-6 6-6-6 6-6z" />
      <path d="M14 4l-3 7-7 3 3-7 7-3z" />
      <path d="M5 19c2 0 3-1 4-3-2 1-3 2-4 3z" />
    </svg>
  ),
  Chat: ({ className = "w-5 h-5" }) => (
    <svg
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
    >
      <path d="M21 12a8 8 0 10-3.8 6.8L21 21l-2.2-3.2A8 8 0 0021 12z" />
      <path d="M8 11h8M8 15h5M8 7h8" />
    </svg>
  ),
}

const features: Feature[] = [
  {
    title: "Video Processing",
    description: "Summarize, take notes, translate, and extract transcripts from any video.",
    points: ["Summarization", "Auto-notes", "Translation", "Transcript export"],
    cta: "Open Video Tools",
    href: "/notes",
    Icon: icons.Video,
  },
  {
    title: "Planner & Tracker",
    description: "Plan your learning path and track real progress with milestones.",
    points: ["Roadmaps", "Milestones", "Reminders", "Progress charts"],
    cta: "Plan My Path",
    href: "#planner",
    Icon: icons.Plan,
  },
  {
    title: "Project Recommendations",
    description: "Get real-world project ideas tailored to your topic and skill level.",
    points: ["Topic-based ideas", "Difficulty tiers", "Tech stacks", "Outcome-focused"],
    cta: "Find Projects",
    href: "/proj_recommendations",
    Icon: icons.Rocket,
  },
  {
    title: "Community Chat",
    description: "Discuss ideas, get feedback, and learn with other users in real time.",
    points: ["Topic channels", "Threaded replies", "Mentor tips", "Resource sharing"],
    cta: "Join the Chat",
    href: "#chat",
    Icon: icons.Chat,
  },
]

export function FeatureCards() {
  return (
    <div className="mx-auto max-w-6xl">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 md:gap-6 items-stretch">
        {features.map((f) => (
          <Card
            key={f.title}
            className="group relative h-full overflow-hidden bg-background/50 backdrop-blur-md border border-foreground/10 transition-all duration-300 will-change-transform hover:-translate-y-0.5 hover:shadow-[0_20px_50px_-10px_rgba(34,211,238,0.25)] before:pointer-events-none before:absolute before:inset-0 before:bg-[radial-gradient(240px_160px_at_top,rgba(34,211,238,0.12),transparent_60%)] before:opacity-0 group-hover:before:opacity-100 before:transition-opacity"
          >
            <CardHeader className="flex flex-row items-center gap-3 pb-2">
              <div className="flex size-9 items-center justify-center rounded-full bg-foreground/10 text-foreground ring-1 ring-foreground/10 group-hover:text-[rgb(34,211,238)]">
                <f.Icon className="w-4.5 h-4.5" />
              </div>
              <CardTitle className="text-lg">{f.title}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col h-full">
              <p className="text-sm text-foreground leading-relaxed mb-3">{f.description}</p>
              <ul className="text-sm text-foreground space-y-1.5 mb-4">
                {f.points.map((p) => (
                  <li key={p} className="flex items-center gap-2">
                    <span className="inline-block size-1.5 rounded-full bg-foreground/40" aria-hidden="true" />
                    {p}
                  </li>
                ))}
              </ul>
              <div className="mt-auto flex justify-center pt-2">
                <a
                  href={f.href}
                  className="inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-medium bg-foreground text-background transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_10px_30px_-10px_rgba(34,211,238,0.45)] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label={f.cta}
                >
                  {f.cta}
                </a>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
