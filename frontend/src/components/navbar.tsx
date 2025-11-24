"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Search, BarChart3, Lightbulb, User } from "lucide-react"

export function Navbar() {
  const pathname = usePathname() //current URL path.

  const navItems = [
    { href: "/", label: "Home", icon: Lightbulb },
    { href: "/projects/search", label: "Search Projects", icon: Search },
    { href: "/recommendations", label: "Recommendations", icon: BarChart3 },
  ]

  return (
    <nav className="border-b border-white/10 bg-white/5 backdrop-blur-md">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-semibold text-lg text-white">
            <div className="h-8 w-8 rounded-lg bg-cyan-600 flex items-center justify-center text-white">
              ◆
            </div>
            <span className="hidden sm:inline">ProjectMatch</span>
          </Link>

          {/* Nav Items */}
          <div className="flex items-center gap-1">
            {navItems.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  pathname === href
                    ? "bg-cyan-600 text-white"
                    : "text-white/70 hover:bg-white/10 hover:text-cyan-400",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            ))}

            <Link
              href="/profile"
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                pathname === "/profile" || pathname.startsWith("/profile/")
                  ? "bg-cyan-600 text-white"
                  : "text-white/70 hover:bg-white/10 hover:text-cyan-400",
              )}
              aria-label="Go to profile page"
            >
              <User className="h-4 w-4" />
              <span className="hidden sm:inline">Profile</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}