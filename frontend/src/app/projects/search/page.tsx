"use client"

import { useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { SearchFilters } from "@/components/search-filters"
import { ProjectCard } from "@/components/project-card"
import { searchProjects } from "@/lib/api"

interface Project {
  id: number
  title: string
  description: string
  difficulty: string
  source: string
  participants?: number
  prize?: string
  match_score?: number 
}

export default function SearchPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  // Initialize state from URL params
  const [query, setQuery] = useState(searchParams.get("q") || "")
  const [difficulty, setDifficulty] = useState(searchParams.get("difficulty") || "All")
  const [source, setSource] = useState(searchParams.get("source") || "All")
  const [useSemanticSearch, setUseSemanticSearch] = useState(searchParams.get("semantic") === "true")
  const [searchResults, setSearchResults] = useState<Project[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState("")

  // Update URL when search params change
  const updateURL = (params: Record<string, string>) => {
    const newParams = new URLSearchParams(searchParams.toString())
    
    Object.entries(params).forEach(([key, value]) => {
      if (value && value !== "All") {
        newParams.set(key, value)
      } else {
        newParams.delete(key)
      }
    })
    
    router.push(`/projects/search?${newParams.toString()}`, { scroll: false })
  }

  // Perform search on mount if URL has params
  useEffect(() => {
    const hasParams = searchParams.get("q") || 
                      searchParams.get("difficulty") || 
                      searchParams.get("source")
    
    if (hasParams) {
      handleSearch()
    }
  }, []) // Only run on mount

  const handleSearch = async () => {
    setIsSearching(true)
    setError("")
    
    // Update URL with current search params
    updateURL({
      q: query,
      difficulty: difficulty,
      source: source,
      semantic: useSemanticSearch.toString()
    })

    console.log("[v0] Searching with params:", {
      query,
      difficulty,
      source,
      useSemanticSearch,
    })

    try {
      const results: Project[] = await searchProjects(
        query || undefined,
        difficulty !== "All" ? difficulty : undefined,
        source !== "All" ? source : undefined,
        useSemanticSearch,
        20,
      )
      setSearchResults(results)
    } catch (err) {
      console.error("[v0] Search error:", err)
      setError("Failed to search projects. Please try again.")
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="min-h-screen">
      {/* 🩵 Background with Light Rays */}
      <div className="min-h-screen fixed inset-0 -z-10 pointer-events-none bg-gradient-to-b from-[#0a0a10] via-[#0b1c24] to-[#0a0a10]">
        <div className="fixed inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,255,0.08)_0%,transparent_60%)]" />
        {/* subtle radial vignette */}
        <div className="fixed inset-0 bg-[radial-gradient(80%_60%_at_50%_40%,transparent_0%,oklch(0.145_0_0/.85)_100%)]" />
      </div>

      <Navbar />

      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Search Projects</h1>
          <p className="text-muted-foreground">Find projects that match your skills and interests</p>
        </div>

        <SearchFilters
          query={query}
          setQuery={setQuery}
          difficulty={difficulty}
          setDifficulty={setDifficulty}
          source={source}
          setSource={setSource}
          useSemanticSearch={useSemanticSearch}
          setUseSemanticSearch={setUseSemanticSearch}
          onSearch={handleSearch}
        />

        {/* Results */}
        <div className="mt-8">
          {error && (
            <div className="mb-4 p-4 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
              {error}
            </div>
          )}
          {searchResults.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground text-lg">
                {isSearching ? "Searching..." : "Enter a search query to find projects"}
              </p>
            </div>
          ) : (
            <div>
              <p className="text-sm text-muted-foreground mb-4">Found {searchResults.length} projects</p>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {searchResults.map((project) => (
                  <ProjectCard
                    key={project.id}
                    id={project.id.toString()}
                    title={project.title}
                    description={project.description}
                    difficulty={project.difficulty}
                    source={project.source}
                    matchScore={project.match_score || 0}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}