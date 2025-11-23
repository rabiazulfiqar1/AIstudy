"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Search } from "lucide-react"

interface SearchFiltersProps {
  query: string
  setQuery: (q: string) => void
  difficulty: string
  setDifficulty: (d: string) => void
  source: string
  setSource: (s: string) => void
  useSemanticSearch: boolean
  setUseSemanticSearch: (v: boolean) => void
  onSearch: () => void
}

export function SearchFilters({
  query,
  setQuery,
  difficulty,
  setDifficulty,
  source,
  setSource,
  useSemanticSearch,
  setUseSemanticSearch,
  onSearch,
}: SearchFiltersProps) {
  const sources = ["All", "kaggle_competition", "kaggle_dataset", "GitHub"]
  const difficulties = ["All", "beginner", "intermediate", "advanced"]

  return (
    <div className="space-y-4 p-6 rounded-lg border border-white/10 bg-[#0b1c24]/60 backdrop-blur-sm hover:border-cyan-500/50 transition-colors">
      {/* Search Input */}
      <div className="space-y-2">
        <Label htmlFor="search-query" className="text-white">
          Search Query
        </Label>
        <div className="flex gap-2">
          <Input
            id="search-query"
            placeholder="Search for projects..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && onSearch()}
            className="bg-[#0b1c24]/40 border-white/20 text-white placeholder:text-white/60 focus:ring-2 focus:ring-cyan-500"
          />
          <Button
            onClick={onSearch}
            className="bg-cyan-500 text-black hover:bg-cyan-500/70 gap-2"
          >
            <Search className="h-4 w-4" />
            Search
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Difficulty Filter */}
        <div className="space-y-2">
          <Label htmlFor="difficulty" className="text-white">
            Difficulty
          </Label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="w-full px-3 py-2 bg-[#0b1c24]/40 border border-white/20 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {difficulties.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {/* Source Filter */}
        <div className="space-y-2">
          <Label htmlFor="source" className="text-white">
            Source
          </Label>
          <select
            id="source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="w-full px-3 py-2 bg-[#0b1c24]/40 border border-white/20 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {sources.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {/* Search Type Toggle */}
        <div className="space-y-2">
          <Label className="text-white">Search Type</Label>
          <div className="flex gap-2 h-10">
            <button
              onClick={() => setUseSemanticSearch(false)}
              className={`flex-1 rounded-md text-sm font-medium transition-colors ${
                !useSemanticSearch
                  ? "bg-cyan-500 text-black hover:bg-cyan-500/70"
                  : "bg-white/10 text-white/80 border border-white/20 hover:bg-white/20"
              }`}
            >
              Keyword
            </button>
            <button
              onClick={() => setUseSemanticSearch(true)}
              className={`flex-1 rounded-md text-sm font-medium transition-colors ${
                useSemanticSearch
                  ? "bg-cyan-500 text-black hover:bg-cyan-500/70"
                  : "bg-white/10 text-white/80 border border-white/20 hover:bg-white/20"
              }`}
            >
              Semantic
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
