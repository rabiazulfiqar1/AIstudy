"use client"

import { useState, type ChangeEvent, type FormEvent, type DragEvent } from "react"
import { Download, Loader2, Upload } from 'lucide-react'

interface TranscriptResponse {
  segments?: Array<{ start: number; end: number; text: string }>
  full_text?: string
  transcript?: string
}

interface SummaryResponse {
  summary: string
}

interface NotesResponse {
  notes: string
}

export function VideoUploadForm() {
  const [file, setFile] = useState<File | null>(null)
  const [link, setLink] = useState("")
  const [loading, setLoading] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [summary, setSummary] = useState("")
  const [notes, setNotes] = useState("")
  const [error, setError] = useState("")
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(e.type === "dragenter" || e.type === "dragover")
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length > 0) {
      const selectedFile = droppedFiles[0]
      if (selectedFile.type.startsWith("video/") || selectedFile.type.startsWith("audio/")) {
        setFile(selectedFile)
        setLink("")
        setError("")
      } else {
        setError("Please upload a valid video or audio file")
      }
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setLink("")
      setError("")
    }
  }

  const handleLinkChange = (e: ChangeEvent<HTMLInputElement>) => {
    setLink(e.target.value)
    setFile(null)
    setError("")
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!file && !link) {
      setError("Please upload a file or provide a link")
      return
    }

    setLoading(true)
    setError("")
    setTranscript("")
    setSummary("")
    setNotes("")

    try {
      const formData = new FormData()
      if (file) {
        formData.append("file", file)
      } else if (link) {
        formData.append("link", link)
      }

      const transcriptResponse = await fetch("http://localhost:8000/api/get_transcript", {
        method: "POST",
        body: formData,
      })

      if (!transcriptResponse.ok) {
        throw new Error("Failed to generate transcript")
      }

      const transcriptData: TranscriptResponse = await transcriptResponse.json()
      const transcriptText = transcriptData.full_text || transcriptData.transcript || ""
      setTranscript(transcriptText)

      const summaryResponse = await fetch("http://localhost:8000/api/get_summary", {
        method: "POST",
        body: formData,
      })

      if (!summaryResponse.ok) {
        throw new Error("Failed to generate summary")
      }

      const summaryData: SummaryResponse = await summaryResponse.json()
      setSummary(summaryData.summary)

      const notesResponse = await fetch("http://localhost:8000/api/generate_notes", {
        method: "POST",
        body: formData,
      })

      if (!notesResponse.ok) {
        throw new Error("Failed to generate notes")
      }

      const notesData: NotesResponse = await notesResponse.json()
      setNotes(notesData.notes)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const downloadFile = (content: string, filename: string) => {
    const element = document.createElement("a")
    const file = new Blob([content], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = filename
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const downloadTranscript = () => {
    downloadFile(transcript, `transcript-${Date.now()}.txt`)
  }

  const downloadSummary = () => {
    downloadFile(summary, `summary-${Date.now()}.txt`)
  }

  const downloadNotes = () => {
    downloadFile(notes, `notes-${Date.now()}.md`)
  }

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drag and Drop Area */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ${
            dragActive
              ? "border-cyan-500/80 bg-cyan-500/10"
              : "border-white/20 bg-white/5 hover:border-white/30 hover:bg-white/10"
          }`}
        >
          <input
            type="file"
            onChange={handleFileChange}
            accept="video/*,audio/*"
            className="hidden"
            id="file-input"
            aria-label="Upload video or audio file"
          />
          <label htmlFor="file-input" className="cursor-pointer">
            <div className="flex flex-col items-center justify-center gap-3">
              <Upload className="w-8 h-8 text-cyan-500/60" />
              <div className="text-center">
                <p className="text-white/80 font-medium">Drag and drop your video here</p>
                <p className="text-white/50 text-sm mt-1">or click to browse</p>
              </div>
            </div>
          </label>

          {file && (
            <div className="absolute top-3 right-3 bg-cyan-500/20 text-cyan-300 px-3 py-1 rounded-full text-sm flex items-center gap-2">
              <span className="w-2 h-2 bg-cyan-400 rounded-full" />
              {file.name}
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-white/10" />
          <span className="text-white/50 text-sm">Or</span>
          <div className="flex-1 h-px bg-white/10" />
        </div>

        {/* Link Input */}
        <div className="space-y-2">
          <label htmlFor="link-input" className="block text-sm font-medium text-white/80">
            Paste video link
          </label>
          <input
            id="link-input"
            type="url"
            placeholder="https://example.com/video.mp4"
            value={link}
            onChange={handleLinkChange}
            className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:border-cyan-500/50 focus:bg-white/[0.15] transition-all"
            aria-label="Video or audio link"
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full px-6 py-3 rounded-lg bg-foreground text-background font-semibold transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring flex items-center justify-center gap-2"
          aria-label={loading ? "Processing..." : "Process video"}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing...
            </>
          ) : (
            "Process Video"
          )}
        </button>
      </form>

      {/* Results Section */}
      {(transcript || summary || notes) && (
        <div className="space-y-6 animate-fadeIn">
          {/* Transcript Section */}
          {transcript && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white/90">Transcript</h2>
                <button
                  onClick={downloadTranscript}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 transition-colors text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Download transcript"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              <p className="text-white/70 leading-relaxed max-h-64 overflow-y-auto pr-2">{transcript}</p>
            </div>
          )}

          {/* Summary Section */}
          {summary && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white/90">Summary</h2>
                <button
                  onClick={downloadSummary}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 transition-colors text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Download summary"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              <p className="text-white/70 leading-relaxed max-h-64 overflow-y-auto pr-2">{summary}</p>
            </div>
          )}

          {notes && (
            <div className="bg-white/5 border border-white/10 rounded-xl p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white/90">Notes</h2>
                <button
                  onClick={downloadNotes}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 transition-colors text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Download notes"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              <div className="text-white/70 leading-relaxed max-h-64 overflow-y-auto pr-2 prose prose-invert">
                <div className="whitespace-pre-wrap">{notes}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
