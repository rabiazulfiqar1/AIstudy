"use client"
import { LogOut } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

import { supabase } from "@/lib/supabaseClient"

type SignOutDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm?: () => Promise<void> | void
}

export function SignOutDialog({ open, onOpenChange, onConfirm }: SignOutDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background/80 backdrop-blur-md border border-foreground/10">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <LogOut className="size-5 text-foreground/90" aria-hidden="true" />
            Sign out
          </DialogTitle>
          <DialogDescription className="text-foreground/80">
            You’ll be signed out of your account on this device. You can sign in again at any time.
          </DialogDescription>
        </DialogHeader>

        <DialogFooter className="gap-2">
          <Button
            variant="ghost"
            className="border border-foreground/10 hover:border-foreground/20"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            className="bg-foreground text-background hover:opacity-90"
            onClick={async () => {
              await supabase.auth.signOut()
              onOpenChange(false)
              alert("Signed out successfully.")
            }}
          >
            Sign out
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
