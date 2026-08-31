"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, Loader2, Trash2, Upload } from "lucide-react";
import { ApiError, deleteDocument, listDocuments, uploadDocument } from "@/lib/api";
import type { UserDocument } from "@/lib/types";

const ACCEPTED = ".csv,.txt,.md,.pdf";

/** Lets a user bring their own data into what ABBot can search — uploaded here,
 * it's chunked and embedded the same way the curated knowledge base is, then folded
 * into the same retrieval pool everywhere ABBot answers (this page, the floating
 * widget, Practice Lab follow-ups), not just in this one conversation. */
export function DocumentUpload() {
  const [documents, setDocuments] = useState<UserDocument[] | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listDocuments()
      .then(setDocuments)
      .catch(() => setDocuments([]));
  }, []);

  async function handleFileSelected(file: File) {
    setError(null);
    setUploading(true);
    try {
      const doc = await uploadDocument(file);
      setDocuments((prev) => [doc, ...(prev ?? [])]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleDelete(id: string) {
    await deleteDocument(id);
    setDocuments((prev) => (prev ?? []).filter((d) => d.id !== id));
  }

  if (documents === null) return null;

  return (
    <div className="mb-3 rounded-xl border border-surface-border bg-surface p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted">
          <FileText className="h-3.5 w-3.5" />
          Your documents
        </p>
        <label
          className={`flex items-center gap-1.5 rounded-lg border border-surface-border px-2.5 py-1 text-xs font-medium hover:bg-surface-2 ${
            uploading ? "cursor-not-allowed opacity-60" : "cursor-pointer"
          }`}
        >
          {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
          Upload
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            disabled={uploading}
            onChange={(e) => e.target.files?.[0] && handleFileSelected(e.target.files[0])}
          />
        </label>
      </div>

      {error && <p className="mt-2 text-xs text-danger">{error}</p>}

      {documents.length === 0 ? (
        <p className="mt-2 text-xs text-muted">
          Upload a CSV, TXT, MD, or PDF and ask ABBot about it directly — it becomes part of what ABBot can search,
          right alongside the knowledge base, everywhere you talk to it.
        </p>
      ) : (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {documents.map((d) => (
            <span key={d.id} className="flex items-center gap-1.5 rounded-full bg-surface-2 px-2.5 py-1 text-xs text-muted">
              {d.filename}
              <button
                onClick={() => handleDelete(d.id)}
                aria-label={`Remove ${d.filename}`}
                className="text-muted hover:text-danger"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
