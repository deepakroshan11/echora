/**
 * ShortsDub API client
 * All API calls centralized here. Set VITE_API_URL env var to point at your backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Fetch supported target languages from the backend.
 * @returns {Promise<Array<{code: string, name: string}>>}
 */
export async function fetchLanguages() {
  const res = await fetch(`${API_BASE}/languages`);
  if (!res.ok) throw new Error(`Failed to fetch languages: ${res.status}`);
  const data = await res.json();
  return data.languages;
}

/**
 * Upload a video file for dubbing.
 * @param {File} file - Video file to upload
 * @param {string} targetLang - BCP-47 target language code (e.g. "en")
 * @returns {Promise<{job_id: string, status: string, message: string}>}
 */
export async function uploadVideo(file, targetLang) {
  const form = new FormData();
  form.append("video", file);
  form.append("target_language", targetLang);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Poll job status.
 * @param {string} jobId
 * @returns {Promise<{job_id: string, status: string, error_message?: string}>}
 */
export async function pollStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}

/**
 * Get the download URL for a completed job.
 * @param {string} jobId
 * @returns {string} Direct download URL
 */
export function getDownloadUrl(jobId) {
  return `${API_BASE}/download/${jobId}`;
}

/**
 * Check if the backend is alive (with timeout).
 * Returns false if the server is sleeping (HF Spaces cold start).
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}
