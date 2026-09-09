/**
 * api.js — Echora API client
 *
 * Set VITE_API_URL in .env.local for local dev:
 *   VITE_API_URL=http://localhost:8000
 *
 * In production (Vercel), set the env var to your HF Spaces URL:
 *   VITE_API_URL=https://<your-hf-space>.hf.space
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Fetch the list of supported target languages from the backend.
 * @returns {Promise<Array<{code: string, name: string}>>}
 */
export async function fetchLanguages() {
  const res = await fetch(`${API_BASE}/languages`);
  if (!res.ok) throw new Error(`Failed to fetch languages: ${res.status}`);
  const data = await res.json();
  return data.languages;
}

/**
 * Upload a video file and start the dubbing job.
 * @param {File} videoFile
 * @param {string} targetLanguage  — BCP-47 code (e.g. "en", "hi")
 * @returns {Promise<string>} job_id
 */
export async function uploadVideo(videoFile, targetLanguage) {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("target_language", targetLanguage);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }

  const data = await res.json();
  return data.job_id;
}

/**
 * Poll job status.
 * @param {string} jobId
 * @returns {Promise<{job_id: string, status: string, error_message: string|null}>}
 */
export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Status check failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Return the download URL for a completed job.
 * (Used as an <a href> — no JS download needed.)
 * @param {string} jobId
 * @returns {string}
 */
export function getDownloadUrl(jobId) {
  return `${API_BASE}/download/${jobId}`;
}
