# Moondream Live Video Player

This demo shows how the [Moondream](https://moondream.ai/) VLM can analyze live video feeds in real time, trigger alerts for specific behaviors, and present the results in a lightweight web UI built with React and Vite.

## ✨ Features

- 🎥 **Real-time webcam/video analysis** - Stream from your camera or upload a video
- 💬 **Live visual summaries** - Continuous narration with custom prompts
- 🎯 **Gesture detection** - Pre-defined triggers (smiling, thumbs up, peace sign, etc.)
- ⚡ **Custom triggers** - Create your own action detectors
- 🖥️ **Fullscreen support** - Immersive viewing experience
- 💾 **Persistent storage** - Custom triggers saved in browser

## Run It Locally

1. **Install dependencies**

   ```bash
   pnpm install
   ```

2. **Add your Moondream API key** to `.env`, `.env.local`, or `.env.development` in the project root:

   ```bash
   MOONDREAM_API_KEY=sk-your-moondream-key
   ```

   You can generate keys from your Moondream dashboard. Keep this file out of version control.
3. **Start the Vite dev server**

   ```bash
   pnpm dev
   ```

   The app runs at <http://localhost:5173> by default. Allow camera access in the browser when prompted.
4. **(Optional) Test the production build**

   ```bash
   pnpm build && pnpm preview
   ```

### Tips

- Point the `Player` component’s `inferenceUrl` prop at a proxy or self-hosted Moondream deployment if you don’t want to call `https://api.moondream.ai/v1` directly.
- Need to test across devices? Run `pnpm dev --host` so phones on your LAN can reach the dev server.

## Code Architecture Overview

- **`src/App.tsx`** mounts the `Player` and injects the Moondream endpoint. This keeps the demo single-purpose and easy to embed elsewhere.
- **Player Orchestration (`src/components/solutions/analyze-live-video/player.tsx`)** wires together three domain-specific hooks:
  - `useVideoSession` manages the video element, fullscreen state, media permissions, and frame capture (built on top of `src/hooks/useLiveVideo.ts`).
  - `useTriggerManager` owns predefined + custom triggers, persists user-defined triggers to `localStorage`, and exposes the active Moondream queries/notification copy.
  - `useAnalysisSession` bridges the video session with `useFrameAnalysis` so captured frames run through Moondream and results are stored in a rolling history.
- **Frame Analysis Pipeline (`src/hooks/useFrameAnalysis.ts`)** debounces frame grabs, calls the Moondream `/query` endpoint with both the free-form summary prompt and the trigger prompt, retries on throttling, and emits structured results/notifications.
- **Presentation Layer (`src/components/solutions/analyze-live-video/player-surface.tsx` and overlays)** renders the actual video canvas plus:
  - `StartOverlay` for webcam/file selection.
  - `PromptControlsOverlay` for the free-form prompt + trigger picker (built with headless UI primitives).
  - `ResultOverlay` for the scrolling stack of answers/notifications so users can see what Moondream detected.
  - `CustomTriggerDialog` for composing new trigger rules without touching code.
- **Utility Hooks (`src/hooks/useLiveVideo.ts`)** encapsulate raw MediaStream handling, auto-mirroring for front cameras, video file ingestion, inactivity timeouts, and frame-to-canvas capture, keeping React components declarative.

Together these layers show how to: (1) capture live video, (2) query Moondream with flexible prompts, and (3) surface detections in a polished UI that end-users can tweak without redeploying code.
