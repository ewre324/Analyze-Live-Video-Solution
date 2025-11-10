import { useCallback, useEffect, useRef, useState } from 'react';

const rawMoondreamKey =
  (typeof __MOONDREAM_API_KEY__ === 'string' ? __MOONDREAM_API_KEY__ : '') ||
  import.meta.env.VITE_MOONDREAM_API_KEY ||
  '';
const MOONDREAM_API_KEY = rawMoondreamKey.trim();

export interface FrameAnalysisConfig {
  query1: string;
  query2: string;
  triggerText: string;
  notificationText: string;
}

export interface FrameAnalysisResult {
  id: number;
  text: string;
  isNotification?: boolean;
}

interface UseFrameAnalysisOptions {
  inferenceUrl: string;
  captureFrame: () => string | null;
  config: FrameAnalysisConfig;
  shouldAnalyze: boolean;
  historyLimit?: number;
  analysisDelayMs?: number;
  onError?: (message: string) => void;
}

interface UseFrameAnalysisReturn {
  resultHistory: FrameAnalysisResult[];
  clearHistory: () => void;
}

export function useFrameAnalysis({
  inferenceUrl,
  captureFrame,
  config,
  shouldAnalyze,
  historyLimit = 3,
  analysisDelayMs = 100,
  onError,
}: UseFrameAnalysisOptions): UseFrameAnalysisReturn {
  const [resultHistory, setResultHistory] = useState<FrameAnalysisResult[]>([]);
  const analyzeFrameRef = useRef<() => Promise<void> | null>(null);

  const runInference = useCallback(
    async (imageData: string, question: string, retryCount = 0): Promise<string> => {
      const sanitizedBase = inferenceUrl?.replace(/\/$/, '') || 'https://api.moondream.ai/v1';
      const endpoint = sanitizedBase.endsWith('/query') ? sanitizedBase : `${sanitizedBase}/query`;
      const maxRetries = 3;
      const apiKey = MOONDREAM_API_KEY;

      if (!apiKey) {
        throw new Error('Missing Moondream API key. Set MOONDREAM_API_KEY in your .env file.');
      }

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Moondream-Auth': apiKey,
          },
          body: JSON.stringify({
            image_url: imageData,
            question,
          }),
        });

        if (!response.ok) {
          if (response.status === 429 && retryCount < maxRetries) {
            const backoffMs = Math.min(1000 * Math.pow(2, retryCount), 8000);
            console.log('Backoff for', backoffMs, 'ms before retrying...');
            await new Promise(resolve => setTimeout(resolve, backoffMs));
            return runInference(imageData, question, retryCount + 1);
          }
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        return data.answer || '';
      } catch (error) {
        if (error instanceof Error && error.message.includes('Rate limit')) {
          throw error;
        }
        throw new Error('Network error. Please check your connection.');
      }
    },
    [inferenceUrl]
  );

  const analyzeFrame = useCallback(async () => {
    onError?.('');

    const frame = captureFrame();
    if (!frame) {
      return;
    }

    const effectiveQuery1 = config.query1.trim() || 'summarize what you see in one short sentence';

    const [answer1, answer2] = await Promise.all([
      runInference(frame, effectiveQuery1),
      runInference(frame, config.query2),
    ]);

    const triggerMatch = config.triggerText.trim();
    const detected = triggerMatch
      ? answer2.toLowerCase().includes(triggerMatch.toLowerCase())
      : false;

    setResultHistory(prev => {
      let newEntry: FrameAnalysisResult | null = null;

      if (detected) {
        newEntry = { id: Date.now(), text: config.notificationText, isNotification: true };
      } else if (answer1) {
        newEntry = { id: Date.now(), text: answer1 };
      }

      if (!newEntry) {
        return prev;
      }

      return [newEntry, ...prev].slice(0, historyLimit);
    });
  }, [captureFrame, config, historyLimit, onError, runInference]);

  useEffect(() => {
    analyzeFrameRef.current = analyzeFrame;
  }, [analyzeFrame]);

  useEffect(() => {
    if (!shouldAnalyze) {
      return undefined;
    }

    let shouldContinue = true;

    const loop = async () => {
      while (shouldContinue) {
        try {
          if (analyzeFrameRef.current) {
            await analyzeFrameRef.current();
          }
        } catch (error) {
          console.error('Analysis error:', error);
          onError?.(error instanceof Error ? error.message : 'Analysis failed');
        }

        await new Promise(resolve => setTimeout(resolve, analysisDelayMs));
      }
    };

    loop();

    return () => {
      shouldContinue = false;
    };
  }, [analysisDelayMs, onError, shouldAnalyze]);

  const clearHistory = useCallback(() => setResultHistory([]), []);

  return {
    resultHistory,
    clearHistory,
  };
}
