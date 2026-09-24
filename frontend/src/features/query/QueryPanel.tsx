/** Natural-language questions and generated briefs over a job's findings. */

import { useState } from "react";

import { api, RequestError } from "@/api/client";
import { Spinner } from "@/components/Indicators";

interface Props {
  jobId: number | null;
  ready: boolean;
}

interface Answer {
  text: string;
  provider: string;
  model: string;
  latencyMs: number;
}

const SUGGESTIONS = [
  "What was detected in this scene?",
  "Which targets should be prioritised?",
  "Are there any high-confidence vessels?",
];

export function QueryPanel({ jobId, ready }: Props) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(task: () => Promise<Answer>) {
    setIsBusy(true);
    setError(null);

    try {
      setAnswer(await task());
    } catch (cause) {
      setError(
        cause instanceof RequestError
          ? cause.message
          : "The request could not be completed",
      );
    } finally {
      setIsBusy(false);
    }
  }

  async function ask(text: string) {
    if (jobId === null || !text.trim()) {
      return;
    }
    await run(async () => {
      const result = await api.query(jobId, text.trim());
      return {
        text: result.answer,
        provider: result.provider,
        model: result.model,
        latencyMs: result.latency_ms,
      };
    });
  }

  async function requestBrief() {
    if (jobId === null) {
      return;
    }
    await run(async () => {
      const result = await api.brief(jobId);
      return {
        text: result.brief,
        provider: result.provider,
        model: result.model,
        latencyMs: result.latency_ms,
      };
    });
  }

  return (
    <div className="card">
      <h2>Ask the analyst</h2>

      {!ready ? (
        <p className="empty" style={{ padding: "12px 0" }}>
          Available once analysis completes.
        </p>
      ) : (
        <>
          <div className="field">
            <textarea
              rows={3}
              placeholder="Ask about this scene..."
              value={question}
              disabled={isBusy}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void ask(question);
                }
              }}
            />
          </div>

          <div className="row" style={{ marginBottom: 10 }}>
            <button
              type="button"
              className="primary"
              disabled={isBusy || !question.trim()}
              onClick={() => void ask(question)}
            >
              Ask
            </button>
            <button type="button" disabled={isBusy} onClick={() => void requestBrief()}>
              Generate brief
            </button>
          </div>

          <div className="row" style={{ flexWrap: "wrap", marginBottom: 10 }}>
            {SUGGESTIONS.map((text) => (
              <button
                key={text}
                type="button"
                disabled={isBusy}
                style={{ padding: "3px 8px", fontSize: 11 }}
                onClick={() => {
                  setQuestion(text);
                  void ask(text);
                }}
              >
                {text}
              </button>
            ))}
          </div>

          {isBusy && (
            <p className="row muted" style={{ fontSize: 13 }}>
              <Spinner /> Consulting the model
            </p>
          )}

          {error && <p className="error-text">{error}</p>}

          {answer && !isBusy && (
            <>
              <p className="answer">{answer.text}</p>
              <p className="provider-tag" style={{ margin: "6px 0 0" }}>
                {answer.provider} · {answer.model} · {answer.latencyMs} ms
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
