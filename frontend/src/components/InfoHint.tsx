/**
 * Inline explanation for a metric or control.
 *
 * Several figures on screen are only meaningful if you know how they were
 * produced. Rather than crowd the layout with prose, each sits behind a marker
 * that reveals the explanation on demand.
 */

import { useEffect, useId, useRef, useState } from "react";

interface Props {
  children: string;
}

export function InfoHint({ children }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement | null>(null);
  const describedBy = useId();

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    function dismiss(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    function dismissOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", dismiss);
    document.addEventListener("keydown", dismissOnEscape);
    return () => {
      document.removeEventListener("mousedown", dismiss);
      document.removeEventListener("keydown", dismissOnEscape);
    };
  }, [isOpen]);

  return (
    <span className="hint" ref={containerRef}>
      <button
        type="button"
        className="hint-trigger"
        aria-label="Explain this"
        aria-expanded={isOpen}
        aria-describedby={isOpen ? describedBy : undefined}
        onClick={() => setIsOpen((open) => !open)}
      >
        i
      </button>
      {isOpen && (
        <span className="hint-body" id={describedBy} role="tooltip">
          {children}
        </span>
      )}
    </span>
  );
}
