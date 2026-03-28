'use client';

import React, { useState, useRef, useEffect } from 'react';

interface TooltipProps {
  text: string;
  children?: React.ReactNode;
}

export default function Tooltip({ text, children }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState<'above' | 'below'>('above');
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const tooltipId = useRef(`tooltip-${Math.random().toString(36).slice(2, 8)}`);

  useEffect(() => {
    if (visible && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setPosition(rect.top < 120 ? 'below' : 'above');
    }
  }, [visible]);

  return (
    <span className="relative inline-flex items-center">
      {children}
      <button
        ref={triggerRef}
        type="button"
        className="ml-1.5 w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold transition-all duration-200 cursor-help focus:outline-none focus:ring-2"
        style={{
          background: 'oklch(1 0 0 / 0.06)',
          color: 'var(--color-text-tertiary)',
          border: '1px solid oklch(1 0 0 / 0.08)',
          focusRingColor: 'oklch(0.78 0.15 195 / 0.3)',
        }}
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        aria-describedby={tooltipId.current}
      >
        ?
      </button>
      {visible && (
        <div
          ref={tooltipRef}
          id={tooltipId.current}
          role="tooltip"
          className="absolute z-50 px-3 py-2 text-xs leading-relaxed rounded-lg max-w-[260px] whitespace-normal pointer-events-none animate-fade-in-up"
          style={{
            background: 'oklch(0.15 0.01 260 / 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid oklch(1 0 0 / 0.1)',
            color: 'var(--color-text-secondary)',
            boxShadow: '0 8px 32px oklch(0 0 0 / 0.4)',
            left: '50%',
            transform: 'translateX(-50%)',
            ...(position === 'above'
              ? { bottom: 'calc(100% + 8px)' }
              : { top: 'calc(100% + 8px)' }),
            animationDuration: '0.15s',
          }}
        >
          {text}
        </div>
      )}
    </span>
  );
}
