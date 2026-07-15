import { useEffect, useRef, useState } from 'react';
import { getLanguage, type Language } from '../i18n';

/**
 * Web Speech API hook. Locale is explicitly bound to the active language
 * ('en' -> 'en-US', 'al' -> 'sq-AL'). Falls back to the i18n module state
 * when no explicit lang is provided. onend/onerror always flip listening
 * back to false so the UI cannot get stuck in a "listening" state after
 * silence, permission denial, or network interrupts.
 */
export function useSpeechRecognition(
  onFinal: (transcript: string) => void,
  onInterim: (transcript: string) => void,
  currentLang?: Language,
) {
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(true);
  const recognitionRef = useRef<any>(null);
  const onFinalRef = useRef(onFinal);
  const onInterimRef = useRef(onInterim);
  const langRef = useRef<Language | undefined>(currentLang);

  useEffect(() => { onFinalRef.current = onFinal; }, [onFinal]);
  useEffect(() => { onInterimRef.current = onInterim; }, [onInterim]);
  useEffect(() => { langRef.current = currentLang; }, [currentLang]);

  const resolveLocale = (override?: string): string => {
    if (override) return override;
    const l = langRef.current ?? getLanguage();
    return l === 'en' ? 'en-US' : 'sq-AL';
  };

  useEffect(() => {
    const SR: any = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) { setSupported(false); return; }
    const rec = new SR();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = resolveLocale();
    rec.onresult = (event: any) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      if (interim) onInterimRef.current(interim);
      if (final) onFinalRef.current(final);
    };
    // Safety: any terminal state must reset the UI listening flag.
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    rec.onnomatch = () => setListening(false);
    recognitionRef.current = rec;
    return () => { try { rec.stop(); } catch {} };
  }, []);

  const start = (langOverride?: string) => {
    const rec = recognitionRef.current;
    if (!rec) return;
    try {
      rec.lang = resolveLocale(langOverride);
      rec.start();
      setListening(true);
    } catch {
      setListening(false);
    }
  };
  const stop = () => {
    const rec = recognitionRef.current;
    if (!rec) return;
    try { rec.stop(); } catch {}
    setListening(false);
  };

  return { listening, supported, start, stop };
}
