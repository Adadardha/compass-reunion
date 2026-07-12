import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, Lock, Scale, X } from 'lucide-react';
import { useLanguage } from '../i18n';

// -----------------------------------------------------------------------------
// Responsible AI & Guardrails drawer — bilingual copy kept local to this
// component, mirroring the UI_CONTENT dictionary pattern used in Index.tsx.
// -----------------------------------------------------------------------------
const DRAWER_CONTENT = {
  en: {
    title: 'Responsible AI & Guardrails',
    subtitle: 'How Digital Compass protects students',
    privacyTitle: 'Data Privacy — Zero Cloud Storage',
    privacyText:
      'Student voice transcripts captured through the Web Speech API are processed entirely in the browser and are never uploaded or stored in the cloud. Quiz answers and interview responses exist only in your local session — nothing personally identifiable leaves your device without your explicit action.',
    biasTitle: 'Bias & Profiling Guardrails',
    biasText:
      'Our system prompts are engineered to explicitly prevent demographic profiling. The AI is instructed never to weigh gender, ethnicity, region of origin, or socioeconomic background when mapping careers — so youth from every Albanian region receive the same quality of guidance, free from algorithmic bias.',
    close: 'Close',
  },
  al: {
    title: 'Etika e IA & Mbrojtja',
    subtitle: 'Si i mbron studentët Busulla Digjitale',
    privacyTitle: 'Privatësia e të Dhënave — Zero Ruajtje në Cloud',
    privacyText:
      'Transkriptet zanore të studentëve, të kapura përmes Web Speech API, përpunohen tërësisht në shfletues dhe nuk ngarkohen e as ruhen kurrë në cloud. Përgjigjet e kuizit dhe të intervistës ekzistojnë vetëm në sesionin tuaj lokal — asgjë personale nuk largohet nga pajisja juaj pa veprimin tuaj të qartë.',
    biasTitle: 'Mbrojtja nga Paragjykimet & Profilizimi',
    biasText:
      'Udhëzimet tona të sistemit janë ndërtuar për të parandaluar shprehimisht profilizimin demografik. IA është e instruktuar të mos marrë kurrë parasysh gjininë, etninë, rajonin e origjinës apo prejardhjen socio-ekonomike gjatë orientimit të karrierës — që të rinjtë e çdo rajoni shqiptar të marrin të njëjtën cilësi udhëzimi, pa paragjykime algoritmike.',
    close: 'Mbyll',
  },
} as const;

interface ResponsibleAIDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const ResponsibleAIDrawer: React.FC<ResponsibleAIDrawerProps> = ({ isOpen, onClose }) => {
  const { lang } = useLanguage();
  const t = DRAWER_CONTENT[lang];

  // Close on Escape for accessibility
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 bg-background/70 backdrop-blur-sm z-[60] print:hidden"
            aria-hidden="true"
          />

          {/* Right-side sliding drawer */}
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-label={t.title}
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'tween', duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
            className="fixed top-0 right-0 h-full w-full max-w-md z-[70] bg-background border-l border-border shadow-2xl overflow-y-auto print:hidden"
          >
            <div className="p-6 md:p-8 space-y-6">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <h2 className="font-heading font-bold text-xl md:text-2xl uppercase tracking-tight flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 shrink-0 text-accent" />
                    {t.title}
                  </h2>
                  <p className="text-sm text-muted-foreground">{t.subtitle}</p>
                </div>
                <button
                  onClick={onClose}
                  aria-label={t.close}
                  className="p-2 border border-border hover:bg-foreground hover:text-background transition-all shrink-0"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="border-t border-border" />

              <section className="space-y-3">
                <h3 className="text-xs uppercase tracking-widest font-bold flex items-center gap-2 text-accent">
                  <Lock className="w-4 h-4 shrink-0" />
                  {t.privacyTitle}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{t.privacyText}</p>
              </section>

              <section className="space-y-3">
                <h3 className="text-xs uppercase tracking-widest font-bold flex items-center gap-2 text-accent">
                  <Scale className="w-4 h-4 shrink-0" />
                  {t.biasTitle}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{t.biasText}</p>
              </section>

              <button
                onClick={onClose}
                className="w-full p-3 brutalist-border hover:bg-foreground/10 transition-all font-bold uppercase text-xs tracking-widest"
              >
                {t.close}
              </button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
};

export default ResponsibleAIDrawer;
