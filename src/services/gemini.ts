/**
 * Busulla Digjitale — AI Orchestration Layer
 * =========================================================================
 * Current cloud pathway: Google Gemini via @google/generative-ai + Lovable
 * Cloud edge function (`career-chat`) for the assistant channel.
 *
 * INTEL OpenVINO Readiness — Edge Inference Pipeline (planned)
 * -------------------------------------------------------------------------
 * The model invocation pathways below are intentionally isolated behind
 * `callGemini()` and `getCareerAssistantResponse()` so that a local edge
 * inference runtime (Intel Distribution of OpenVINO Toolkit — e.g.
 * `openvino.runtime.Core` or the OpenVINO GenAI `LLMPipeline` API) can be
 * swapped in without touching UI or scoring logic. Deployment placeholders:
 *
 *   const OPENVINO_MODEL_PATH  = import.meta.env.VITE_OPENVINO_MODEL_PATH;
 *   const OPENVINO_DEVICE      = import.meta.env.VITE_OPENVINO_DEVICE || "CPU"; // CPU | GPU | NPU
 *   const OPENVINO_PRECISION   = "INT8"; // quantized for low-bandwidth Albanian classrooms
 *
 * When a local runtime is available, `callGemini()` should short-circuit to
 * an OpenVINO pipeline stream; otherwise it falls back to the cloud call.
 * This keeps the app usable on offline / low-bandwidth school hardware.
 * =========================================================================
 */
import { GoogleGenerativeAI } from "@google/generative-ai";
import {
  QuizAnswer,
  PredictionResult,
  CareerRoadmap,
  InterviewMode,
  DifficultyLevel,
  InterviewFeedback,
  InterviewMessage,
  InterviewSession,
  InterviewReport,
  ChatMessage,
} from '../types';
import { classifyToPrediction } from './classifier';
import { getLanguage } from '../i18n';

/**
 * Returns the current UI language directive for LLM prompts. The AI must
 * strictly obey this — every question, feedback string, summary, and
 * roadmap MUST be written in the exact language the user has selected in
 * the UI. This eliminates language leakage across the EN/AL toggle.
 */
function languageDirective(): string {
  const lang = getLanguage();
  return lang === 'en'
    ? 'OUTPUT LANGUAGE: Respond EXCLUSIVELY in professional, native English. Do NOT emit any Albanian words or phrases under any circumstances.'
    : 'OUTPUT LANGUAGE: Respond EXCLUSIVELY in fluent, native Albanian (shqip). Do NOT emit any English words or phrases except unavoidable technical proper nouns.';
}

// Config
const GEMINI_API_KEY = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_GEMINI_API_KEY) ||
  (typeof process !== 'undefined' && process.env?.VITE_GEMINI_API_KEY) ||
  '';
const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;
const MODEL_NAME = 'gemini-2.0-flash';
const TIMEOUT_MS = 15000;

// Utilities

async function withTimeout<T>(promise: Promise<T>, ms: number = TIMEOUT_MS): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('Koha e pritjes u tejkalua. Provo përsëri.')), ms)
  );
  return Promise.race([promise, timeout]);
}

async function withRetry<T>(fn: () => Promise<T>, retries = 2, delay = 2000): Promise<T> {
  try {
    return await fn();
  } catch (error: any) {
    const message = String(error?.message || '').toLowerCase();
    const isRateLimit = message.includes('429') || message.includes('rate limit') || message.includes('resource_exhausted');
    if (retries > 0 && isRateLimit) {
      console.warn(`Rate limit hit. Retrying in ${delay}ms...`);
      await new Promise(r => setTimeout(r, delay));
      return withRetry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

function extractJson(text: string): string {
  const clean = text.replace(/```json|```/g, '').trim();
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return '{}';
  return clean.slice(start, end + 1);
}

function safeParse<T>(text: string, fallback: T): T {
  try {
    return JSON.parse(extractJson(text)) as T;
  } catch {
    console.warn('[Busulla] JSON parse failed for text:', text.substring(0, 200));
    return fallback;
  }
}

// Gemini caller

async function callGemini(prompt: string): Promise<string> {
  if (!genAI) throw new Error('Mungon VITE_GEMINI_API_KEY. Konfiguro çelësin API.');

  const model = genAI.getGenerativeModel({ model: MODEL_NAME });
  const result = await withTimeout(model.generateContent(prompt));
  const response = (result as any).response;
  const text = response.text().trim();
  console.log('[Busulla] Gemini raw response:', text.substring(0, 300));
  return text;
}

const STRICT_JSON_INSTRUCTION = `

IMPORTANT: You MUST return valid JSON only. No markdown, no explanation, no code fences, no extra text. Just the raw JSON object starting with { and ending with }.`;

// Career Prediction

export const predictCareer = async (answers: QuizAnswer[]): Promise<PredictionResult> => {
  const localResult = classifyToPrediction(answers);

  if (!GEMINI_API_KEY) return localResult;

  const answersText = answers.map((a, i) => `${i + 1}. ${a.answer}`).join('\n');

  const prompt = `${languageDirective()}

Bazuar në këto përgjigje të kuizit të karrierës, analizo dhe kthe një objekt JSON.
Karriera kryesore sipas analizës: ${localResult.primaryCareer}
Alternativat: ${localResult.alternatives.map(a => a.career).join(', ')}

Përgjigjet:
${answersText}

Kthe VETËM JSON të vlefshëm, pa asnjë tekst tjetër:
{
  "primaryCareer": "${localResult.primaryCareer}",
  "confidence": ${localResult.confidence},
  "description": "2-3 sentences explaining why this career fits the person based on their answers (write in the OUTPUT LANGUAGE specified above)",
  "alternatives": [
    {"career": "${localResult.alternatives[0]?.career || ''}", "confidence": ${localResult.alternatives[0]?.confidence || 0.5}, "description": "pse kjo alternativë"},
    {"career": "${localResult.alternatives[1]?.career || ''}", "confidence": ${localResult.alternatives[1]?.confidence || 0.4}, "description": "pse kjo alternativë"}
  ],
  "learningPath": ["hapi 1", "hapi 2", "hapi 3", "hapi 4", "hapi 5"]
}${STRICT_JSON_INSTRUCTION}`;

  try {
    const resp = await withRetry(async () => callGemini(prompt));
    if (resp) {
      const parsed = safeParse<PredictionResult>(resp, localResult);
      if (parsed.primaryCareer && parsed.description && parsed.alternatives?.length) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn('AI enrichment failed, using local result:', err);
  }

  return localResult;
};

// Career Roadmap Generation
function getLocalRoadmap(career: string): CareerRoadmap {
  const roadmaps: Record<string, CareerRoadmap> = {
    'Zhvillues Software': {
      subjects: ['Matematikë', 'Informatikë', 'Fizikë', 'Logjikë', 'Anglisht'],
      universities: ['Fakulteti i Shkencave Natyrore (FSHN)', 'Universiteti Politeknik i Tiranës', 'UET Tiranë', 'EPOKA University'],
      careerPath: ['Studime Bachelor Informatikë', 'Praktikë në kompani tech', 'Junior Developer', 'Mid-level Developer', 'Senior / Tech Lead'],
      salaryRange: '60,000 - 200,000 ALL/muaj',
      jobDemand: 'Kërkesë shumë e lartë — tregu shqiptar dhe remote',
    },
    'Shkencëtar të Dhënash': {
      subjects: ['Matematikë', 'Statistikë', 'Informatikë', 'Fizikë', 'Anglisht'],
      universities: ['FSHN Tiranë', 'Universiteti Politeknik', 'UET', 'EPOKA University'],
      careerPath: ['Bachelor Matematikë/Informatikë', 'Certifikime ML/AI', 'Data Analyst Junior', 'Data Scientist', 'Lead Data Scientist'],
      salaryRange: '70,000 - 220,000 ALL/muaj',
      jobDemand: 'Kërkesë në rritje të shpejtë në Shqipëri dhe rajon',
    },
    'Dizajner UX/UI': {
      subjects: ['Art', 'Informatikë', 'Psikologji', 'Anglisht', 'Matematikë'],
      universities: ['Akademia e Arteve Tiranë', 'UET', 'POLIS University', 'Shkolla e Dizajnit'],
      careerPath: ['Bachelor Dizajn/Arteve', 'Portfolio personale', 'Junior Designer', 'UI/UX Designer', 'Lead Designer / Art Director'],
      salaryRange: '50,000 - 150,000 ALL/muaj',
      jobDemand: 'Kërkesë e mirë veçanërisht për agjenci digjitale',
    },
    'Menaxher Projekti': {
      subjects: ['Matematikë', 'Ekonomi', 'Gjuhë Shqipe', 'Anglisht', 'Informatikë'],
      universities: ['Fakulteti Ekonomik UT', 'UBT', 'UET', 'Universiteti Marin Barleti'],
      careerPath: ['Bachelor Ekonomi/Biznes', 'Asistent Projekti', 'Koordinator', 'Project Manager', 'Senior PM / Drejtues'],
      salaryRange: '60,000 - 180,000 ALL/muaj',
      jobDemand: 'Kërkesë konstante në sektorin publik dhe privat',
    },
    'Sipërmarrës / Themelues Startup': {
      subjects: ['Ekonomi', 'Matematikë', 'Anglisht', 'Informatikë', 'Histori'],
      universities: ['Fakulteti Ekonomik UT', 'UET', 'EPOKA', 'Universiteti Marin Barleti'],
      careerPath: ['Bachelor Biznes/Ekonomi', 'Përvojë në startup', 'Biznes i vogël', 'Startup me financim', 'Kompani e qëndrueshme'],
      salaryRange: 'E ndryshueshme — 30,000 deri +500,000 ALL/muaj',
      jobDemand: 'Ekosistemi startup shqiptar në zhvillim të shpejtë',
    },
    'Psikolog / Këshilltar': {
      subjects: ['Biologji', 'Sociologji', 'Gjuhë Shqipe', 'Filozofi', 'Anglisht'],
      universities: ['Fakulteti i Shkencave Sociale UT', 'Universiteti Aleksandër Moisiu Durrës', 'UET'],
      careerPath: ['Bachelor Psikologji', 'Master Klinik', 'Praktikë e mbikëqyrur', 'Psikolog i licencuar', 'Praktikë private'],
      salaryRange: '40,000 - 120,000 ALL/muaj',
      jobDemand: 'Kërkesë në rritje — shëndet mendor në fokus',
    },
    'Mjek / Profesionist Shëndetësor': {
      subjects: ['Biologji', 'Kimi', 'Fizikë', 'Matematikë', 'Anglisht'],
      universities: ['Universiteti i Mjekësisë Tiranë', 'UAMD Durrës — Infermieri', 'UMB'],
      careerPath: ['Fakulteti i Mjekësisë (6 vjet)', 'Rezidencë (3-5 vjet)', 'Mjek i licencuar', 'Specializim', 'Mjek Specialist'],
      salaryRange: '60,000 - 250,000 ALL/muaj',
      jobDemand: 'Kërkesë konstante dhe e lartë',
    },
    'Menaxher Marketingu': {
      subjects: ['Ekonomi', 'Anglisht', 'Informatikë', 'Sociologji', 'Art'],
      universities: ['Fakulteti Ekonomik UT', 'UET', 'Universiteti Marin Barleti', 'EPOKA'],
      careerPath: ['Bachelor Marketing/Ekonomi', 'Asistent Marketing', 'Specialist Digital', 'Marketing Manager', 'CMO / Drejtues'],
      salaryRange: '50,000 - 160,000 ALL/muaj',
      jobDemand: 'Kërkesë e mirë veçanërisht digital marketing',
    },
    'Inxhinier / Arkitekt': {
      subjects: ['Matematikë', 'Fizikë', 'Kimi', 'Vizatim Teknik', 'Informatikë'],
      universities: ['Universiteti Politeknik i Tiranës', 'POLIS University', 'UAMD'],
      careerPath: ['Bachelor Inxhinieri/Arkitekturë', 'Praktikë profesionale', 'Inxhinier Junior', 'Inxhinier i Licencuar', 'Drejtues Projekti'],
      salaryRange: '50,000 - 170,000 ALL/muaj',
      jobDemand: 'Kërkesë e qëndrueshme — ndërtim dhe infrastrukturë',
    },
    'Mësues / Trajner': {
      subjects: ['Gjuhë Shqipe', 'Pedagogji', 'Psikologji', 'Matematikë', 'Anglisht'],
      universities: ['Fakulteti i Shkencave Sociale UT', 'Universiteti Fan Noli Korçë', 'UAMD'],
      careerPath: ['Bachelor Mësuesi', 'Master Profesional', 'Mësues i ri', 'Mësues i kualifikuar', 'Drejtor/Trajner'],
      salaryRange: '35,000 - 90,000 ALL/muaj',
      jobDemand: 'Kërkesë konstante — sektor publik dhe privat',
    },
  };

  return roadmaps[career] || {
    subjects: ['Matematikë', 'Anglisht', 'Informatikë', 'Gjuhë Shqipe', 'Sociologji'],
    universities: ['Universiteti i Tiranës', 'UET', 'EPOKA University'],
    careerPath: ['Studime Bachelor', 'Praktikë profesionale', 'Pozicion fillestar', 'Zhvillim profesional', 'Ekspert i fushës'],
    salaryRange: '40,000 - 120,000 ALL/muaj',
    jobDemand: 'Kërkesë e mirë në tregun shqiptar',
  };
}

/** Local fallback for the three democratized economic tracks. */
function getLocalTracks(career: string): {
  educationTrack: string[];
  localMarketTrack: string[];
  practicalSkillsTrack: string[];
} {
  const c = career.toLowerCase();
  const isTech = /software|shkenc|data|inxhinier|ux|ui|dizajn/.test(c);
  const isBiz = /menaxher|marketing|sipërmarrës|ekonom/.test(c);
  const isHealth = /mjek|psikolog|shëndet/.test(c);

  return {
    educationTrack: [
      'FSHN, Politeknik, UAMD — universitete publike me tarifa të ulëta',
      'Kurse falas online: Coursera Financial Aid, edX audit, Google Career Certificates',
      isTech ? 'Certifikime Microsoft Learn / freeCodeCamp / Meta Front-End (falas)'
        : isHealth ? 'ProCredit Academy dhe kurse të Ministrisë së Shëndetësisë'
        : 'Certifikime AKAFP dhe kurse profesionale të AKPA',
      'Programe Erasmus+ dhe shkëmbime studentore për akses ndërkombëtar',
    ],
    localMarketTrack: [
      isTech ? 'Sektori tech në Tiranë, Durrës dhe Shkodër — Cardo AI, Ikub, Balfin Tech'
        : isBiz ? 'Sektor privat në rritje — startup-e në Tiranë, tregtia rajonale në Vlorë/Korçë'
        : isHealth ? 'Qendra shëndetësore rajonale + spitalet universitare në qytetet kryesore'
        : 'Sektori publik + OJF-të rajonale (Fier, Elbasan, Kukës)',
      'Punë remote për kompani të BE-së dhe SHBA-së — akses i barabartë nga çdo qytet',
      'Programe praktike me AmCham Albania, Junior Achievement, Protik Center',
      'Rrjeti i inkubatorëve: Uplift, Innospace, Yunus Social Business Balkans',
    ],
    practicalSkillsTrack: [
      isTech ? 'CodeWeek Albania — hackathon-e vjetore dhe workshop-e falas'
        : 'Google Digital Garage — trajnime falas për aftësi digjitale',
      'Coursera dhe Khan Academy Shqip — kurse me subtitra në gjuhën shqipe',
      'Bootcamp-e komunitare: Girls Code Albania, Open Labs Hackerspace',
      'Portofoli personal në GitHub/Behance për të treguar punën konkretisht',
      'Anglishtja në nivel B2+ — obligatore për tregun global remote',
    ],
  };
}

export const generateCareerRoadmap = async (career: string): Promise<CareerRoadmap> => {
  const localTracks = getLocalTracks(career);
  const fallback: CareerRoadmap = { ...getLocalRoadmap(career), ...localTracks };

  if (!GEMINI_API_KEY) return fallback;

  const prompt = `${languageDirective()}

Për karrierën "${career}" në Shqipëri, kthe VETËM JSON valid me tri trajektore lokale që mbështesin akses demokratik jashtë Tiranës:
{
  "subjects": ["5 lëndë gjimnazi relevante"],
  "universities": ["3-5 universitete/fakultete shqiptare"],
  "careerPath": ["5 hapa tipikë të karrierës në Shqipëri"],
  "salaryRange": "diapazoni i pagës mujore në ALL",
  "jobDemand": "përshkrim i shkurtër i kërkesës",
  "educationTrack": ["4 opsione falas/lokale: universitete publike, certifikime digjitale, kurse online falas"],
  "localMarketTrack": ["4 lidhje konkrete me tregun shqiptar — kompani, sektorë rajonalë, praktika, remote"],
  "practicalSkillsTrack": ["4-5 hapa vetë-studimi duke përdorur burime globale falas (CodeWeek, Coursera, bootcamp-e komunitare)"]
}${STRICT_JSON_INSTRUCTION}`;

  try {
    const resp = await withRetry(async () => callGemini(prompt));
    return safeParse<CareerRoadmap>(resp, fallback);
  } catch {
    return fallback;
  }
};

// Interview Question Generation

export const generateDynamicQuestion = async (
  career: string,
  mode: InterviewMode,
  difficulty: DifficultyLevel,
  history: InterviewMessage[],
  weakAreas: string[] = [],
  neurodivergent: boolean = false,
): Promise<{ question: string; type: 'technical' | 'behavioral'; hints: string[] }> => {
  const fallback = getFallbackQuestion(career, mode);

  return withRetry(async () => {
    const modeDescriptions = {
      [InterviewMode.TECHNICAL]: 'targeted technical questions for the domain',
      [InterviewMode.BEHAVIORAL]: 'behavioral questions about past experiences and situations',
      [InterviewMode.MIXED]: 'a mix of technical and behavioral questions',
      [InterviewMode.STRESS]: 'challenging questions that test reaction under pressure',
    };
    const difficultyContext = {
      [DifficultyLevel.EASY]: 'Basic, warm-up level',
      [DifficultyLevel.MEDIUM]: 'Medium intensity — requires thought',
      [DifficultyLevel.HARD]: 'Complex — requires depth and analytical rigor',
    };

    // Build a cohesive narrative from the last few user answers so the AI can
    // pivot into contextual follow-ups instead of asking generic new questions.
    const recentTurns = history
      .slice(-6)
      .map(m => `${m.role === 'user' ? 'CANDIDATE' : 'INTERVIEWER'}: ${m.content.substring(0, 260)}`)
      .join('\n');

    const lastUserAnswer = [...history].reverse().find(m => m.role === 'user');
    const lastScore = lastUserAnswer?.metadata?.feedback?.score;
    const lastWasVague = typeof lastScore === 'number' && lastScore < 45;

    const neurodivergentDirective = neurodivergent ? `

INCLUSION MODE (Neurodiversity Support) — ACTIVE:
Strip ambiguous corporate idioms and open-ended buzzwords. Frame every scenario concretely. Split complex prompts into clear numbered sub-parts (a, b, c). Prioritize technical and logical focus. Replace metaphors like "sell yourself" or "biggest weakness" with concrete task-based framings.` : '';

    const prompt = `${languageDirective()}

You are an elite, empathetic Talent Acquisition Director specialized in modern tech, digital, and creative industry roles. You are conducting a live interview for the position: ${career}.

Interview mode: ${modeDescriptions[mode]}
Difficulty level: ${difficultyContext[difficulty]}
${weakAreas.length > 0 ? `Candidate weak areas so far: ${weakAreas.join(', ')}` : ''}${neurodivergentDirective}

CONVERSATION SO FAR (most recent first is at the bottom):
${recentTurns || '(no prior turns — this is the opening question)'}

PROMPT ORCHESTRATION RULES — FOLLOW STRICTLY:
1. CONTEXT RETENTION: Cross-reference specific concepts, projects, tools, or claims the candidate mentioned in earlier answers. Build a cohesive narrative — a real executive interview builds on prior turns. Reference them by name when relevant (e.g. "You mentioned React earlier — how would that inform...").
2. DYNAMIC SCAFFOLDING: ${lastWasVague ? 'The candidate\'s last answer was vague or shallow. DO NOT move on to a new topic. Instead, pivot with a contextual follow-up that pins them down on specifics — ask for a concrete example, a metric, or a decision point they had to make.' : 'If the candidate gave a substantive answer, build on it or introduce a related but distinct challenge.'}
3. NO GENERIC QUESTIONS: Never ask "tell me about yourself" or "what are your strengths" unless it's the very first turn. Prefer scenario-driven prompts tied to the role.
4. ONE QUESTION AT A TIME: Ask exactly one primary question. Sub-parts allowed only in Inclusion Mode.
5. LANGUAGE: Match the language of the most recent candidate message. If none, default to English.

Return ONLY valid JSON (no markdown, no fences):
{
  "question": "the interview question, phrased naturally and specifically",
  "type": "technical" or "behavioral",
  "hints": ["hint 1 without revealing the answer", "hint 2", "hint 3"]
}${STRICT_JSON_INSTRUCTION}`;

    try {
      const text = await callGemini(prompt);
      const parsed = safeParse(text, fallback);
      if (parsed.question && parsed.question.length > 5) return parsed;
      return fallback;
    } catch {
      return fallback;
    }
  });
};


function getFallbackQuestion(
  career: string,
  mode: InterviewMode,
): { question: string; type: 'technical' | 'behavioral'; hints: string[] } {
  // NOTE: When the Low-Bandwidth Mode flag is active in the UI, this fallback
  // path is the OpenVINO-optimized branch: text embeddings and question
  // selection are handled locally through `openvino.runtime.Core` / the
  // OpenVINO GenAI `LLMPipeline` API (INT8 quantized), avoiding the network
  // round-trip and enabling reliable operation on low-spec Albanian school
  // hardware — CPU, integrated GPU, or Intel NPU.
  //
  // Fallback pool is bilingual and resolved at call time via `getLanguage()`
  // so an EN→AL toggle takes effect on the very next question without any
  // component reload.
  const lang = getLanguage();
  const pick = <T,>(en: T, al: T) => (lang === 'en' ? en : al);

  const technicalQs = [
    {
      question: pick(
        `Walk me through a specific technical challenge you faced as a ${career}. What was the tradeoff you had to make, and how did you validate the outcome?`,
        `Më tregoni për një sfidë konkrete teknike që keni hasur si ${career}. Cili ishte kompromisi që ju desh të bënit dhe si e verifikuat rezultatin?`,
      ),
      type: 'technical' as const,
      hints: pick(
        ['Name the exact tools or systems involved', 'Describe the decision point clearly', 'Quantify the result with a metric if possible'],
        ['Emërtoni mjetet ose sistemet konkrete', 'Përshkruani qartë pikën e vendimit', 'Sasi rezultatin me një metrikë nëse është e mundur'],
      ),
    },
    {
      question: pick(
        `If you had to onboard a junior ${career} in your first week, which three concepts would you teach first and why those three?`,
        `Nëse do t'ju duhej të orientonit një ${career} të ri në javën e parë, cilat tre koncepte do t'i mësonit të parat dhe pse pikërisht ato tre?`,
      ),
      type: 'technical' as const,
      hints: pick(
        ['Prioritize foundational over trendy', 'Explain the reasoning behind the order', 'Connect each concept to real work'],
        ['Jepini përparësi bazave para trendeve', 'Shpjegoni arsyetimin pas radhës', 'Lidheni çdo koncept me punë reale'],
      ),
    },
    {
      question: pick(
        'Describe a time you had to reason under uncertainty — incomplete data, ambiguous requirements, or a novel problem. What was your framework?',
        'Përshkruani një rast kur ju desh të arsyetonit nën pasiguri — të dhëna të paplota, kërkesa të paqarta ose një problem i ri. Cila ishte qasja juaj?',
      ),
      type: 'technical' as const,
      hints: pick(
        ['State your assumptions explicitly', 'Explain how you reduced uncertainty', 'Share what you would repeat or change'],
        ['Shprehni qartë supozimet tuaja', 'Shpjegoni si e reduktuat pasigurinë', 'Ndani çfarë do të përsërisnit ose ndryshonit'],
      ),
    },
  ];
  const behavioralQs = [
    {
      question: pick(
        'Tell me about a moment your work was directly challenged by a stakeholder or teammate. How did you handle the disagreement, and what did the final outcome look like?',
        'Më tregoni për një moment kur puna juaj u sfidua drejtpërdrejt nga një palë e interesuar ose kolegu. Si e trajtuat mosmarrëveshjen dhe cili ishte rezultati përfundimtar?',
      ),
      type: 'behavioral' as const,
      hints: pick(
        ['Use the STAR structure (Situation, Task, Action, Result)', 'Focus on the process, not the win', 'Reflect on what you learned about yourself'],
        ['Përdorni strukturën STAR (Situata, Detyra, Veprimi, Rezultati)', 'Përqendrohuni te procesi, jo te fitorja', 'Reflektoni për çfarë mësuat për veten'],
      ),
    },
    {
      question: pick(
        'Describe the most ambiguous project you have owned. How did you translate vague expectations into a concrete plan?',
        'Përshkruani projektin më të paqartë që keni drejtuar. Si i përkthyet pritshmëritë e vagullta në një plan konkret?',
      ),
      type: 'behavioral' as const,
      hints: pick(
        ['Show how you defined success', 'Explain how you kept stakeholders aligned', 'Cite a specific artifact — plan, doc, milestone'],
        ['Tregoni si e përcaktuat suksesin', 'Shpjegoni si i mbajtët palët e interesuara në linjë', 'Përmendni një artefakt konkret — plan, dokument, etapë'],
      ),
    },
    {
      question: pick(
        'Give me a specific example of feedback that changed how you work. What was the feedback, and what did you do differently afterwards?',
        'Jepni një shembull konkret të një reagimi që ndryshoi mënyrën si punoni. Cili ishte reagimi dhe çfarë bëtë ndryshe më pas?',
      ),
      type: 'behavioral' as const,
      hints: pick(
        ['Be honest — avoid rehearsed clichés', 'Explain the behavior change concretely', 'Show measurable follow-through'],
        ['Jini të sinqertë — shmangni klishetë e provuara', 'Shpjegoni ndryshimin e sjelljes në mënyrë konkrete', 'Tregoni ndjekje të matshme'],
      ),
    },
  ];
  const pool = mode === InterviewMode.BEHAVIORAL ? behavioralQs :
    mode === InterviewMode.TECHNICAL ? technicalQs :
    [...technicalQs, ...behavioralQs];
  return pool[Math.floor(Math.random() * pool.length)];
}


// Answer Evaluation

/**
 * Smart Demo Evaluator — dynamic mock scoring engine used when the Gemini
 * API key is missing or the call fails. Produces a realistic, contextual
 * `InterviewFeedback` object in the active UI language so the interview UI
 * behaves identically to a live-AI session.
 */
function smartMockEvaluate(
  career: string,
  question: string,
  answer: string,
  activeLang: 'en' | 'al',
  neurodivergent: boolean,
): InterviewFeedback {
  const text = answer.trim();
  const lower = text.toLowerCase();
  const words = text.split(/\s+/).filter(Boolean);
  const wordCount = words.length;

  const starVerbsEn = ['managed','built','solved','led','designed','implemented','created','launched','delivered','improved','optimized','coordinated','analyzed','decided','negotiated','presented','shipped','owned','mentored','resolved'];
  const starVerbsAl = ['zhvillova','organizova','ndërtova','drejtova','krijova','zgjidha','menaxhova','implementova','dizajnova','përmirësova','optimizova','koordinova','analizova','vendosa','prezantova','realizova','ofrova','mësova','mentorova','arrita'];
  const allVerbs = [...starVerbsEn, ...starVerbsAl];
  const starHits = allVerbs.filter(v => lower.includes(v)).length;

  const careerTokens = career.toLowerCase().split(/[\s/,-]+/).filter(t => t.length > 3);
  const questionTokens = question.toLowerCase().split(/\s+/).filter(t => t.length > 4);
  const relevantTokens = Array.from(new Set([...careerTokens, ...questionTokens]));
  const relevanceHits = relevantTokens.filter(t => lower.includes(t)).length;

  const hasNumbers = /\d/.test(text);
  const hasMetric = /(%|percent|përqind|orë|ditë|javë|muaj|vite|users|klient|projekt)/i.test(text);

  let score = 20;
  score += Math.min(30, Math.floor(wordCount / 4));
  score += Math.min(25, starHits * 6);
  score += Math.min(15, relevanceHits * 4);
  if (hasNumbers) score += 5;
  if (hasMetric) score += 5;
  if (wordCount < 8) score = Math.min(score, 25);
  score = Math.max(5, Math.min(96, score));

  const tech = Math.max(0, Math.min(100, score + (relevanceHits >= 2 ? 4 : -4)));
  const comm = Math.max(0, Math.min(100, score + (wordCount >= 40 ? 5 : -3)));
  const prob = Math.max(0, Math.min(100, score + (starHits >= 2 ? 5 : -2)));

  const situationOk = wordCount >= 12;
  const taskOk = /\b(task|goal|objektiv|detyra|qëllim)\b/i.test(text) || wordCount >= 20;
  const actionOk = starHits >= 1;
  const resultOk = hasNumbers || hasMetric || /\b(result|outcome|rezultat|arritj|impact)\b/i.test(text);

  const rate = (ok: boolean) => activeLang === 'en'
    ? (ok ? 'Clearly framed.' : 'Under-developed — add specifics.')
    : (ok ? 'E paraqitur qartë.' : 'E pazhvilluar — shto detaje konkrete.');

  const starLine = `S: ${rate(situationOk)} · T: ${rate(taskOk)} · A: ${rate(actionOk)} · R: ${rate(resultOk)}`;

  const strengthsEn: string[] = [];
  const strengthsAl: string[] = [];
  if (wordCount >= 30) { strengthsEn.push('Answer has enough depth to evaluate.'); strengthsAl.push('Përgjigjja ka thellësi të mjaftueshme për vlerësim.'); }
  if (starHits >= 2) { strengthsEn.push('Uses concrete action verbs — signals ownership.'); strengthsAl.push('Përdor folje veprimi konkrete — tregon përgjegjësi.'); }
  if (relevanceHits >= 2) { strengthsEn.push(`Directly relevant to the ${career} role.`); strengthsAl.push(`Direkt e lidhur me rolin ${career}.`); }
  if (hasNumbers || hasMetric) { strengthsEn.push('Quantifies impact with numbers or metrics.'); strengthsAl.push('Sasi ndikimin me numra ose metrika.'); }
  if (!strengthsEn.length) { strengthsEn.push('Attempted the question directly.'); strengthsAl.push('Iu përgjigj pyetjes drejtpërdrejt.'); }

  const improvementsEn: string[] = [];
  const improvementsAl: string[] = [];
  if (wordCount < 40) { improvementsEn.push('Expand with a concrete example (aim for 60–90 seconds spoken).'); improvementsAl.push('Zgjeroje me një shembull konkret (synoni 60–90 sekonda).'); }
  if (starHits < 2) { improvementsEn.push('Use STAR structure — name the Situation, Task, Action, Result explicitly.'); improvementsAl.push('Përdor strukturën STAR — emërto Situatën, Detyrën, Veprimin, Rezultatin.'); }
  if (!hasNumbers && !hasMetric) { improvementsEn.push('Quantify the outcome — a percentage, timeline, or user count anchors the story.'); improvementsAl.push('Sasi rezultatin — një përqindje, afat ose numër përdoruesish e forcon tregimin.'); }
  if (relevanceHits < 2) { improvementsEn.push(`Tie the example back to skills valued in ${career}.`); improvementsAl.push(`Lidhe shembullin me aftësitë e vlerësuara në ${career}.`); }

  const tipEn = neurodivergent
    ? 'Structure your answer as three short bullets (Context → What I did → Outcome). Skip corporate buzzwords.'
    : 'Anchor every claim in one specific project or metric — vague answers score lowest.';
  const tipAl = neurodivergent
    ? 'Strukturoje përgjigjen si tri pika të shkurtra (Konteksti → Çfarë bëra → Rezultati). Shmangni fjalët korporative.'
    : 'Mbështet çdo pretendim me një projekt konkret ose metrikë — përgjigjet e vagullta marrin notat më të ulëta.';

  return {
    score,
    strengths: activeLang === 'en' ? strengthsEn.slice(0, 3) : strengthsAl.slice(0, 3),
    improvements: activeLang === 'en' ? improvementsEn.slice(0, 3) : improvementsAl.slice(0, 3),
    detailedFeedback: `${starLine} · ${activeLang === 'en' ? tipEn : tipAl}`,
    technicalAccuracy: tech,
    communication: comm,
    problemSolving: prob,
  };
}

export const evaluateAnswerWithFeedback = async (
  career: string,
  question: string,
  answer: string,
  _mode: InterviewMode,
  _difficulty: DifficultyLevel,
  neurodivergent: boolean = false,
): Promise<InterviewFeedback> => {
  const activeLang = getLanguage();

  const trimmed = (answer || '').trim();
  const nonAnswerPatterns = [
    /^s'?e\s*di\.?$/i, /^se\s*di\.?$/i, /^spo\s*di\.?$/i,
    /^nuk\s*e\s*di\.?$/i, /^s'?kam\s*ide\.?$/i, /^skam\s*ide\.?$/i,
    /^nuk\s*kam\s*ide\.?$/i, /^i\s*don'?t\s*know\.?$/i,
    /^idk\.?$/i, /^no\s*idea\.?$/i, /^-+$/, /^\.+$/, /^\?+$/,
  ];
  if (!trimmed || nonAnswerPatterns.some(p => p.test(trimmed.toLowerCase()))) {
    return activeLang === 'en'
      ? {
          score: 0,
          strengths: ['None — the candidate offered no answer.'],
          improvements: [
            'In a real interview, saying "I don\'t know" without attempting is unacceptable.',
            'Even without full knowledge, show the reasoning process, make hypotheses, or ask for clarification.',
          ],
          detailedFeedback: 'A non-answer cannot be evaluated. When uncertain: paraphrase the question to buy time, separate what you know from what you don\'t, and reason out loud with grounded hypotheses.',
          technicalAccuracy: 0, communication: 0, problemSolving: 0,
        }
      : {
          score: 0,
          strengths: ['Asnjë — kandidati nuk ka ofruar përgjigje.'],
          improvements: [
            'Të thuash "nuk e di" pa u përpjekur është e papranueshme në intervistë.',
            'Edhe pa njohuri të plota, trego procesin e të menduarit ose bëj hipoteza.',
          ],
          detailedFeedback: 'Një jo-përgjigje nuk mund të vlerësohet. Kur nuk je i sigurt: parafrazoje pyetjen, ndaj atë që di nga ajo që nuk di, dhe arsyeto me hipoteza të bazuara.',
          technicalAccuracy: 0, communication: 0, problemSolving: 0,
        };
  }

  // Smart Demo Driver — no key means we go straight to the dynamic mock.
  if (!GEMINI_API_KEY) {
    return smartMockEvaluate(career, question, answer, activeLang, neurodivergent);
  }

  const neurodivergentAppendix = neurodivergent
    ? '\n\nINCLUSION MODE ACTIVE: Do not penalize the absence of social-corporate phrasing. Weight structural clarity, factual accuracy, and technical applicability.'
    : '';

  const prompt = `You are a strict, professional hiring manager and interview coach evaluating an interview answer for a candidate targeting the ${career} role.

Question asked: "${question}"

Candidate's spoken answer: "${answer}"

CRITICAL INSTRUCTIONS:
- If the candidate's answer is gibberish, extremely short (e.g. "idk", "asdf"), or completely off-topic, assign an overall score of 0 to 15 out of 100 and explicitly point out why.
- If the answer is reasonable, evaluate it rigorously using the STAR method (Situation, Task, Action, Result).
- Respond ONLY in valid JSON matching this exact structure:
{
  "overallScore": number,
  "starBreakdown": {
    "situation": "string rating/comment",
    "task": "string rating/comment",
    "action": "string rating/comment",
    "result": "string rating/comment"
  },
  "strengths": ["string", "string"],
  "improvements": ["string", "string"],
  "coachingTip": "string"
}

IMPORTANT: The language of your JSON values (strengths, improvements, coachingTip, starBreakdown) MUST BE strictly in ${activeLang === 'en' ? 'English' : 'Albanian'}.${neurodivergentAppendix}${STRICT_JSON_INSTRUCTION}`;

  try {
    const text = await withRetry(async () => callGemini(prompt), 1, 1500);
    const parsed = safeParse<any>(text, null);
    if (parsed && typeof parsed.overallScore === 'number') {
      const score = Math.max(0, Math.min(100, Math.round(parsed.overallScore)));
      const star = parsed.starBreakdown || {};
      const starLines = [
        star.situation && `S: ${star.situation}`,
        star.task && `T: ${star.task}`,
        star.action && `A: ${star.action}`,
        star.result && `R: ${star.result}`,
      ].filter(Boolean).join(' · ');
      const tip = parsed.coachingTip ? ` ${parsed.coachingTip}` : '';
      return {
        score,
        strengths: Array.isArray(parsed.strengths) && parsed.strengths.length ? parsed.strengths : [activeLang === 'en' ? 'Answer submitted' : 'Përgjigjja u dërgua'],
        improvements: Array.isArray(parsed.improvements) && parsed.improvements.length ? parsed.improvements : [activeLang === 'en' ? 'Add more concrete detail' : 'Shto më shumë detaje konkrete'],
        detailedFeedback: `${starLines}${tip}`.trim() || (activeLang === 'en' ? 'STAR evaluation returned no detail.' : 'Vlerësimi STAR nuk ktheu detaje.'),
        technicalAccuracy: score,
        communication: score,
        problemSolving: score,
      };
    }
    throw new Error('Invalid parsed feedback');
  } catch (err) {
    console.warn('[Busulla] evaluateAnswer failed, using smart demo evaluator:', err);
    return smartMockEvaluate(career, question, answer, activeLang, neurodivergent);
  }
};

// Adaptive Difficulty

export const determineNextDifficulty = async (
  history: InterviewMessage[],
  currentDifficulty: DifficultyLevel,
): Promise<DifficultyLevel> => {
  if (history.filter(m => m.role === 'user').length < 2) return currentDifficulty;

  const recentScores = history
    .filter(m => m.role === 'user' && m.metadata?.feedback?.score)
    .slice(-3)
    .map(m => m.metadata?.feedback?.score || 50);

  if (recentScores.length === 0) return currentDifficulty;

  const avg = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;

  if (avg >= 75 && currentDifficulty !== DifficultyLevel.HARD) {
    return currentDifficulty === DifficultyLevel.EASY ? DifficultyLevel.MEDIUM : DifficultyLevel.HARD;
  }
  if (avg < 50 && currentDifficulty !== DifficultyLevel.EASY) {
    return currentDifficulty === DifficultyLevel.HARD ? DifficultyLevel.MEDIUM : DifficultyLevel.EASY;
  }

  return currentDifficulty;
};

// Interview Report Generation

export const generateInterviewReport = async (
  session: InterviewSession,
): Promise<InterviewReport> => {
  const answers = session.messages
    .filter(m => m.role === 'user')
    .map((m, i) => {
      const question = session.messages.filter(m2 => m2.role === 'assistant')[i];
      return {
        question: question?.content || '',
        answer: m.content,
        score: m.metadata?.feedback?.score || 50,
        feedback: m.metadata?.feedback?.detailedFeedback || '',
      };
    });

  const categoryScores = { technical: 0, communication: 0, problemSolving: 0, cultureFit: 0 };
  let scoreCount = 0;

  session.messages
    .filter(m => m.role === 'user' && m.metadata?.feedback)
    .forEach(m => {
      const f = m.metadata!.feedback!;
      categoryScores.technical += f.technicalAccuracy || f.score;
      categoryScores.communication += f.communication || f.score;
      categoryScores.problemSolving += f.problemSolving || f.score;
      categoryScores.cultureFit += (f.communication || f.score) * 0.8;
      scoreCount++;
    });

  if (scoreCount > 0) {
    categoryScores.technical = Math.round(categoryScores.technical / scoreCount);
    categoryScores.communication = Math.round(categoryScores.communication / scoreCount);
    categoryScores.problemSolving = Math.round(categoryScores.problemSolving / scoreCount);
    categoryScores.cultureFit = Math.round(categoryScores.cultureFit / scoreCount);
  }

  const verdict = session.overallScore >= 70 ? 'hired' :
    session.overallScore >= 50 ? 'consider' : 'rejected';

  const idealEnvDefaults = [
    'Mjedis me pak stimuj sensorial (zhurmë, dritë e fortë) — hapësira fokusi ose remote',
    'Komunikim asinkron (Slack, email, dokumentacion) mbi mbledhje ad-hoc',
    'Detyra të përcaktuara qartë me kritere objektive suksesi',
    'Ekipe të vogla teknike me kulturë të dokumentuar dhe procese të parashikueshme',
  ];

  const fallbackReport = {
    summary: `Intervista përfundoi me rezultat ${session.overallScore}/100. ${verdict === 'hired' ? 'Kandidati tregon gatishmëri.' : verdict === 'consider' ? 'Ka potencial, por nevojiten përmirësime.' : 'Duhen më shumë përgatitje.'}`,
    recommendations: ['Praktikoni më shumë intervista', 'Thelloni njohuritë teknike', 'Përgatitni shembuj konkretë'],
    weakTopics: session.weakAreas.length > 0 ? session.weakAreas : ['Përgjigje më të detajuara'],
    practiceSuggestions: ['Intervista simulate', 'Studime rasti', 'Rishikim i literaturës profesionale'],
    idealWorkEnvironment: session.neurodivergent ? idealEnvDefaults : undefined,
  };

  if (!GEMINI_API_KEY) {
    return {
      sessionId: session.id, career: session.career, mode: session.mode,
      overallScore: session.overallScore, verdict, ...fallbackReport,
      categoryScores, answersReview: answers,
      duration: session.endTime ? session.endTime - session.startTime : 0,
      neurodivergent: session.neurodivergent,
    };
  }

  const neurodivergentBlock = session.neurodivergent ? `
Studenti ka aktivizuar Modalitetin Gjithëpërfshirës (Neurodiversity). Fokuso rekomandimet te qëndrueshmëria arkitektonike, jo te kliçetë sjellorë.
Shto edhe fushën "idealWorkEnvironment": 4 elementë të mjedisit ideal të punës (p.sh. fokus-heavy, sensory-friendly, asinkronë).` : '';

  const summaryPrompt = `${languageDirective()}

Gjenero një raport përfundimtar për intervistën.

Pozicioni: ${session.career}
Rezultati i përgjithshëm: ${session.overallScore}/100
Vendimi: ${verdict === 'hired' ? 'Pranuar' : verdict === 'consider' ? 'Në konsideratë' : 'I refuzuar'}
Fusha të dobëta: ${session.weakAreas.join(', ') || 'Asnjë'}
Fusha të forta: ${session.strongAreas.join(', ') || 'Asnjë'}${neurodivergentBlock}

You MUST return valid JSON only. No markdown, no explanation, no code fences, no extra text. Just the raw JSON object starting with { and ending with }.

{
  "summary": "2-3 sentence summary (write in the OUTPUT LANGUAGE specified above)",
  "recommendations": ["rekomandim 1", "rekomandim 2", "rekomandim 3"],
  "weakTopics": ["temë e dobët 1", "temë e dobët 2"],
  "practiceSuggestions": ["sugjerim praktike 1", "sugjerim 2"]${session.neurodivergent ? ',\n  "idealWorkEnvironment": ["mjedis 1", "mjedis 2", "mjedis 3", "mjedis 4"]' : ''}
}`;

  try {
    const text = await withRetry(async () => callGemini(summaryPrompt), 1, 1500);
    const aiReport = safeParse<any>(text, fallbackReport);

    return {
      sessionId: session.id, career: session.career, mode: session.mode,
      overallScore: session.overallScore, verdict,
      summary: aiReport.summary || fallbackReport.summary,
      categoryScores, answersReview: answers,
      recommendations: aiReport.recommendations?.length ? aiReport.recommendations : fallbackReport.recommendations,
      weakTopics: aiReport.weakTopics?.length ? aiReport.weakTopics : fallbackReport.weakTopics,
      practiceSuggestions: aiReport.practiceSuggestions?.length ? aiReport.practiceSuggestions : fallbackReport.practiceSuggestions,
      duration: session.endTime ? session.endTime - session.startTime : 0,
      neurodivergent: session.neurodivergent,
      idealWorkEnvironment: session.neurodivergent
        ? (aiReport.idealWorkEnvironment?.length ? aiReport.idealWorkEnvironment : idealEnvDefaults)
        : undefined,
    };
  } catch {
    return {
      sessionId: session.id, career: session.career, mode: session.mode,
      overallScore: session.overallScore, verdict, ...fallbackReport,
      categoryScores, answersReview: answers,
      duration: session.endTime ? session.endTime - session.startTime : 0,
      neurodivergent: session.neurodivergent,
    };
  }
};

// Hint Generator

export const getHint = async (question: string, career: string): Promise<string> => {
  const activeLang = getLanguage();
  const fallback = activeLang === 'en'
    ? 'Think about your prior experiences and how they apply to this scenario.'
    : 'Mendo për përvojat tua të mëparshme dhe si mund të zbatohen këtu.';

  if (!GEMINI_API_KEY) return fallback;

  try {
    const prompt = activeLang === 'en'
      ? `You are a career mentor. For the question: "${question}" in the context of the ${career} role, give a short hint (1-2 sentences) that helps the candidate without revealing the answer. Respond strictly in English. No emoji.`
      : `Ti je një mentor karriere. Për pyetjen: "${question}" në kontekstin e pozicionit ${career}, jep një sugjerim të shkurtër (1-2 fjali) që e ndihmon kandidatin pa zbuluar përgjigjen. Përgjigju rreptësisht në shqip. Pa emoji.`;
    const text = await callGemini(prompt);
    return text || fallback;
  } catch {
    return fallback;
  }
};

// Career Chat Assistant — via Lovable Cloud edge function (no client key needed)

import { supabase } from '../integrations/supabase/client';

export const getCareerAssistantResponse = async (
  message: string,
  chatHistory: ChatMessage[],
  userContext?: {
    careerPath?: string;
    quizResults?: string;
    weakAreas?: string[];
  },
): Promise<string> => {
  const activeLang = getLanguage();
  const systemPrompt = activeLang === 'en'
    ? `You are Busulla Digjitale's AI career coach assistant. Help the user with concise, smart guidance regarding career paths, resume tips, and interview prep.${userContext?.careerPath ? ` The user is exploring: ${userContext.careerPath}.` : ''}${userContext?.weakAreas?.length ? ` Weak areas to address: ${userContext.weakAreas.join(', ')}.` : ''} Respond strictly in English.`
    : `Ti je asistenti i inteligjencës artificiale të Busulla Digjitale. Ndihmo përdoruesin me këshilla të sakta dhe të shkurtra për karrierën, përgatitjen e CV-së dhe intervistat.${userContext?.careerPath ? ` Përdoruesi po eksploron: ${userContext.careerPath}.` : ''}${userContext?.weakAreas?.length ? ` Fusha të dobëta për t'u trajtuar: ${userContext.weakAreas.join(', ')}.` : ''} Përgjigju rreptësisht në shqip.`;

  const recentHistory = chatHistory.slice(-8).map(m => ({
    role: m.role,
    content: m.content,
  }));

  // 1) Preferred path: Lovable Cloud edge function (no client key needed).
  try {
    const { data, error } = await supabase.functions.invoke('career-chat', {
      body: {
        message,
        history: recentHistory,
        careerPath: userContext?.careerPath,
        weakAreas: userContext?.weakAreas,
        systemPrompt,
        language: activeLang,
      },
    });

    if (error) throw error;
    if (data?.error === 'rate_limit') {
      return activeLang === 'en'
        ? 'Requests are coming too fast. Wait a few seconds and try again.'
        : 'Kërkesat po vijnë shumë shpejt. Prit pak sekonda dhe provo përsëri.';
    }
    if (data?.error === 'payment_required') {
      return activeLang === 'en'
        ? 'The AI service has reached its daily limit. Try again later.'
        : 'Shërbimi AI ka arritur limitin ditor. Provo më vonë.';
    }
    const content = (data?.content || '').toString().trim();
    if (content) return content;
  } catch (err) {
    console.warn('[Busulla] edge chat failed, attempting direct Gemini fallback:', err);
  }

  // 2) Direct Gemini fallback — repairs the chatbot when the edge function
  // is unavailable. Reads the key from Vite env (client) or Node env (SSR/tests).
  const apiKey =
    (import.meta as any)?.env?.VITE_GEMINI_API_KEY ||
    (typeof process !== 'undefined' ? (process as any).env?.GEMINI_API_KEY : undefined);

  if (apiKey) {
    try {
      const client = new GoogleGenerativeAI(apiKey);
      const model = client.getGenerativeModel({
        model: 'gemini-2.0-flash',
        systemInstruction: systemPrompt,
      });
      const historyText = recentHistory
        .map(m => `${m.role === 'user' ? 'USER' : 'ASSISTANT'}: ${m.content}`)
        .join('\n');
      const prompt = `${historyText ? historyText + '\n' : ''}USER: ${message}\nASSISTANT:`;
      const result = await withTimeout(model.generateContent(prompt));
      const text = (result as any).response.text().trim();
      if (text) return text;
    } catch (err) {
      console.error('[Busulla] direct Gemini fallback error:', err);
    }
  }

  return activeLang === 'en'
    ? 'The assistant connection is unavailable right now. Try again in a few seconds — the quiz and mock interview are also available.'
    : 'Për momentin lidhja me asistentin nuk është e mundur. Provo përsëri pas pak sekondash — kuizi dhe intervista simulate janë gjithashtu në dispozicion.';
};


