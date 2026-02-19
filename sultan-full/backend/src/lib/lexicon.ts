// Lexicon utilities for Arabic normalization and Quranic occurrences

export interface OccurrenceLocation {
  surah: number;
  surahName: string;
  ayahIndex: number; // 0-based within our content arrays
  verse: string;
}

export interface OccurrenceSummary {
  total: number;
  locations: OccurrenceLocation[];
}

// Remove diacritics and normalize common characters
export function normalizeArabic(input: string): string {
  const diacritics = /[\u064B-\u0652\u0670]/g; // tanwin & harakat
  let s = input.replace(diacritics, '');
  // Normalize alif variants
  s = s.replace(/[\u0622\u0623\u0625]/g, 'ا');
  // Normalize ya/alef maqsura
  s = s.replace(/[\u0649]/g, 'ي');
  // Normalize ta marbuta to ha-like for matching (optional)
  s = s.replace(/[\u0629]/g, 'ه');
  // Trim and collapse spaces
  s = s.replace(/\s+/g, ' ').trim();
  return s;
}

export function countOccurrences(term: string, quranicData: any): OccurrenceSummary {
  const t = normalizeArabic(term);
  let total = 0;
  const locations: OccurrenceLocation[] = [];
  const surahs = Array.isArray(quranicData?.surahs) ? quranicData.surahs : [];
  for (const s of surahs) {
    const verses: string[] = Array.isArray(s?.content) ? s.content : [];
    for (let i = 0; i < verses.length; i++) {
      const vNorm = normalizeArabic(verses[i] || '');
      if (vNorm.includes(t)) {
        total++;
        locations.push({ surah: s.number, surahName: s.name, ayahIndex: i, verse: verses[i] });
      }
    }
  }
  return { total, locations };
}

export function summarizeContexts(term: string, occ: OccurrenceSummary): { examples: string[] } {
  const examples: string[] = [];
  for (const loc of occ.locations.slice(0, 3)) {
    examples.push(`${loc.surahName}:${loc.ayahIndex + 1} — ${loc.verse.slice(0, 60)}${loc.verse.length > 60 ? '…' : ''}`);
  }
  return { examples };
}

// Very simple heuristic: predefined pairs; fallback empty
const CONFUSABLE_MAP: Record<string, string[]> = {
  'رحمه': ['رفق', 'حنان'],
  'رحمة': ['رفق', 'حنان'],
  'غفران': ['صفح', 'ستر'],
  'قلب': ['فؤاد', 'صدر'],
};

export function suggestSimilarWords(term: string): string[] {
  const t = normalizeArabic(term);
  // try direct and without spaces
  const direct = CONFUSABLE_MAP[t];
  if (direct) return direct;
  return [];
}
