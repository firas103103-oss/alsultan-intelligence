/**
 * SultanAnalysisEngine — Semantic search over Quranic verses
 * Uses Ollama nomic-embed-text for embeddings, in-memory cosine similarity
 * Grounded, non-hallucinated results
 */
import { QURANIC_DATA } from "../data/quranic.js";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://nexus_ollama:11434";
const EMBED_MODEL = "nomic-embed-text";

export interface Verse {
  ayahId: string;
  surahNumber: number;
  surahName: string;
  ayahNumber: number;
  text: string;
}

export interface SearchResult {
  verse: Verse;
  score: number;
}

const verses: Verse[] = [];
for (const surah of QURANIC_DATA.surahs) {
  const content = (surah as { content?: string[] }).content ?? [];
  content.forEach((text, idx) => {
    verses.push({
      ayahId: `${surah.number}:${idx + 1}`,
      surahNumber: surah.number,
      surahName: surah.name,
      ayahNumber: idx + 1,
      text,
    });
  });
}

const embeddingCache = new Map<string, number[]>();

async function embed(text: string): Promise<number[]> {
  const cacheKey = text.slice(0, 200);
  const cached = embeddingCache.get(cacheKey);
  if (cached) return cached;

  const res = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama embed failed: ${res.status} ${err}`);
  }
  const data = (await res.json()) as { embeddings?: number[][] };
  const vec = data.embeddings?.[0];
  if (!vec) throw new Error("No embedding returned");
  embeddingCache.set(cacheKey, vec);
  return vec;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

const MAX_VERSES_TO_EMBED = 80; // Limit for MVP to avoid timeout

/**
 * Semantic search — returns top-k verses by similarity
 * Fallback to keyword match if Ollama unavailable
 */
export async function semanticSearch(
  query: string,
  topK = 10
): Promise<SearchResult[]> {
  const q = query.trim();
  if (!q) return [];

  const searchPool = verses.slice(0, MAX_VERSES_TO_EMBED);

  try {
    const queryEmbedding = await embed(q);
    const scored = await Promise.all(
      searchPool.map(async (v) => {
        const vec = await embed(v.text);
        const score = cosineSimilarity(queryEmbedding, vec);
        return { verse: v, score };
      })
    );
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  } catch {
    // Fallback: keyword search
    const lower = q.toLowerCase();
    const keywordResults = verses
      .filter((v) => v.text.includes(q) || v.text.toLowerCase().includes(lower))
      .slice(0, topK)
      .map((v) => ({ verse: v, score: 0.9 }));
    return keywordResults;
  }
}

/**
 * Get context verses for RAG — used before LLM response
 */
export async function getContextForQuery(query: string, maxVerses = 5): Promise<Verse[]> {
  const results = await semanticSearch(query, maxVerses);
  return results.map((r) => r.verse);
}

export const SultanAnalysisEngine = {
  semanticSearch,
  getContextForQuery,
  verses,
};

export default SultanAnalysisEngine;
