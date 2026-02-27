import { Router } from 'express';
import { google } from '@ai-sdk/google';
import { streamText } from 'ai';
import { createClient } from '@supabase/supabase-js';
import { QURANIC_DATA, QURAN_SYSTEM_PROMPT } from '../data/quranic.js';
import { normalizeArabic, countOccurrences, summarizeContexts, suggestSimilarWords } from '../lib/lexicon.js';
import { buildWordTemplate, buildGeneralTemplate } from '../lib/responseTemplate.js';
import { SultanAnalysisEngine } from '../engine/SultanAnalysisEngine.js';

const router = Router();

// Initialize Supabase
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const isSupabaseEnabled = supabaseUrl && supabaseKey;
const supabase = isSupabaseEnabled
  ? createClient(supabaseUrl, supabaseKey)
  : null;

interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at?: string;
}

// Save conversation to Supabase
async function saveConversation(messages: ChatMessage[], conversationId?: string) {
  if (!isSupabaseEnabled || !supabase) return conversationId;

  try {
    if (!conversationId) {
      const firstUserMessage = messages.find((m) => m.role === 'user')?.content || 'محادثة جديدة';
      const title = firstUserMessage.slice(0, 50) + (firstUserMessage.length > 50 ? '...' : '');

      const { data: conversation, error: convError } = await supabase
        .from('conversations')
        .insert({ title })
        .select()
        .single();

      if (convError) throw convError;
      conversationId = conversation.id;
    } else {
      await supabase
        .from('conversations')
        .update({ updated_at: new Date().toISOString() })
        .eq('id', conversationId);
    }

    const lastMessage = messages[messages.length - 1];
    if (lastMessage) {
      await supabase
        .from('messages')
        .insert({
          conversation_id: conversationId,
          role: lastMessage.role,
          content: lastMessage.content,
        });
    }

    return conversationId;
  } catch (error) {
    console.error('Error saving conversation:', error);
    return conversationId;
  }
}

// GET: Fetch Quranic data
router.get('/quranic/data', (req, res) => {
  try {
    res.json({
      data: QURANIC_DATA,
      systemPrompt: QURAN_SYSTEM_PROMPT,
      settings: {
        temperature: 0.2,
        precision: 'high',
        persona: 'SULTAN',
        source: 'Quran-only'
      }
    });
  } catch (error) {
    console.error('Error fetching Quranic data:', error);
    res.status(500).json({ error: 'Failed to fetch Quranic data' });
  }
});

// POST: Semantic search — grounded Quranic verses (SultanAnalysisEngine)
router.post('/semantic-search', async (req, res) => {
  try {
    const { query, topK = 10 } = req.body as { query?: string; topK?: number };
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'query required' });
    }
    const results = await SultanAnalysisEngine.semanticSearch(query, Math.min(topK || 10, 20));
    res.json({
      verses: results.map((r) => ({ verse: r.verse, score: r.score })),
      count: results.length,
    });
  } catch (error) {
    console.error('Error in semantic search:', error);
    res.status(500).json({ error: 'Semantic search failed' });
  }
});

// GET: Fetch conversations or messages
router.get('/', async (req, res) => {
  try {
    if (!isSupabaseEnabled || !supabase) {
      return res.json({ conversations: [], messages: [] });
    }

    const conversationId = req.query.conversationId as string;

    if (conversationId) {
      const { data: messages, error } = await supabase
        .from('messages')
        .select('*')
        .eq('conversation_id', conversationId)
        .order('created_at', { ascending: true });

      if (error) throw error;
      return res.json({ messages });
    }

    const { data: conversations, error } = await supabase
      .from('conversations')
      .select('*')
      .order('updated_at', { ascending: false })
      .limit(50);

    if (error) throw error;
    res.json({ conversations });
  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).json({ error: 'Failed to fetch data' });
  }
});

// POST: Stream chat response
router.post('/', async (req, res) => {
  try {
    const { messages: rawMessages, conversationId } = req.body as { messages: ChatMessage[]; conversationId?: string };
    const messages: ChatMessage[] = Array.isArray(rawMessages) ? rawMessages : [];

    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      return res.status(500).json({ error: 'Google API key not configured' });
    }

    const model = google('gemini-2.0-flash-exp');

    // Save conversation
    const newConversationId = await saveConversation(messages, conversationId);

    // Set headers for streaming
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Transfer-Encoding', 'chunked');
    if (newConversationId) {
      res.setHeader('x-conversation-id', newConversationId);
    }

    // Prepare system message with Sultan persona & Quranic constraints
    const systemMessage = {
      role: 'system' as const,
      content: QURAN_SYSTEM_PROMPT
    };

    // Combine with Quranic data context
    const contextMessage = {
      role: 'user' as const,
      content: `المرجع الوحيد: نصوص القرآن؛ اللسان العربي هو معيار الدلالة؛ بيانات السياق: ${JSON.stringify(QURANIC_DATA.metadata)}`
    };

    const enhancedMessages: ChatMessage[] = [systemMessage];
    if (!messages.some((m: ChatMessage) => m.role === 'system')) {
      enhancedMessages.push(contextMessage);
    }

    // Detect word query and inject lexicon analysis context
    const lastUser = (messages || []).filter((m: ChatMessage) => m.role === 'user').slice(-1)[0];
    const isArabicChar = (ch: string) => /[\u0600-\u06FF]/.test(ch);
    const isWordQuery = lastUser && lastUser.content && lastUser.content.length <= 24 && [...lastUser.content].every(c => isArabicChar(c) || c === ' ');

    if (isWordQuery && lastUser) {
      const term = normalizeArabic(lastUser.content.trim());
      const occ = countOccurrences(term, QURANIC_DATA);
      const summary = summarizeContexts(term, occ);
      const similar = suggestSimilarWords(term);
      const template = buildWordTemplate(term, occ, summary.examples, similar);
      enhancedMessages.push({ role: 'user', content: template });
    }

    // For general contexts, provide a structured instruction template
    if (!isWordQuery && lastUser && lastUser.content) {
      const isFirstMessage = messages.length <= 1; // Only first turn
      const genTemplate = buildGeneralTemplate(lastUser.content, isFirstMessage);
      enhancedMessages.push({ role: 'user', content: genTemplate });

      // RAG: Inject semantic search context from SultanAnalysisEngine (when Ollama available)
      try {
        const contextVerses = await SultanAnalysisEngine.getContextForQuery(lastUser.content, 5);
        if (contextVerses.length > 0) {
          const contextText = contextVerses
            .map((v) => `[${v.surahName} ${v.ayahNumber}] ${v.text}`)
            .join('\n');
          enhancedMessages.push({
            role: 'user',
            content: `آيات ذات صلة دلالية بالاستفسار:\n${contextText}`,
          });
        }
      } catch {
        // Ollama may be unavailable — continue without RAG
      }
    }

    enhancedMessages.push(...messages);

    const result = streamText({
      model,
      messages: enhancedMessages,
      temperature: 0.2, // دقة عالية - Precision mode
      maxTokens: 2048,
    });

    let fullResponse = '';

    for await (const chunk of result.textStream) {
      fullResponse += chunk;
      res.write(chunk);
    }

    // Save assistant response
    if (newConversationId) {
      await saveConversation(
        [...messages, { role: 'assistant', content: fullResponse }],
        newConversationId
      );
    }

    res.end();
  } catch (error) {
    console.error('Error in chat route:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// DELETE: Delete conversation
router.delete('/', async (req, res) => {
  try {
    if (!isSupabaseEnabled || !supabase) {
      return res.status(400).json({ error: 'Supabase not enabled' });
    }

    const conversationId = req.query.conversationId as string;
    if (!conversationId) {
      return res.status(400).json({ error: 'conversationId required' });
    }

    const { error } = await supabase
      .from('conversations')
      .delete()
      .eq('id', conversationId);

    if (error) throw error;
    res.json({ success: true });
  } catch (error) {
    console.error('Error deleting conversation:', error);
    res.status(500).json({ error: 'Failed to delete conversation' });
  }
});

export default router;
