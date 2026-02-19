import React, { useState, useEffect } from 'react';
import { supabase, isSupabaseEnabled } from '../lib/supabase';
import type { Conversation } from '../types';

interface ConversationsSidebarProps {
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
}

const ConversationsSidebar: React.FC<ConversationsSidebarProps> = ({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
}) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    if (!isSupabaseEnabled || !supabase) {
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('/api/chat');
      const data = await response.json();
      setConversations(data.conversations || []);
    } catch (error) {
      console.error('Error loading conversations:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteConversation = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('هل أنت متأكد من حذف هذه المحادثة؟')) return;

    try {
      await fetch(`/api/chat?conversationId=${id}`, { method: 'DELETE' });
      setConversations(conversations.filter((c) => c.id !== id));
      if (currentConversationId === id) {
        onNewConversation();
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };

  if (!isSupabaseEnabled) {
    return (
      <div className="w-64 bg-gray-900 border-l border-gray-800 p-4">
        <p className="text-gray-400 text-sm">Supabase غير مفعّل</p>
      </div>
    );
  }

  return (
    <div className="w-64 bg-gray-900 border-l border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <button
          onClick={onNewConversation}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
        >
          + محادثة جديدة
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {loading ? (
          <p className="text-gray-400 text-sm text-center p-4">جاري التحميل...</p>
        ) : conversations.length === 0 ? (
          <p className="text-gray-400 text-sm text-center p-4">لا توجد محادثات</p>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={`p-3 mb-2 rounded-lg cursor-pointer transition-colors group ${
                currentConversationId === conv.id
                  ? 'bg-gray-800'
                  : 'hover:bg-gray-800'
              }`}
            >
              <div className="flex justify-between items-start">
                <p className="text-sm text-white truncate flex-1">{conv.title}</p>
                <button
                  onClick={(e) => deleteConversation(conv.id, e)}
                  className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-400 text-xs ml-2"
                >
                  ✕
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {new Date(conv.updated_at).toLocaleDateString('ar')}
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ConversationsSidebar;
