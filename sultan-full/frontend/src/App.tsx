import { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import ConversationsSidebar from './components/ConversationsSidebar';

function App() {
  const [conversationId, setConversationId] = useState<string | null>(null);

  const handleNewConversation = () => {
    setConversationId(null);
  };

  const handleSelectConversation = (id: string) => {
    setConversationId(id);
  };

  const handleConversationCreated = (id: string) => {
    setConversationId(id);
  };

  return (
    <div className="flex h-screen">
      <ChatInterface
        conversationId={conversationId}
        onConversationCreated={handleConversationCreated}
      />
      <ConversationsSidebar
        currentConversationId={conversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />
    </div>
  );
}

export default App;
