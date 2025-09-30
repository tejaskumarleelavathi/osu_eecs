// ChatApp.jsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './ChatApp.css';
import ReactMarkdown from 'react-markdown';

function ChatApp() {
  const [input, setInput] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const [hasInteracted, setHasInteracted] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatLog]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setHasInteracted(true);
    const newQuestion = { type: 'question', text: input };
    setChatLog((prev) => [...prev, newQuestion]);
    setInput('');

    const response = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: newQuestion.text }),
    });

    const data = await response.json();
    const newAnswer = { type: 'answer', text: data.answer };
    setChatLog((prev) => [...prev, newAnswer]);
  };

  return (
    <div className="chat-wrapper">
      <div className="chat-header">Oregon State University</div>
      <div className="chat-log">
        <AnimatePresence>
          {chatLog.map((msg, idx) => (
            <motion.div
              key={idx}
              className={`chat-bubble ${msg.type}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={chatEndRef} />
      </div>
      <motion.div
            className="chat-input-floating-wrapper"
            initial={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', position: 'fixed' }}
            animate={{
                top: hasInteracted ? 'auto' : '50%',
                bottom: hasInteracted ? 0 : 'auto',
                left: hasInteracted ? 0 : '50%',
                transform: hasInteracted ? 'translateY(0)' : 'translate(-50%, -50%)',
                position: hasInteracted ? 'fixed' : 'fixed'
            }}
            transition={{ type: 'spring', stiffness: 60 }}
            >


        <form className="chat-input-floating" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything..."
          />
          <button type="submit">Ask</button>
        </form>
      </motion.div>
    </div>
  );
}

export default ChatApp;