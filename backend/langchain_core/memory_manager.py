"""
Memory Manager
Manages conversation memory and inter-agent communication
Stores context, agent outputs, and conversation history
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from backend.langchain_core.schema import AgentMessage, AgentType, MessageType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Stores conversation history and context"""
    
    def __init__(self, max_messages: int = 100):
        self.messages: deque = deque(maxlen=max_messages)
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
    
    def add_message(self, message: AgentMessage):
        """Add a message to conversation history"""
        self.messages.append(message)
    
    def get_messages(self, agent_type: Optional[AgentType] = None,
                    limit: Optional[int] = None) -> List[AgentMessage]:
        """Get messages, optionally filtered by agent type"""
        messages = list(self.messages)
        
        if agent_type:
            messages = [m for m in messages if m.agent_type == agent_type]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_last_message(self, agent_type: Optional[AgentType] = None) -> Optional[AgentMessage]:
        """Get the last message, optionally by agent type"""
        messages = self.get_messages(agent_type=agent_type)
        return messages[-1] if messages else None
    
    def clear(self):
        """Clear all messages"""
        self.messages.clear()
        logger.info("Conversation memory cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of conversation"""
        return {
            "total_messages": len(self.messages),
            "agents_involved": list(set(m.agent_type for m in self.messages)),
            "created_at": self.created_at.isoformat(),
            "duration_seconds": (datetime.now() - self.created_at).total_seconds()
        }


class AgentWorkingMemory:
    """Working memory for agent-to-agent communication"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.access_log: Dict[str, datetime] = {}
    
    def set(self, key: str, value: Any, agent_type: AgentType):
        """Store data in working memory"""
        self.memory[key] = {
            "value": value,
            "agent": agent_type,
            "timestamp": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=self.ttl_seconds)
        }
        self.access_log[key] = datetime.now()
        logger.debug(f"Set working memory: {key} by {agent_type}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from working memory"""
        if key not in self.memory:
            return None
        
        entry = self.memory[key]
        
        # Check if expired
        if datetime.now() > entry["expires_at"]:
            del self.memory[key]
            logger.debug(f"Working memory expired: {key}")
            return None
        
        self.access_log[key] = datetime.now()
        return entry["value"]
    
    def delete(self, key: str):
        """Delete data from working memory"""
        if key in self.memory:
            del self.memory[key]
            logger.debug(f"Deleted working memory: {key}")
    
    def clear_expired(self):
        """Clear expired entries"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self.memory.items()
            if now > v["expires_at"]
        ]
        
        for key in expired_keys:
            del self.memory[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired memory entries")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all non-expired entries"""
        self.clear_expired()
        return {k: v["value"] for k, v in self.memory.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        self.clear_expired()
        return {
            "total_entries": len(self.memory),
            "keys": list(self.memory.keys()),
            "agents": list(set(v["agent"] for v in self.memory.values()))
        }


class MemoryManager:
    """Main memory manager coordinating all memory types"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationMemory] = {}
        self.working_memory = AgentWorkingMemory()
        self.agent_results: Dict[str, Dict[str, Any]] = {}
    
    def create_conversation(self, conversation_id: str) -> ConversationMemory:
        """Create a new conversation memory"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMemory()
            logger.info(f"Created conversation: {conversation_id}")
        return self.conversations[conversation_id]
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Get an existing conversation"""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
    
    def store_agent_result(self, agent_type: AgentType, result: Dict[str, Any],
                          conversation_id: Optional[str] = None):
        """Store agent result for later retrieval"""
        key = f"{agent_type}_{conversation_id or 'default'}"
        self.agent_results[key] = {
            "result": result,
            "timestamp": datetime.now(),
            "agent": agent_type
        }
        
        # Also store in working memory
        self.working_memory.set(f"result_{agent_type}", result, agent_type)
        
        logger.debug(f"Stored result from {agent_type}")
    
    def get_agent_result(self, agent_type: AgentType,
                        conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve stored agent result"""
        key = f"{agent_type}_{conversation_id or 'default'}"
        entry = self.agent_results.get(key)
        return entry["result"] if entry else None
    
    def get_all_agent_results(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get results from all agents in a conversation"""
        suffix = f"_{conversation_id}" if conversation_id else "_default"
        results = {}
        
        for key, entry in self.agent_results.items():
            if key.endswith(suffix):
                agent_type = entry["agent"]
                results[agent_type] = entry["result"]
        
        return results
    
    def clear_agent_results(self, conversation_id: Optional[str] = None):
        """Clear agent results for a conversation"""
        suffix = f"_{conversation_id}" if conversation_id else "_default"
        keys_to_delete = [k for k in self.agent_results.keys() if k.endswith(suffix)]
        
        for key in keys_to_delete:
            del self.agent_results[key]
        
        logger.info(f"Cleared {len(keys_to_delete)} agent results")
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get full context for a conversation"""
        conversation = self.get_conversation(conversation_id)
        
        context = {
            "conversation_id": conversation_id,
            "conversation_summary": conversation.get_summary() if conversation else None,
            "recent_messages": conversation.get_messages(limit=10) if conversation else [],
            "agent_results": self.get_all_agent_results(conversation_id),
            "working_memory": self.working_memory.get_all()
        }
        
        return context
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations"""
        now = datetime.now()
        old_conversations = []
        
        for conv_id, conv in self.conversations.items():
            age = (now - conv.created_at).total_seconds() / 3600
            if age > max_age_hours:
                old_conversations.append(conv_id)
        
        for conv_id in old_conversations:
            self.delete_conversation(conv_id)
        
        if old_conversations:
            logger.info(f"Cleaned up {len(old_conversations)} old conversations")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory statistics"""
        return {
            "active_conversations": len(self.conversations),
            "total_agent_results": len(self.agent_results),
            "working_memory_stats": self.working_memory.get_stats()
        }


# Create global instance
memory_manager = MemoryManager()


# Utility functions
def store_result(agent_type: AgentType, result: Dict[str, Any],
                conversation_id: Optional[str] = None):
    """Quick access to store agent result"""
    memory_manager.store_agent_result(agent_type, result, conversation_id)


def get_result(agent_type: AgentType, conversation_id: Optional[str] = None) -> Optional[Dict]:
    """Quick access to get agent result"""
    return memory_manager.get_agent_result(agent_type, conversation_id)


def get_all_results(conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Quick access to get all agent results"""
    return memory_manager.get_all_agent_results(conversation_id)