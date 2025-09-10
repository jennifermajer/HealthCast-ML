#!/usr/bin/env python3
"""
Conversation Persistence System for Claude Code Sessions

This script helps preserve conversation context between system crashes and sessions.
It automatically saves conversation state and can restore previous context.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

class ConversationPersistence:
    """Manages conversation state persistence across sessions"""
    
    def __init__(self, session_dir: str = None):
        """Initialize conversation persistence system"""
        self.session_dir = Path(session_dir) if session_dir else Path.home() / '.claude_sessions'
        self.session_dir.mkdir(exist_ok=True)
        self.session_file = self.session_dir / f"session_{datetime.now().strftime('%Y%m%d')}.json"
        
    def save_conversation_state(self, context: Dict[str, Any]) -> str:
        """
        Save current conversation state
        
        Args:
            context: Dictionary containing conversation context
                    - project_path: Current working directory
                    - tasks: List of current tasks/todos
                    - context_summary: Summary of what's been done
                    - important_files: List of important files accessed
                    - errors_encountered: List of errors and solutions
        
        Returns:
            Session ID for later retrieval
        """
        session_id = f"session_{int(time.time())}"
        
        session_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'project_path': str(Path.cwd()),
            **context
        }
        
        # Load existing sessions
        existing_sessions = self._load_sessions()
        existing_sessions[session_id] = session_data
        
        # Save to file
        with open(self.session_file, 'w') as f:
            json.dump(existing_sessions, f, indent=2)
        
        print(f"✅ Session saved: {session_id}")
        return session_id
    
    def restore_conversation_context(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Restore conversation context from previous session
        
        Args:
            session_id: Specific session to restore (if None, returns latest)
            
        Returns:
            Conversation context dictionary or None if not found
        """
        sessions = self._load_sessions()
        
        if not sessions:
            print("❌ No previous sessions found")
            return None
        
        if session_id:
            context = sessions.get(session_id)
            if not context:
                print(f"❌ Session {session_id} not found")
                return None
        else:
            # Get most recent session
            latest_session = max(sessions.values(), key=lambda x: x['timestamp'])
            context = latest_session
            session_id = context['session_id']
        
        print(f"✅ Restored session: {session_id}")
        print(f"📅 From: {context['timestamp']}")
        
        return context
    
    def list_sessions(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """List recent conversation sessions"""
        sessions = self._load_sessions()
        
        if not sessions:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = []
        
        for session_data in sessions.values():
            session_time = datetime.fromisoformat(session_data['timestamp'])
            if session_time >= cutoff_date:
                recent_sessions.append(session_data)
        
        # Sort by timestamp (newest first)
        recent_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent_sessions
    
    def cleanup_old_sessions(self, days_to_keep: int = 30):
        """Remove sessions older than specified days"""
        sessions = self._load_sessions()
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_sessions = {}
        removed_count = 0
        
        for session_id, session_data in sessions.items():
            session_time = datetime.fromisoformat(session_data['timestamp'])
            if session_time >= cutoff_date:
                cleaned_sessions[session_id] = session_data
            else:
                removed_count += 1
        
        with open(self.session_file, 'w') as f:
            json.dump(cleaned_sessions, f, indent=2)
        
        print(f"🧹 Cleaned up {removed_count} old sessions")
    
    def _load_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Load existing sessions from file"""
        if not self.session_file.exists():
            return {}
        
        try:
            with open(self.session_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

def save_current_context():
    """Save current context - can be called from anywhere"""
    persistence = ConversationPersistence()
    
    # Gather context information
    context = {
        'tasks': get_current_tasks(),
        'context_summary': generate_context_summary(),
        'important_files': get_important_files(),
        'system_state': get_system_state()
    }
    
    return persistence.save_conversation_state(context)

def restore_last_context():
    """Restore and display last conversation context"""
    persistence = ConversationPersistence()
    context = persistence.restore_conversation_context()
    
    if context:
        print("\n" + "="*60)
        print("📝 PREVIOUS SESSION CONTEXT")
        print("="*60)
        
        if 'context_summary' in context:
            print(f"📋 Summary: {context['context_summary']}")
        
        if 'tasks' in context and context['tasks']:
            print(f"\n📝 Previous tasks:")
            for task in context['tasks']:
                print(f"  • {task}")
        
        if 'important_files' in context and context['important_files']:
            print(f"\n📁 Important files:")
            for file_path in context['important_files']:
                print(f"  • {file_path}")
        
        print("="*60 + "\n")
    
    return context

def get_current_tasks() -> List[str]:
    """Get current tasks/todos - placeholder for integration"""
    # This would integrate with your todo system
    return [
        "Climate-health ML analysis pipeline",
        "System crash investigation", 
        "Memory optimization"
    ]

def generate_context_summary() -> str:
    """Generate summary of current work context"""
    return f"Working on climate-health ML analysis pipeline in {Path.cwd().name}"

def get_important_files() -> List[str]:
    """Get list of important files accessed in session"""
    # This could track recently accessed files
    important_patterns = [
        'run_analysis.py',
        'config.yaml',
        'requirements.txt'
    ]
    
    important_files = []
    for pattern in important_patterns:
        for file_path in Path.cwd().glob(f"**/{pattern}"):
            important_files.append(str(file_path.relative_to(Path.cwd())))
    
    return important_files

def get_system_state() -> Dict[str, Any]:
    """Get current system state information"""
    return {
        'working_directory': str(Path.cwd()),
        'python_version': os.sys.version,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Conversation Persistence Manager')
    parser.add_argument('--save', action='store_true', help='Save current context')
    parser.add_argument('--restore', action='store_true', help='Restore last context')
    parser.add_argument('--list', action='store_true', help='List recent sessions')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old sessions')
    
    args = parser.parse_args()
    
    if args.save:
        save_current_context()
    elif args.restore:
        restore_last_context()
    elif args.list:
        persistence = ConversationPersistence()
        sessions = persistence.list_sessions()
        if sessions:
            print("\n📋 Recent Sessions:")
            for session in sessions:
                print(f"  {session['session_id']}: {session['timestamp']} - {session.get('context_summary', 'No summary')}")
        else:
            print("❌ No recent sessions found")
    elif args.cleanup:
        persistence = ConversationPersistence()
        persistence.cleanup_old_sessions()
    else:
        # Interactive mode
        print("🔄 Conversation Persistence System")
        restore_last_context()