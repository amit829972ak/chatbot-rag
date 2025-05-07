import os
import time
import datetime
import json
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the database engine
def get_database_url():
    # Use the environment variable if available
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # If not, use SQLite in memory for simplicity
        db_url = "sqlite:///chat_database.db"
    return db_url

# Create engine and base
engine = create_engine(get_database_url())
Base = declarative_base()

# Define models
class User(Base):
    """User model to store user information."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

class Conversation(Base):
    """Conversation model to group related messages."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"

class Message(Base):
    """Message model to store chat messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    image_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"

class KnowledgeItem(Base):
    """Model for storing knowledge items with embeddings."""
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Text)  # JSON serialized embedding
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<KnowledgeItem(id={self.id}, source={self.source})>"

def initialize_database():
    """Create database tables if they don't exist with retry logic."""
    try:
        execute_with_retry(Base.metadata.create_all, engine)
        logger.info("Database tables created successfully")
        print("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        print(f"Error initializing database: {str(e)}")

def execute_with_retry(func, *args, **kwargs):
    """Execute a database function with retry logic."""
    max_retries = 5
    retry_delay = 1.5  # seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.error(f"Database connection error: {str(e)}. Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"Database connection error after {max_retries} attempts: {str(e)}")
                raise

def get_or_create_user(username=None):
    """Get or create a user."""
    def _get_or_create_user():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Try to get the first user (for simplicity we just use one user)
            user = session.query(User).first()
            
            # If no user exists, create one
            if not user:
                user = User(username=username or "Default User")
                session.add(user)
                session.commit()
            
            return user
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_user)

def get_or_create_conversation(user_id, title=None):
    """Get or create a conversation for a user."""
    def _get_or_create_conversation():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Try to get the most recent conversation
            conversation = session.query(Conversation)\
                .filter(Conversation.user_id == user_id)\
                .order_by(Conversation.created_at.desc())\
                .first()
            
            # If no conversation exists or a new title is provided, create one
            if not conversation or title:
                conversation = Conversation(
                    user_id=user_id,
                    title=title or "New Conversation"
                )
                session.add(conversation)
                session.commit()
            
            return conversation
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_conversation)

def add_message_to_db(conversation_id, role, content, image_data=None):
    """Add a message to the database."""
    def _add_message():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                image_data=image_data
            )
            session.add(message)
            session.commit()
            return True
        finally:
            session.close()
    
    return execute_with_retry(_add_message)

def get_conversation_messages(conversation_id, limit=100):
    """Get messages for a conversation."""
    def _get_messages():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            messages = session.query(Message)\
                .filter(Message.conversation_id == conversation_id)\
                .order_by(Message.timestamp)\
                .limit(limit)\
                .all()
            
            # Convert to dictionary for easy serialization
            result = []
            for message in messages:
                result.append({
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat()
                })
            
            return result
        finally:
            session.close()
    
    return execute_with_retry(_get_messages)

def add_knowledge_item(content, embedding, source=None):
    """Add a knowledge item with embedding to the database."""
    def _add_knowledge_item():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Convert embedding to JSON string
            embedding_json = json.dumps(embedding)
            
            item = KnowledgeItem(
                content=content,
                embedding=embedding_json,
                source=source
            )
            session.add(item)
            session.commit()
            return item.id
        finally:
            session.close()
    
    return execute_with_retry(_add_knowledge_item)

def get_all_knowledge_items():
    """Get all knowledge items with embeddings."""
    def _get_items():
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            items = session.query(KnowledgeItem).all()
            
            result = []
            for item in items:
                # Parse the embedding JSON
                embedding = json.loads(item.embedding)
                result.append({
                    "id": item.id,
                    "content": item.content,
                    "embedding": embedding,
                    "source": item.source,
                    "created_at": item.created_at.isoformat()
                })
            
            return result
        finally:
            session.close()
    
    return execute_with_retry(_get_items)
