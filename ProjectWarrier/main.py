import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from datetime import datetime, timezone
import json
import logging

# --- SQLAlchemy / Database Setup ---
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete

DATABASE_URL = "sqlite+aiosqlite:///./home_ai.db"

# Note: check_same_thread is for SQLite only.
engine = create_async_engine(DATABASE_URL, connect_args={"check_same_thread": False})
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class Base(DeclarativeBase):
    pass

# --- SQLAlchemy Models (Tables) ---

class Setting(Base):
    """Stores key-value settings like API keys."""
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(primary_key=True, index=True)
    value: Mapped[Optional[str]]

class Todo(Base):
    """Stores todo list items."""
    __tablename__ = "todos"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    task: Mapped[str]
    completed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc).isoformat())

class Reminder(Base):
    """Stores reminders."""
    __tablename__ = "reminders"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    task: Mapped[str]
    due_time: Mapped[datetime]
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc).isoformat())

# --- Pydantic Schemas (API Request/Response Models) ---

# Settings
class SettingSchema(BaseModel):
    key: str
    value: Optional[str] = None

class SettingUpdateSchema(BaseModel):
    openrouter_api_key: Optional[str] = None
    selected_llm_model: Optional[str] = None

# Todo
class TodoCreate(BaseModel):
    task: str

class TodoSchema(BaseModel):
    id: int
    task: str
    completed: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TodoUpdate(BaseModel):
    task: Optional[str] = None
    completed: Optional[bool] = None

# Reminder
class ReminderCreate(BaseModel):
    task: str
    due_time: datetime

class ReminderSchema(BaseModel):
    id: int
    task: str
    due_time: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class ReminderUpdate(BaseModel):
    task: Optional[str] = None
    due_time: Optional[datetime] = None

# Chat
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    data: Optional[dict] = None

# --- Database Dependency ---

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session

async def create_db_and_tables():
    """Create all database tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# --- FastAPI App Initialization ---

app = FastAPI(title="Home AI Assistant Server")

# Add CORS middleware to allow requests from your React GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your GUI's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    """Run database migration on startup."""
    logging.info("Creating database tables...")
    await create_db_and_tables()
    logging.info("Database tables created.")

# --- Database CRUD Helper Functions ---

# Settings
async def get_setting(db: AsyncSession, key: str) -> Optional[str]:
    result = await db.get(Setting, key)
    return result.value if result else None

async def set_setting(db: AsyncSession, key: str, value: str):
    setting = await db.get(Setting, key)
    if setting:
        setting.value = value
    else:
        setting = Setting(key=key, value=value)
        db.add(setting)
    await db.commit()

# Todos
async def create_todo(db: AsyncSession, todo: TodoCreate) -> Todo:
    db_todo = Todo(task=todo.task, completed=False, created_at=datetime.now())
    db.add(db_todo)
    await db.commit()
    await db.refresh(db_todo)
    return db_todo

async def get_todos(db: AsyncSession) -> List[Todo]:
    result = await db.execute(select(Todo).order_by(Todo.created_at.desc()))
    return list(result.scalars().all())

async def update_todo(db: AsyncSession, todo_id: int, todo_data: TodoUpdate) -> Optional[Todo]:
    db_todo = await db.get(Todo, todo_id)
    if not db_todo:
        return None

    update_data = todo_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_todo, key, value)

    await db.commit()
    await db.refresh(db_todo)
    return db_todo

async def delete_todo(db: AsyncSession, todo_id: int) -> bool:
    db_todo = await db.get(Todo, todo_id)
    if db_todo:
        await db.delete(db_todo)
        await db.commit()
        return True
    return False

# Reminders (Similar CRUD functions)
async def create_reminder(db: AsyncSession, reminder: ReminderCreate) -> Reminder:
    db_reminder = Reminder(task=reminder.task, due_time=reminder.due_time, created_at=datetime.now())
    db.add(db_reminder)
    await db.commit()
    await db.refresh(db_reminder)
    return db_reminder

async def get_reminders(db: AsyncSession) -> List[Reminder]:
    result = await db.execute(select(Reminder).order_by(Reminder.due_time.asc()))
    return list(result.scalars().all())

# (Update/Delete for Reminders would be built similarly)

# --- LLM Service ---

# We use the official 'openai' library, as OpenRouter is OpenAI-compatible.
from openai import OpenAI, AsyncOpenAI

class LLMService:
    """Handles all interactions with the OpenRouter LLM."""
    
    def get_client(self, api_key: str) -> AsyncOpenAI:
        """Initializes the AsyncOpenAI client configured for OpenRouter."""
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def perform_intent_recognition(self, prompt: str, client: AsyncOpenAI, model: str) -> dict:
        """
        Uses the LLM to classify the user's prompt into a specific intent and extracts data.
        """
        # This system prompt is the "brain" of the intent recognizer.
        # It forces the LLM to reply ONLY with JSON.
        system_prompt = f"""
        You are an intent recognition engine for a home assistant.
        Analyze the user's prompt and classify it into one of the following intents.
        Your response MUST be a single, valid JSON object and nothing else.

        Available Intents:
        1. "create_todo": User wants to add a task to their to-do list.
           - Required data: "task" (string)
        2. "create_reminder": User wants to be reminded of something.
           - Required data: "task" (string), "time" (string in ISO 8601 format, e.g., "2025-11-17T17:35:42.123456+00:00")
        3. "iot_control": User wants to control a smart home device.
           - Required data: "device" (string), "action" (string, e.g., "on", "off", "dim"), "location" (string, optional)
        4. "conversation": User is just chatting, asking a question, or the intent doesn't match others.

        Examples:
        - Prompt: "add milk to my shopping list"
          JSON: {{"intent": "create_todo", "task": "buy milk"}}
        - Prompt: "remind me to call mom at 5pm"
          JSON: {{"intent": "create_reminder", "task": "call mom", "time": "YYYY-MM-DDTH17:00:00"}}
        - Prompt: "turn on the living room light"
          JSON: {{"intent": "iot_control", "device": "light", "action": "on", "location": "living room"}}
        - Prompt: "who won the world series in 2020?"
          JSON: {{"intent": "conversation"}}
        """

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"} # Force JSON output
            )
            
            intent_data = json.loads(response.choices[0].message.content)
            return intent_data
        
        except json.JSONDecodeError:
            logging.error("LLM did not return valid JSON.")
            return {"intent": "error", "error": "Invalid JSON response from LLM."}
        except Exception as e:
            logging.error(f"Error in LLM call: {e}")
            return {"intent": "error", "error": str(e)}

    async def get_conversational_reply(self, prompt: str, client: AsyncOpenAI, model: str) -> str:
        """Gets a general conversational response from the LLM."""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful home AI assistant. Be friendly, concise, and helpful."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in LLM call: {e}")
            return f"Sorry, I encountered an error: {e}"

# --- API Endpoints ---

# Settings Endpoints
@app.post("/api/settings")
async def save_settings(settings: SettingUpdateSchema, db: AsyncSession = Depends(get_db)):
    """Saves the OpenRouter API key and selected model."""
    if settings.openrouter_api_key is not None:
        await set_setting(db, "openrouter_api_key", settings.openrouter_api_key)
    if settings.selected_llm_model is not None:
        await set_setting(db, "selected_llm_model", settings.selected_llm_model)
    return {"message": "Settings saved successfully."}

@app.get("/api/settings", response_model=SettingUpdateSchema)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Retrieves the saved OpenRouter API key and model."""
    api_key = await get_setting(db, "openrouter_api_key")
    model = await get_setting(db, "selected_llm_model")
    return SettingUpdateSchema(openrouter_api_key=api_key, selected_llm_model=model)

# Todo Endpoints
@app.post("/api/todos", response_model=TodoSchema)
async def add_todo_endpoint(todo: TodoCreate, db: AsyncSession = Depends(get_db)):
    return await create_todo(db, todo)

@app.get("/api/todos", response_model=List[TodoSchema])
async def get_todos_endpoint(db: AsyncSession = Depends(get_db)):
    return await get_todos(db)

@app.put("/api/todos/{todo_id}", response_model=TodoSchema)
async def update_todo_endpoint(todo_id: int, todo_data: TodoUpdate, db: AsyncSession = Depends(get_db)):
    updated_todo = await update_todo(db, todo_id, todo_data)
    if not updated_todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return updated_todo

@app.delete("/api/todos/{todo_id}")
async def delete_todo_endpoint(todo_id: int, db: AsyncSession = Depends(get_db)):
    if not await delete_todo(db, todo_id):
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"message": "Todo deleted"}

# Reminder Endpoints
@app.post("/api/reminders", response_model=ReminderSchema)
async def add_reminder_endpoint(reminder: ReminderCreate, db: AsyncSession = Depends(get_db)):
    return await create_reminder(db, reminder)

@app.get("/api/reminders", response_model=List[ReminderSchema])
async def get_reminders_endpoint(db: AsyncSession = Depends(get_db)):
    return await get_reminders(db)


# --- The Main "Brain" Endpoint ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Main endpoint for all user interactions.
    It performs intent recognition and routes to the correct service.
    """
    # 1. Get API key and model from database
    api_key = await get_setting(db, "openrouter_api_key")
    model = await get_setting(db, "selected_llm_model")

    if not api_key:
        api_key = "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"
        # raise HTTPException(status_code=400, detail="OpenRouter API key not set. Please set it in /api/settings.")
    if not model:
        # Use a sensible default if not set
        model = "xiaomi/mimo-v2-flash:free"

    # 2. Initialize LLM Service
    llm_service = LLMService()
    try:
        client = llm_service.get_client(api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM client: {e}")

    # 3. Perform Intent Recognition
    intent_data = await llm_service.perform_intent_recognition(request.prompt, client, model)
    intent = intent_data.get("intent")

    # 4. Route based on intent
    
    if intent == "create_todo":
        task = intent_data.get("task")
        if not task:
            return ChatResponse(response="I see you want to add a todo, but I didn't catch the task.", intent=intent, data=intent_data)
        
        new_todo = await create_todo(db, TodoCreate(task=task))

        todo_data = TodoSchema.model_validate(new_todo).model_dump()
        return ChatResponse(
            response=f"I've added '{new_todo.task}' to your to-do list.",
            intent=intent,
            data=todo_data
        )

    elif intent == "create_reminder":
        task = intent_data.get("task")
        time_str = intent_data.get("time")
        
        if not task or not time_str:
            return ChatResponse(response="I can set a reminder, but I need both a task and a time.", intent=intent, data=intent_data)

        try:
            # Parse the ISO format string from the LLM
            due_time = datetime.fromisoformat(time_str)
            new_reminder = await create_reminder(db, ReminderCreate(task=task, due_time=due_time))

            reminder_data = ReminderSchema.model_validate(new_reminder).model_dump()
            return ChatResponse(
                response=f"OK, I'll remind you to '{new_reminder.task}' at {new_reminder.due_time.strftime('%I:%M %p on %B %d')}.",
                intent=intent,
                data=reminder_data
            )
        except Exception as e:
            logging.error(f"Failed to create reminder: {e}")
            return ChatResponse(response=f"I couldn't understand the time '{time_str}'. Please try again.", intent=intent, data=intent_data)

    elif intent == "iot_control":
        # Phase 2: This is where we'll hook in the device control logic
        return ChatResponse(
            response=f"I understood you want to control a device: {intent_data.get('device')}. Device control is not implemented yet.",
            intent=intent,
            data=intent_data
        )
    
    elif intent == "conversation":
        # Just get a standard conversational reply
        reply = await llm_service.get_conversational_reply(request.prompt, client, model)
        return ChatResponse(response=reply, intent=intent)

    else:
        # Handle errors or unknown intents
        reply = f"Sorry, I'm not sure how to handle that. (Intent: {intent})"
        return ChatResponse(response=reply, intent="unknown")

# --- Run the Server ---

if __name__ == "__main__":
    """Run the FastAPI server with uvicorn."""
    print("Starting Home AI Backend Server...")
    print("Access the API docs at http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
