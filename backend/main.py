# # backend/main.py
# # FINAL, FULLY-FEATURED SINGLE-FILE BACKEND WITH AUTOMATED REPORT SUMMARY

# import os
# import faiss
# import numpy as np
# import io
# import json
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# from pydantic_settings import BaseSettings
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
# from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
# from sqlalchemy.orm import sessionmaker, Session
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
# from pydantic import BaseModel
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta

# # --- 1. CONFIGURATION ---
# class Settings(BaseSettings):
#     google_api_key: str
#     class Config:
#         env_file = ".env"
# settings = Settings()

# # --- 2. DATABASE SETUP ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "visuhealth.db")
# SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # --- 3. DATABASE MODELS ---
# class User(Base):
#     __tablename__ = "users"; id = Column(Integer, primary_key=True, index=True); email = Column(String, unique=True, index=True); hashed_password = Column(String); age = Column(Integer); weight_kg = Column(Float, nullable=True); height_cm = Column(Float, nullable=True)
# class ChatSession(Base):
#     __tablename__ = "chat_sessions"; id = Column(Integer, primary_key=True, index=True); user_id = Column(Integer, ForeignKey("users.id")); created_at = Column(DateTime(timezone=True), server_default=func.now()); title = Column(String, default="New Conversation")
# class ChatMessage(Base):
#     __tablename__ = "chat_messages"; id = Column(Integer, primary_key=True, index=True); session_id = Column(Integer, ForeignKey("chat_sessions.id")); sender = Column(String); text = Column(String); timestamp = Column(DateTime(timezone=True), server_default=func.now())
# class MedicalReport(Base):
#     __tablename__ = "medical_reports"; id = Column(Integer, primary_key=True, index=True); user_id = Column(Integer, ForeignKey("users.id")); filename = Column(String); extracted_text = Column(String); uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
# Base.metadata.create_all(bind=engine)

# # --- 4. PYDANTIC SCHEMAS ---
# class Token(BaseModel): access_token: str; token_type: str
# class TokenData(BaseModel): email: Optional[str] = None
# class UserBase(BaseModel): email: str
# class UserCreate(UserBase): password: str; age: int; weight_kg: Optional[float] = None; height_cm: Optional[float] = None
# class UserSchema(UserBase):
#     id: int; age: int; weight_kg: Optional[float] = None; height_cm: Optional[float] = None
#     class Config: from_attributes = True
# class ChatMessageBase(BaseModel): sender: str; text: str
# class ChatMessageSchema(ChatMessageBase):
#     id: int; timestamp: datetime
#     class Config: from_attributes = True
# class ChatSessionSchema(BaseModel):
#     id: int; user_id: int; created_at: datetime; title: str
#     class Config: from_attributes = True
# class MedicalReportSchema(BaseModel):
#     id: int; user_id: int; filename: str; uploaded_at: datetime
#     extracted_text: str 
#     class Config: from_attributes = True
# class ReportTextInput(BaseModel): report_text: str
# class ReportSummaryOutput(BaseModel): key_findings: str; what_it_means: str; next_steps: str

# # --- 5. AUTHENTICATION LOGIC ---
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# SECRET_KEY = "a_very_secret_key_for_visuhealth"; ALGORITHM = "HS256"; ACCESS_TOKEN_EXPIRE_MINUTES = 30
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# def verify_password(p, h): return pwd_context.verify(p, h)
# def get_password_hash(p): return pwd_context.hash(p)
# def create_access_token(data: dict):
#     expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES); data.update({"exp": expire})
#     return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# # --- 6. CRUD (DATABASE OPERATIONS) ---
# def get_user_by_email(db: Session, email: str): return db.query(User).filter(User.email == email).first()
# def create_chat_session(db: Session, user_id: int, title: str = "New Conversation"):
#     s = ChatSession(user_id=user_id, title=title); db.add(s); db.commit(); db.refresh(s); return s
# def create_chat_message(db: Session, session_id: int, sender: str, text: str):
#     m = ChatMessage(session_id=session_id, sender=sender, text=text); db.add(m); db.commit(); db.refresh(m); return m
# def create_medical_report(db: Session, user_id: int, filename: str, text: str):
#     r = MedicalReport(user_id=user_id, filename=filename, extracted_text=text); db.add(r); db.commit(); db.refresh(r); return r
# def get_latest_user_report_text(db: Session, user_id: int):
#     r = db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).first()
#     return r.extracted_text if r else None



# def delete_user_reports(db: Session, user_id: int):
#     """Deletes all medical reports for a specific user."""
#     # Query for all reports owned by the user
#     reports_to_delete = db.query(MedicalReport).filter(MedicalReport.user_id == user_id).all()
    
#     if not reports_to_delete:
#         return 0 # Return 0 if no reports were found to delete
        
#     num_deleted = len(reports_to_delete)
    
#     for report in reports_to_delete:
#         db.delete(report)
        
#     db.commit()
#     return num_deleted # Return the number of deleted reports



# def get_user_reports(db: Session, user_id: int):
#     """Retrieves all reports for a specific user."""
#     return db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).all()

# # --- 7. RAG SERVICE ---
# KNOWLEDGE_BASE = {"liver_alcohol": "...", "heart_diet": "...", "lungs_smoking": "...", "general_health": "..."}
# class RagService:
#     def __init__(self):
#         print("Initializing RAG Service..."); genai.configure(api_key=settings.google_api_key); self.model = genai.GenerativeModel('gemini-1.5-flash-latest'); self.encoder = SentenceTransformer('all-MiniLM-L6-v2'); self.documents = list(KNOWLEDGE_BASE.values()); self.doc_embeddings = self.encoder.encode(self.documents); self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1]); self.index.add(self.doc_embeddings.astype(np.float32)); print("RAG Service Initialized Successfully.")
#     def _retrieve_documents(self, q: str, k: int = 2):
#         qe = self.encoder.encode([q]).astype(np.float32); _, i = self.index.search(qe, k); return [self.documents[idx] for idx in i[0]]
#     def ask_question(self, question: str, user_profile: dict, lifestyle_data: dict, report_text: str | None):
#         general_context = "\n\n".join(self._retrieve_documents(question)); prompt_parts = ["You are a helpful and empathetic health educator for VisuHealth...", "---", f"General Context:\n{general_context}", "---", f"User Profile:\n{user_profile}", f"Lifestyle Data:\n{lifestyle_data}", "---"]
#         if report_text: prompt_parts.extend([f"CRITICAL CONTEXT from user's uploaded medical report:\n{report_text}", "---"])
#         prompt_parts.extend([f"User's Question: \"{question}\"", "Based on all info, provide a helpful answer."]); prompt = "\n".join(prompt_parts)
#         try: return self.model.generate_content(prompt).text
#         except Exception as e: print(f"Error calling Gemini API: {e}"); return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."
#     def summarize_report(self, report_text: str):
#         prompt = f"""
#         You are a medical data analyst AI. Your task is to read the following medical lab report text
#         and return a structured summary in a single, valid JSON object. The JSON object must have
#         three string keys: "key_findings", "what_it_means", and "next_steps".
#         - "key_findings": A markdown bulleted list of the most important results.
#         - "what_it_means": A simple explanation of what these findings suggest.
#         - "next_steps": A short, bulleted list of topics to discuss with a doctor. Do not give medical advice.
#         --- MEDICAL REPORT TEXT ---
#         {report_text}
#         --- END OF REPORT ---
#         Now, generate ONLY the JSON object.
#         """
#         try:
#             # THE FIX: We no longer use the JSON mime type, as it can be unreliable.
#             # We will ask the model for text and clean it up ourselves for maximum robustness.
#             response = self.model.generate_content(prompt)
            
#             # Clean the response to ensure it's valid JSON
#             cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
            
#             return cleaned_text # Return the cleaned JSON string
            
#         except Exception as e:
#             print(f"Error calling Gemini API for summary: {e}")
#             return '{"key_findings": "Error: Could not analyze report.", "what_it_means": "The AI service encountered an issue.", "next_steps": "Please try again later."}'

# # --- 8. FASTAPI APP & LIFESPAN ---
# lifespan_data = {}
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server starting up..."); lifespan_data["rag_service"] = RagService(); yield; print("Server shutting down...")
# app = FastAPI(lifespan=lifespan)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # --- 9. API DEPENDENCIES & ENDPOINTS ---
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db      
#     finally:
#         db.close()
# async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); email: str = payload.get("sub")
#         if email is None: raise exception
#     except JWTError: raise exception
#     user = get_user_by_email(db, email=email)
#     if user is None: raise exception
#     return user

# @app.post("/register/", response_model=UserSchema)
# def register_user(user: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user_by_email(db, email=user.email);
#     if db_user: raise HTTPException(status_code=400, detail="Email already registered")
#     hashed_password = get_password_hash(user.password)
#     new_user = User(email=user.email, hashed_password=hashed_password, age=user.age, weight_kg=user.weight_kg, height_cm=user.height_cm)
#     db.add(new_user); db.commit(); db.refresh(new_user); return new_user

# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = get_user_by_email(db, email=form_data.username)
#     if not user or not verify_password(form_data.password, user.hashed_password): raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
#     access_token = create_access_token(data={"sub": user.email}); return {"access_token": access_token, "token_type": "bearer"}



# @app.get("/users/me/", response_model=UserSchema)
# async def read_users_me(current_user: User = Depends(get_current_user)): return current_user


# @app.get("/reports", response_model=list[MedicalReportSchema])
# async def get_reports_for_user(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Fetches a list of all reports uploaded by the current user."""
#     return get_user_reports(db=db, user_id=current_user.id)


# @app.delete("/reports/clear")
# async def clear_all_reports_for_user(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Deletes all reports for the currently authenticated user."""
#     num_deleted = delete_user_reports(db=db, user_id=current_user.id)
#     return {"message": f"Successfully deleted {num_deleted} report(s)."}






# @app.post("/reports/upload", response_model=MedicalReportSchema)
# async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     if file.content_type != "application/pdf": raise HTTPException(status_code=400, detail="Invalid file type.")
#     try:
#         pdf_bytes = await file.read(); pdf_stream = io.BytesIO(pdf_bytes); reader = PdfReader(pdf_stream)
#         text = "".join(page.extract_text() + "\n" for page in reader.pages)
#         if not text.strip(): raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
#         return create_medical_report(db=db, user_id=current_user.id, filename=file.filename, text=text)
#     except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# @app.post("/reports/summarize", response_model=ReportSummaryOutput)
# async def summarize_medical_report(report_input: ReportTextInput, current_user: User = Depends(get_current_user)):
#     summary_json_string = lifespan_data["rag_service"].summarize_report(report_text=report_input.report_text)
#     try: return json.loads(summary_json_string)
#     except json.JSONDecodeError: raise HTTPException(status_code=500, detail="AI returned an invalid summary format.")

# @app.post("/chat/sessions", response_model=ChatSessionSchema)
# async def create_new_session(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return create_chat_session(db, user_id=current_user.id)

# class ChatInput(BaseModel): question: str; lifestyle_data: dict
# @app.post("/chat/sessions/{session_id}", response_model=ChatMessageSchema)
# async def post_message_to_session(session_id: int, chat_input: ChatInput, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
#     if not session: raise HTTPException(status_code=404, detail="Session not found")
#     create_chat_message(db, session_id=session_id, sender="user", text=chat_input.question)
#     rag_service = lifespan_data["rag_service"]
#     user_profile = UserSchema.model_validate(current_user).model_dump()
#     report_text = get_latest_user_report_text(db, user_id=current_user.id)
#     answer = rag_service.ask_question(question=chat_input.question, user_profile=user_profile, lifestyle_data=chat_input.lifestyle_data, report_text=report_text)
#     ai_message = create_chat_message(db, session_id=session_id, sender="ai", text=answer)
#     return ai_message







# backend/main.py
# FINAL, FULLY-FEATURED SINGLE-FILE BACKEND

# import os
# import faiss
# import numpy as np
# import io
# import json
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# from pydantic_settings import BaseSettings
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
# from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
# from sqlalchemy.orm import sessionmaker, Session
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
# from pydantic import BaseModel
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta

# # --- 1. CONFIGURATION ---
# class Settings(BaseSettings):
#     google_api_key: str
#     class Config:
#         env_file = ".env"
# settings = Settings()

# # --- 2. DATABASE SETUP ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "visuhealth.db")
# SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # --- 3. DATABASE MODELS ---
# class User(Base):
#     __tablename__ = "users"; id = Column(Integer, primary_key=True, index=True); email = Column(String, unique=True, index=True); hashed_password = Column(String); age = Column(Integer); weight_kg = Column(Float, nullable=True); height_cm = Column(Float, nullable=True)
# class ChatSession(Base):
#     __tablename__ = "chat_sessions"; id = Column(Integer, primary_key=True, index=True); user_id = Column(Integer, ForeignKey("users.id")); created_at = Column(DateTime(timezone=True), server_default=func.now()); title = Column(String, default="New Conversation")
# class ChatMessage(Base):
#     __tablename__ = "chat_messages"; id = Column(Integer, primary_key=True, index=True); session_id = Column(Integer, ForeignKey("chat_sessions.id")); sender = Column(String); text = Column(String); timestamp = Column(DateTime(timezone=True), server_default=func.now())
# class MedicalReport(Base):
#     __tablename__ = "medical_reports"; id = Column(Integer, primary_key=True, index=True); user_id = Column(Integer, ForeignKey("users.id")); filename = Column(String); extracted_text = Column(String); uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
# Base.metadata.create_all(bind=engine)

# # --- 4. PYDANTIC SCHEMAS ---
# class Token(BaseModel): access_token: str; token_type: str
# class TokenData(BaseModel): email: Optional[str] = None
# class UserBase(BaseModel): email: str
# class UserCreate(UserBase): password: str; age: int; weight_kg: Optional[float] = None; height_cm: Optional[float] = None
# class UserSchema(UserBase):
#     id: int; age: int; weight_kg: Optional[float] = None; height_cm: Optional[float] = None
#     class Config: from_attributes = True
# class ChatMessageBase(BaseModel): sender: str; text: str
# class ChatMessageSchema(ChatMessageBase):
#     id: int; timestamp: datetime
#     class Config: from_attributes = True
# class ChatSessionSchema(BaseModel):
#     id: int; user_id: int; created_at: datetime; title: str
#     class Config: from_attributes = True
# class MedicalReportSchema(BaseModel):
#     id: int; user_id: int; filename: str; uploaded_at: datetime; extracted_text: str
#     class Config: from_attributes = True
# class ReportTextInput(BaseModel): report_text: str
# class ReportSummaryOutput(BaseModel): key_findings: str; what_it_means: str; next_steps: str
# class ChatInput(BaseModel):
#     question: str
#     lifestyle_data: dict
#     language: Optional[str] = "English"

# # --- 5. AUTHENTICATION LOGIC ---
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# SECRET_KEY = "a_very_secret_key_for_visuhealth"; ALGORITHM = "HS256"; ACCESS_TOKEN_EXPIRE_MINUTES = 30
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# def verify_password(p, h): return pwd_context.verify(p, h)
# def get_password_hash(p): return pwd_context.hash(p)
# def create_access_token(data: dict):
#     expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES); data.update({"exp": expire})
#     return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# # --- 6. CRUD (DATABASE OPERATIONS) ---
# def get_user_by_email(db: Session, email: str): return db.query(User).filter(User.email == email).first()
# def create_chat_session(db: Session, user_id: int, title: str = "New Conversation"):
#     s = ChatSession(user_id=user_id, title=title); db.add(s); db.commit(); db.refresh(s); return s
# def create_chat_message(db: Session, session_id: int, sender: str, text: str):
#     m = ChatMessage(session_id=session_id, sender=sender, text=text); db.add(m); db.commit(); db.refresh(m); return m
# def create_medical_report(db: Session, user_id: int, filename: str, text: str):
#     r = MedicalReport(user_id=user_id, filename=filename, extracted_text=text); db.add(r); db.commit(); db.refresh(r); return r
# def get_latest_user_report_text(db: Session, user_id: int):
#     r = db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).first()
#     return r.extracted_text if r else None
# def get_user_reports(db: Session, user_id: int): return db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).all()
# def delete_user_reports(db: Session, user_id: int):
#     reports_to_delete = db.query(MedicalReport).filter(MedicalReport.user_id == user_id); num_deleted = reports_to_delete.count(); reports_to_delete.delete(); db.commit(); return num_deleted
# def get_user_chat_sessions(db: Session, user_id: int): return db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
# def get_session_messages(db: Session, session_id: int): return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()

# # --- 7. RAG SERVICE ---
# KNOWLEDGE_BASE = { "liver_alcohol": "...", "heart_diet": "...", "lungs_smoking": "...", "general_health": "..."}
# class RagService:
#     def __init__(self):
#         print("Initializing RAG Service..."); genai.configure(api_key=settings.google_api_key); self.model = genai.GenerativeModel('gemini-1.5-flash-latest'); self.encoder = SentenceTransformer('all-MiniLM-L6-v2'); self.documents = list(KNOWLEDGE_BASE.values()); self.doc_embeddings = self.encoder.encode(self.documents); self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1]); self.index.add(self.doc_embeddings.astype(np.float32)); print("RAG Service Initialized Successfully.")
#     def _retrieve_documents(self, q: str, k: int = 2):
#         qe = self.encoder.encode([q]).astype(np.float32); _, i = self.index.search(qe, k); return [self.documents[idx] for idx in i[0]]
#     def ask_question(self, question: str, user_profile: dict, lifestyle_data: dict, report_text: str | None, language: str):
#         general_context = "\n\n".join(self._retrieve_documents(question)); prompt_parts = ["You are a helpful and empathetic health educator...", "---", f"General Context:\n{general_context}", "---", f"User Profile:\n{user_profile}", f"Lifestyle Data:\n{lifestyle_data}", "---"]
#         if report_text: prompt_parts.extend([f"CRITICAL CONTEXT from user's report:\n{report_text}", "---"])
#         prompt_parts.extend([f"User's Question: \"{question}\"", "Provide a helpful answer.", f"IMPORTANT: Your entire response MUST be in {language}."]); prompt = "\n".join(prompt_parts)
#         try: return self.model.generate_content(prompt).text
#         except Exception as e: print(f"Error calling Gemini API: {e}"); return "I'm sorry, an error occurred with the AI service."
#     def summarize_report(self, report_text: str):
#         prompt = f"""You are a medical data analyst AI. Return a JSON object with three keys: "key_findings", "what_it_means", and "next_steps"... Report Text: {report_text} ... Now, generate the JSON summary."""
#         try:
#             config = genai.types.GenerationConfig(response_mime_type="application/json")
#             response = self.model.generate_content(prompt, generation_config=config); return response.text
#         except Exception as e: print(f"Error calling Gemini API for summary: {e}"); return '{"key_findings": "Error.", "what_it_means": "The AI service failed to analyze the report.", "next_steps": "Please try again."}'

# # --- 8. FASTAPI APP & LIFESPAN ---
# lifespan_data = {}
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server starting up..."); lifespan_data["rag_service"] = RagService(); yield; print("Server shutting down...")
# app = FastAPI(lifespan=lifespan)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # --- 9. API DEPENDENCIES & ENDPOINTS ---
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db      
#     finally:
#         db.close()
# async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); email: str = payload.get("sub")
#         if email is None: raise exception
#     except JWTError: raise exception
#     user = get_user_by_email(db, email=email)
#     if user is None: raise exception
#     return user

# @app.post("/register/", response_model=UserSchema)
# def register_user(user: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user_by_email(db, email=user.email);
#     if db_user: raise HTTPException(status_code=400, detail="Email already registered")
#     hashed_password = get_password_hash(user.password)
#     new_user = User(email=user.email, hashed_password=hashed_password, age=user.age, weight_kg=user.weight_kg, height_cm=user.height_cm)
#     db.add(new_user); db.commit(); db.refresh(new_user); return new_user

# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = get_user_by_email(db, email=form_data.username)
#     if not user or not verify_password(form_data.password, user.hashed_password): raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
#     access_token = create_access_token(data={"sub": user.email}); return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me/", response_model=UserSchema)
# async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

# @app.get("/reports", response_model=list[MedicalReportSchema])
# async def get_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return get_user_reports(db=db, user_id=current_user.id)

# @app.delete("/reports/clear")
# async def clear_all_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     num_deleted = delete_user_reports(db=db, user_id=current_user.id)
#     return {"message": f"Successfully deleted {num_deleted} report(s)."}

# @app.post("/reports/upload", response_model=MedicalReportSchema)
# async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     if file.content_type != "application/pdf": raise HTTPException(status_code=400, detail="Invalid file type.")
#     try:
#         pdf_bytes = await file.read(); pdf_stream = io.BytesIO(pdf_bytes); reader = PdfReader(pdf_stream)
#         text = "".join(page.extract_text() + "\n" for page in reader.pages)
#         if not text.strip(): raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
#         return create_medical_report(db=db, user_id=current_user.id, filename=file.filename, text=text)
#     except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# @app.post("/reports/summarize", response_model=ReportSummaryOutput)
# async def summarize_medical_report(report_input: ReportTextInput, current_user: User = Depends(get_current_user)):
#     summary_json_string = lifespan_data["rag_service"].summarize_report(report_text=report_input.report_text)
#     try: return json.loads(summary_json_string)
#     except json.JSONDecodeError: raise HTTPException(status_code=500, detail="AI returned an invalid summary format.")

# @app.post("/chat/sessions", response_model=ChatSessionSchema)
# async def create_new_session(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return create_chat_session(db, user_id=current_user.id)

# @app.get("/chat/sessions/{session_id}", response_model=list[ChatMessageSchema])
# async def get_messages_for_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
#     if not session: raise HTTPException(status_code=404, detail="Session not found")
#     return get_session_messages(db, session_id=session_id)

# @app.post("/chat/sessions/{session_id}", response_model=ChatMessageSchema)
# async def post_message_to_session(session_id: int, chat_input: ChatInput, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
#     if not session: raise HTTPException(status_code=404, detail="Session not found")
#     create_chat_message(db, session_id=session_id, sender="user", text=chat_input.question)
#     rag_service = lifespan_data["rag_service"]
#     user_profile = UserSchema.model_validate(current_user).model_dump()
#     report_text = get_latest_user_report_text(db, user_id=current_user.id)
#     answer = rag_service.ask_question(question=chat_input.question, user_profile=user_profile, lifestyle_data=chat_input.lifestyle_data, report_text=report_text, language=chat_input.language)
#     ai_message = create_chat_message(db, session_id=session_id, sender="ai", text=answer)
#     return ai_message





# backend/main.py
# FINAL, SELF-CONTAINED, AND SYNTACTICALLY CORRECT VERSION

# import os
# import faiss
# import numpy as np
# import io
# import json
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# from pydantic_settings import BaseSettings
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
# from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean
# from sqlalchemy.orm import sessionmaker, Session
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
# from pydantic import BaseModel
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta
# from apscheduler.schedulers.asyncio import AsyncIOScheduler

# # --- 1. CONFIGURATION ---
# class Settings(BaseSettings):
#     google_api_key: str
#     class Config: env_file = ".env"
# settings = Settings()

# # --- 2. DATABASE SETUP ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "visuhealth.db")
# SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class User(Base): __tablename__ = "users"; id=Column(Integer, primary_key=True); email=Column(String, unique=True); hashed_password=Column(String); age=Column(Integer); weight_kg=Column(Float, nullable=True); height_cm=Column(Float, nullable=True); last_smoking_intensity=Column(Float, default=0.0); last_alcohol_intensity=Column(Float, default=0.0); last_diet_intensity=Column(Float, default=0.0)
# class ChatSession(Base): __tablename__ = "chat_sessions"; id=Column(Integer, primary_key=True); user_id=Column(Integer, ForeignKey("users.id")); created_at=Column(DateTime(timezone=True), server_default=func.now()); title=Column(String, default="New Conversation")
# class ChatMessage(Base): __tablename__ = "chat_messages"; id=Column(Integer, primary_key=True); session_id=Column(Integer, ForeignKey("chat_sessions.id")); sender=Column(String); text=Column(String); timestamp=Column(DateTime(timezone=True), server_default=func.now())
# class MedicalReport(Base): __tablename__ = "medical_reports"; id=Column(Integer, primary_key=True); user_id=Column(Integer, ForeignKey("users.id")); filename=Column(String); extracted_text=Column(String); uploaded_at=Column(DateTime(timezone=True), server_default=func.now())
# class Notification(Base): __tablename__ = "notifications"; id=Column(Integer, primary_key=True); user_id=Column(Integer, ForeignKey("users.id")); message=Column(String); is_read=Column(Boolean, default=False); created_at=Column(DateTime(timezone=True), server_default=func.now())
# class BiometricLog(Base): __tablename__ = "biometric_logs"; id=Column(Integer, primary_key=True); user_id=Column(Integer, ForeignKey("users.id")); metric_type=Column(String); value=Column(Float); logged_at=Column(DateTime(timezone=True), server_default=func.now())
# Base.metadata.create_all(bind=engine)
# --- 4. PYDANTIC SCHEMAS ---
# THIS IS THE CORRECTLY FORMATTED VERSION

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TokenData(BaseModel):
#     email: Optional[str] = None

# class UserBase(BaseModel):
#     email: str

# class UserCreate(UserBase):
#     password: str
#     age: int
#     weight_kg: Optional[float] = None
#     height_cm: Optional[float] = None

# class UserSchema(UserBase):
#     id: int
#     age: int
#     weight_kg: Optional[float] = None
#     height_cm: Optional[float] = None
#     class Config:
#         from_attributes = True

# class ChatMessageBase(BaseModel):
#     sender: str
#     text: str

# class ChatMessageSchema(ChatMessageBase):
#     id: int
#     timestamp: datetime
#     class Config:
#         from_attributes = True

# class ChatSessionSchema(BaseModel):
#     id: int
#     user_id: int
#     created_at: datetime
#     title: str
#     class Config:
#         from_attributes = True

# class MedicalReportSchema(BaseModel):
#     id: int
#     user_id: int
#     filename: str
#     uploaded_at: datetime
#     extracted_text: str
#     class Config:
#         from_attributes = True

# class ReportTextInput(BaseModel):
#     report_text: str

# class ReportSummaryOutput(BaseModel):
#     key_findings: str
#     what_it_means: str
#     next_steps: str

# class ChatInput(BaseModel):
#     question: str
#     lifestyle_data: dict
#     language: Optional[str] = "English"

# class LifestyleData(BaseModel):
#     smokingIntensity: float
#     alcoholIntensity: float
#     dietIntensity: float

# class NotificationSchema(BaseModel):
#     id: int
#     user_id: int
#     message: str
#     is_read: bool
#     created_at: datetime
#     class Config:
#         from_attributes = True

# class BiometricLogBase(BaseModel): metric_type: str; value: float
# class BiometricLogCreate(BiometricLogBase): pass
# class BiometricLogSchema(BiometricLogBase): id: int; user_id: int; logged_at: datetime;
# class Config: from_attributes = True        
# # --- 5. AUTHENTICATION LOGIC ---
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto"); SECRET_KEY = "a_very_secret_key_for_visuhealth"; ALGORITHM = "HS256"; ACCESS_TOKEN_EXPIRE_MINUTES = 30; oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# def verify_password(p, h): return pwd_context.verify(p, h)
# def get_password_hash(p): return pwd_context.hash(p)
# def create_access_token(data: dict): expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES); data.update({"exp": expire}); return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# # --- 6. CRUD ---
# def get_user_by_email(db: Session, email: str): return db.query(User).filter(User.email == email).first()
# def create_chat_session(db: Session, user_id: int, title: str = "New Conversation"): s = ChatSession(user_id=user_id, title=title); db.add(s); db.commit(); db.refresh(s); return s
# def create_chat_message(db: Session, session_id: int, sender: str, text: str): m = ChatMessage(session_id=session_id, sender=sender, text=text); db.add(m); db.commit(); db.refresh(m); return m
# def create_medical_report(db: Session, user_id: int, filename: str, text: str): r = MedicalReport(user_id=user_id, filename=filename, extracted_text=text); db.add(r); db.commit(); db.refresh(r); return r
# def get_latest_user_report_text(db: Session, user_id: int): r = db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).first(); return r.extracted_text if r else None
# def get_user_reports(db: Session, user_id: int): return db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).all()
# def delete_user_reports(db: Session, user_id: int): reports_to_delete = db.query(MedicalReport).filter(MedicalReport.user_id == user_id); num_deleted = reports_to_delete.count(); reports_to_delete.delete(); db.commit(); return num_deleted
# def get_user_chat_sessions(db: Session, user_id: int): return db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
# def get_session_messages(db: Session, session_id: int): return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
# def create_notification(db: Session, user_id: int, message: str): n = Notification(user_id=user_id, message=message); db.add(n); db.commit()
# def create_biometric_log(db: Session, user_id: int, log: BiometricLogCreate): db_log = BiometricLog(user_id=user_id, metric_type=log.metric_type, value=log.value); db.add(db_log); db.commit(); db.refresh(db_log); return db_log
# def get_biometric_logs_for_user(db: Session, user_id: int, metric_type: str): return db.query(BiometricLog).filter(BiometricLog.user_id == user_id, BiometricLog.metric_type == metric_type).order_by(BiometricLog.logged_at).all()
# # --- 7. RAG SERVICE ---
# KNOWLEDGE_BASE = { "liver_alcohol": "...", "heart_diet": "...", "lungs_smoking": "...", "general_health": "..."}
# class RagService:
#     def __init__(self): print("Initializing RAG Service..."); genai.configure(api_key=settings.google_api_key); self.model = genai.GenerativeModel('gemini-1.5-flash-latest'); self.encoder = SentenceTransformer('all-MiniLM-L6-v2'); self.documents = list(KNOWLEDGE_BASE.values()); self.doc_embeddings = self.encoder.encode(self.documents); self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1]); self.index.add(self.doc_embeddings.astype(np.float32)); print("RAG Service Initialized Successfully.")
#     def _retrieve_documents(self, q: str, k: int=2): qe = self.encoder.encode([q]).astype(np.float32); _, i = self.index.search(qe, k); return [self.documents[idx] for idx in i[0]]
#     def ask_question(self, question: str, user_profile: dict, lifestyle_data: dict, report_text: str | None, language: str):
#         gc = "\n\n".join(self._retrieve_documents(question)); pp = ["..."]; pp.extend([f"User's Question: \"{question}\"", "...", f"IMPORTANT: Your response MUST be in {language}."]); p = "\n".join(pp)
#         try: return self.model.generate_content(p).text
#         except Exception as e: print(f"Error calling Gemini API: {e}"); return "AI service error."
#     def summarize_report(self, report_text: str):
#         p = f"""You are a medical data analyst... Report: {report_text} ... Generate JSON summary."""; config = genai.types.GenerationConfig(response_mime_type="application/json")
#         try: return self.model.generate_content(p, generation_config=config).text
#         except Exception as e: print(f"Error calling Gemini for summary: {e}"); return '{"key_findings": "Error.", "what_it_means": "AI service failed.", "next_steps": "Try again."}'

# # --- 8. SCHEDULER & BACKGROUND JOB ---
# async def proactive_analysis_job():
#     print("SCHEDULER: Running proactive analysis job..."); db = SessionLocal()
#     try:
#         rag_service: RagService = app.state.rag_service
#         users = db.query(User).all()
#         for user in users:
#             report_text = get_latest_user_report_text(db, user_id=user.id)
#             if not report_text: continue
#             prompt = f"You are a proactive AI health analyst... Report: {report_text} ... Analyze."
#             try:
#                 insight = rag_service.model.generate_content(prompt).text.strip()
#                 if insight and insight != "NO_INSIGHT": print(f"SCHEDULER: Found insight for {user.email}: {insight}"); create_notification(db, user_id=user.id, message=insight)
#             except Exception as e: print(f"SCHEDULER: AI error for {user.email}: {e}")
#     finally: db.close()
# scheduler = AsyncIOScheduler(); scheduler.add_job(proactive_analysis_job, 'interval', minutes=1, id='proactive_analysis_job')

# # --- 9. FASTAPI APP & LIFESPAN ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server starting up..."); app.state.rag_service = RagService(); scheduler.start(); yield; print("Server shutting down..."); scheduler.shutdown()
# app = FastAPI(lifespan=lifespan)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# # --- 10. API DEPENDENCIES & ENDPOINTS ---
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             raise exception
#     except JWTError:
#         raise exception
#     user = get_user_by_email(db, email=email)
#     if user is None:
#         raise exception
#     return user

# @app.post("/register/", response_model=UserSchema)
# def register_user(user: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user_by_email(db, email=user.email);
#     if db_user: raise HTTPException(status_code=400, detail="Email already registered")
#     hp = get_password_hash(user.password); new_user = User(email=user.email, hashed_password=hp, age=user.age, weight_kg=user.weight_kg, height_cm=user.height_cm)
#     db.add(new_user); db.commit(); db.refresh(new_user); return new_user

# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = get_user_by_email(db, email=form_data.username)
#     if not user or not verify_password(form_data.password, user.hashed_password): raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
#     at = create_access_token(data={"sub": user.email}); return {"access_token": at, "token_type": "bearer"}

# @app.get("/users/me/", response_model=UserSchema)
# async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

# @app.post("/users/me/lifestyle")
# async def update_lifestyle_data(lifestyle_data: LifestyleData, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     current_user.last_smoking_intensity=lifestyle_data.smokingIntensity; current_user.last_alcohol_intensity=lifestyle_data.alcoholIntensity; current_user.last_diet_intensity=lifestyle_data.dietIntensity
#     db.commit(); return {"message": "Lifestyle data updated."}

# @app.post("/logbook", response_model=BiometricLogSchema)
# async def log_biometric_data(log_data: BiometricLogCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return create_biometric_log(db=db, user_id=current_user.id, log=log_data)

# @app.get("/logbook/{metric_type}", response_model=list[BiometricLogSchema])
# async def get_logbook_data(metric_type: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return get_biometric_logs_for_user(db=db, user_id=current_user.id, metric_type=metric_type)

# @app.get("/notifications", response_model=list[NotificationSchema])
# async def get_notifications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return db.query(Notification).filter(Notification.user_id == current_user.id).order_by(Notification.created_at.desc()).all()

# @app.post("/notifications/{notification_id}/read")
# async def mark_notification_as_read(notification_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     notif = db.query(Notification).filter(Notification.id == notification_id, Notification.user_id == current_user.id).first()
#     if not notif: raise HTTPException(status_code=404, detail="Notification not found")
#     notif.is_read = True; db.commit(); return {"message": "Notification marked as read."}

# @app.get("/reports", response_model=list[MedicalReportSchema])
# async def get_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return get_user_reports(db=db, user_id=current_user.id)

# @app.delete("/reports/clear")
# async def clear_all_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     num_deleted = delete_user_reports(db=db, user_id=current_user.id)
#     return {"message": f"Successfully deleted {num_deleted} report(s)."}

# @app.post("/reports/upload", response_model=MedicalReportSchema)
# async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     if file.content_type != "application/pdf": raise HTTPException(status_code=400, detail="Invalid file type.")
#     try:
#         pdf_bytes = await file.read(); pdf_stream = io.BytesIO(pdf_bytes); reader = PdfReader(pdf_stream)
#         text = "".join(page.extract_text() + "\n" for page in reader.pages)
#         if not text.strip(): raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
#         return create_medical_report(db=db, user_id=current_user.id, filename=file.filename, text=text)
#     except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# @app.post("/reports/summarize", response_model=ReportSummaryOutput)
# async def summarize_medical_report(report_input: ReportTextInput, current_user: User = Depends(get_current_user)):
#     summary_json_string = app.state.rag_service.summarize_report(report_text=report_input.report_text)
#     try: return json.loads(summary_json_string)
#     except json.JSONDecodeError: raise HTTPException(status_code=500, detail="AI returned an invalid summary format.")

# @app.post("/chat/sessions", response_model=ChatSessionSchema)
# async def create_new_session(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     return create_chat_session(db, user_id=current_user.id)

# @app.get("/chat/sessions/{session_id}", response_model=list[ChatMessageSchema])
# async def get_messages_for_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first();
#     if not session: raise HTTPException(status_code=404, detail="Session not found")
#     return get_session_messages(db, session_id=session_id)

# @app.post("/chat/sessions/{session_id}", response_model=ChatMessageSchema)
# async def post_message_to_session(session_id: int, chat_input: ChatInput, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
#     if not session: raise HTTPException(status_code=404, detail="Session not found")
#     create_chat_message(db, session_id=session_id, sender="user", text=chat_input.question)
#     rag_service = app.state.rag_service
#     user_profile = UserSchema.model_validate(current_user).model_dump()
#     report_text = get_latest_user_report_text(db, user_id=current_user.id)
#     answer = rag_service.ask_question(question=chat_input.question, user_profile=user_profile, lifestyle_data=chat_input.lifestyle_data, report_text=report_text, language=chat_input.language)
#     ai_message = create_chat_message(db, session_id=session_id, sender="ai", text=answer)



#     return ai_message# backend/main.py
# # FINAL, DEFINITIVE VERSION WITH ALL SYNTAX AND INDENTATION CORRECTED

import os
import faiss
import numpy as np
import io
import json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- 1. CONFIGURATION ---
class Settings(BaseSettings):
    google_api_key: str
    class Config: env_file = ".env"
settings = Settings()

# --- 2. DATABASE SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)); DB_PATH = os.path.join(BASE_DIR, "visuhealth.db"); SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine); Base = declarative_base()

# --- 3. DATABASE MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    age = Column(Integer)
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    last_smoking_intensity = Column(Float, default=0.0)
    last_alcohol_intensity = Column(Float, default=0.0)
    last_diet_intensity = Column(Float, default=0.0)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    title = Column(String, default="New Conversation")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    sender = Column(String)
    text = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class MedicalReport(Base):
    __tablename__ = "medical_reports"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    extracted_text = Column(String)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(String)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class BiometricLog(Base):
    __tablename__ = "biometric_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    metric_type = Column(String) # e.g., "weight", "blood_pressure_systolic"
    value = Column(Float)
    logged_at = Column(DateTime(timezone=True), server_default=func.now())

Base.metadata.create_all(bind=engine)

# --- 4. PYDANTIC SCHEMAS ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    age: int
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

class UserSchema(UserBase):
    id: int
    age: int
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    class Config: from_attributes = True

class ChatMessageBase(BaseModel):
    sender: str
    text: str

class ChatMessageSchema(ChatMessageBase):
    id: int
    timestamp: datetime
    class Config: from_attributes = True

class ChatSessionSchema(BaseModel):
    id: int
    user_id: int
    created_at: datetime
    title: str
    class Config: from_attributes = True

class MedicalReportSchema(BaseModel):
    id: int
    user_id: int
    filename: str
    uploaded_at: datetime
    extracted_text: str
    class Config: from_attributes = True

class ReportTextInput(BaseModel):
    report_text: str

class ReportSummaryOutput(BaseModel):
    key_findings: str
    what_it_means: str
    next_steps: str

class ChatInput(BaseModel):
    question: str
    lifestyle_data: dict
    language: Optional[str] = "English"

class LifestyleData(BaseModel):
    smokingIntensity: float
    alcoholIntensity: float
    dietIntensity: float

class NotificationSchema(BaseModel):
    id: int
    user_id: int
    message: str
    is_read: bool
    created_at: datetime
    class Config: from_attributes = True


class BiometricLogBase(BaseModel):
    metric_type: str
    value: float

class BiometricLogCreate(BiometricLogBase):
    pass

class BiometricLogSchema(BiometricLogBase):
    id: int
    user_id: int
    logged_at: datetime
    class Config:
        from_attributes = True

# --- 5. AUTHENTICATION LOGIC ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "a_very_secret_key_for_visuhealth"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- 6. CRUD ---
def get_user_by_email(db: Session, email: str): return db.query(User).filter(User.email == email).first()
def create_chat_session(db: Session, user_id: int, title: str = "New Conversation"): s = ChatSession(user_id=user_id, title=title); db.add(s); db.commit(); db.refresh(s); return s
def create_chat_message(db: Session, session_id: int, sender: str, text: str): m = ChatMessage(session_id=session_id, sender=sender, text=text); db.add(m); db.commit(); db.refresh(m); return m
def create_medical_report(db: Session, user_id: int, filename: str, text: str): r = MedicalReport(user_id=user_id, filename=filename, extracted_text=text); db.add(r); db.commit(); db.refresh(r); return r
def get_latest_user_report_text(db: Session, user_id: int): r = db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).first(); return r.extracted_text if r else None
def get_user_reports(db: Session, user_id: int): return db.query(MedicalReport).filter(MedicalReport.user_id == user_id).order_by(MedicalReport.uploaded_at.desc()).all()
def delete_user_reports(db: Session, user_id: int): reports_to_delete = db.query(MedicalReport).filter(MedicalReport.user_id == user_id); num_deleted = reports_to_delete.count(); reports_to_delete.delete(); db.commit(); return num_deleted
def get_user_chat_sessions(db: Session, user_id: int): return db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
def get_session_messages(db: Session, session_id: int): return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp).all()
def create_notification(db: Session, user_id: int, message: str): n = Notification(user_id=user_id, message=message); db.add(n); db.commit()


def create_biometric_log(db: Session, user_id: int, log: BiometricLogCreate):
    db_log = BiometricLog(user_id=user_id, metric_type=log.metric_type, value=log.value)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_biometric_logs_for_user(db: Session, user_id: int, metric_type: str):
    return db.query(BiometricLog).filter(
        BiometricLog.user_id == user_id,
        BiometricLog.metric_type == metric_type
    ).order_by(BiometricLog.logged_at).all()


# --- 7. RAG SERVICE ---
KNOWLEDGE_BASE = { "liver_alcohol": "...", "heart_diet": "...", "lungs_smoking": "...", "general_health": "..."}
class RagService:
    def __init__(self): print("Initializing RAG Service..."); genai.configure(api_key=settings.google_api_key); self.model = genai.GenerativeModel('gemini-1.5-flash-latest'); self.encoder = SentenceTransformer('all-MiniLM-L6-v2'); self.documents = list(KNOWLEDGE_BASE.values()); self.doc_embeddings = self.encoder.encode(self.documents); self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1]); self.index.add(self.doc_embeddings.astype(np.float32)); print("RAG Service Initialized Successfully.")
    def _retrieve_documents(self, q: str, k: int=2): qe = self.encoder.encode([q]).astype(np.float32); _, i = self.index.search(qe, k); return [self.documents[idx] for idx in i[0]]
    def ask_question(self, question: str, user_profile: dict, lifestyle_data: dict, report_text: str | None, language: str):
        gc = "\n\n".join(self._retrieve_documents(question)); pp = ["..."]; pp.extend([f"User's Question: \"{question}\"", "...", f"IMPORTANT: Your response MUST be in {language}."]); p = "\n".join(pp)
        try: return self.model.generate_content(p).text
        except Exception as e: print(f"Error calling Gemini API: {e}"); return "AI service error."
    # backend/main.py -> inside the RagService class

    def summarize_report(self, report_text: str):
        # --- THE NEW, STRICTER PROMPT ---
        prompt = f"""
        You are an AI data extraction service. Your ONLY job is to read the provided medical lab report text
        and convert it into a simple, valid JSON object. Do not add any extra text, explanations, or markdown formatting.
        The JSON object must have exactly three string keys: "key_findings", "what_it_means", and "next_steps".

        - "key_findings": Create a single string containing a markdown bulleted list ('* ') of the most important results, especially those out of the normal range.
        - "what_it_means": Create a single string with a simple, easy-to-understand explanation of what these findings suggest about the patient's health.
        - "next_steps": Create a single string containing a short, markdown bulleted list ('* ') of potential topics the user could discuss with their doctor. Do not give medical advice.

        Here is the medical report text to analyze:
        ---
        {report_text}
        ---
        
        Now, generate ONLY the raw JSON object and nothing else.
        """
        try:
            config = genai.types.GenerationConfig(response_mime_type="application/json")
            response = self.model.generate_content(prompt, generation_config=config)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini for summary: {e}")
            return '{"key_findings": "* Error: Could not analyze the report.", "what_it_means": "The AI service encountered an issue. This can happen with complex PDFs or during high service load.", "next_steps": "* Please try uploading the report again later."}'
# --- 8. SCHEDULER & BACKGROUND JOB ---
async def proactive_analysis_job():
    print("SCHEDULER: Running proactive analysis job..."); db = SessionLocal()
    try:
        rag_service: RagService = app.state.rag_service; users = db.query(User).all()
        for user in users:
            report_text = get_latest_user_report_text(db, user_id=user.id)
            if not report_text: continue
            prompt = f"You are a proactive AI health analyst... Report: {report_text} ... Analyze."
            try:
                insight = rag_service.model.generate_content(prompt).text.strip()
                if insight and insight != "NO_INSIGHT": print(f"SCHEDULER: Found insight for {user.email}: {insight}"); create_notification(db, user_id=user.id, message=insight)
            except Exception as e: print(f"SCHEDULER: AI error for {user.email}: {e}")
    finally: db.close()
scheduler = AsyncIOScheduler(); scheduler.add_job(proactive_analysis_job, 'interval', minutes=1, id='proactive_analysis_job')

# --- 9. FASTAPI APP & LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up..."); app.state.rag_service = RagService(); scheduler.start(); yield; print("Server shutting down..."); scheduler.shutdown()
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 10. API DEPENDENCIES & ENDPOINTS ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise exception
    except JWTError:
        raise exception
    user = get_user_by_email(db, email=email)
    if user is None:
        raise exception
    return user

@app.post("/register/", response_model=UserSchema)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user: raise HTTPException(status_code=400, detail="Email already registered")
    hp = get_password_hash(user.password); new_user = User(email=user.email, hashed_password=hp, age=user.age, weight_kg=user.weight_kg, height_cm=user.height_cm)
    db.add(new_user); db.commit(); db.refresh(new_user); return new_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password): raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    at = create_access_token(data={"sub": user.email}); return {"access_token": at, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

@app.post("/users/me/lifestyle")
async def update_lifestyle_data(lifestyle_data: LifestyleData, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    current_user.last_smoking_intensity=lifestyle_data.smokingIntensity; current_user.last_alcohol_intensity=lifestyle_data.alcoholIntensity; current_user.last_diet_intensity=lifestyle_data.dietIntensity
    db.commit(); return {"message": "Lifestyle data updated."}


@app.post("/logbook", response_model=BiometricLogSchema)
async def log_biometric_data(log_data: BiometricLogCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_biometric_log(db=db, user_id=current_user.id, log=log_data)

@app.get("/logbook/{metric_type}", response_model=list[BiometricLogSchema])
async def get_logbook_data(metric_type: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_biometric_logs_for_user(db=db, user_id=current_user.id, metric_type=metric_type)

@app.get("/notifications", response_model=list[NotificationSchema])
async def get_notifications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Notification).filter(Notification.user_id == current_user.id).order_by(Notification.created_at.desc()).all()

@app.post("/notifications/{notification_id}/read")
async def mark_notification_as_read(notification_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    notif = db.query(Notification).filter(Notification.id == notification_id, Notification.user_id == current_user.id).first()
    if not notif: raise HTTPException(status_code=404, detail="Notification not found")
    notif.is_read = True; db.commit(); return {"message": "Notification marked as read."}


@app.get("/reports", response_model=list[MedicalReportSchema])
async def get_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_user_reports(db=db, user_id=current_user.id)

@app.delete("/reports/clear")
async def clear_all_reports_for_user(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    num_deleted = delete_user_reports(db=db, user_id=current_user.id)
    return {"message": f"Successfully deleted {num_deleted} report(s)."}

@app.post("/reports/upload", response_model=MedicalReportSchema)
async def upload_report(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if file.content_type != "application/pdf": raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        pdf_bytes = await file.read(); pdf_stream = io.BytesIO(pdf_bytes); reader = PdfReader(pdf_stream)
        text = "".join(page.extract_text() + "\n" for page in reader.pages)
        if not text.strip(): raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        return create_medical_report(db=db, user_id=current_user.id, filename=file.filename, text=text)
    except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/reports/summarize", response_model=ReportSummaryOutput)
async def summarize_medical_report(report_input: ReportTextInput, current_user: User = Depends(get_current_user)):
    summary_json_string = app.state.rag_service.summarize_report(report_text=report_input.report_text)
    try: return json.loads(summary_json_string)
    except json.JSONDecodeError: raise HTTPException(status_code=500, detail="AI returned an invalid summary format.")

@app.post("/chat/sessions", response_model=ChatSessionSchema)
async def create_new_session(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_chat_session(db, user_id=current_user.id)

@app.get("/chat/sessions/{session_id}", response_model=list[ChatMessageSchema])
async def get_messages_for_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first();
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    return get_session_messages(db, session_id=session_id)

@app.post("/chat/sessions/{session_id}", response_model=ChatMessageSchema)
async def post_message_to_session(session_id: int, chat_input: ChatInput, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    create_chat_message(db, session_id=session_id, sender="user", text=chat_input.question)
    rag_service = app.state.rag_service
    user_profile = UserSchema.model_validate(current_user).model_dump()
    report_text = get_latest_user_report_text(db, user_id=current_user.id)
    answer = rag_service.ask_question(question=chat_input.question, user_profile=user_profile, lifestyle_data=chat_input.lifestyle_data, report_text=report_text, language=chat_input.language)
    ai_message = create_chat_message(db, session_id=session_id, sender="ai", text=answer)
    return ai_message