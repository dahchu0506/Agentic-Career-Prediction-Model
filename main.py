print(" MAIN.PY LOADED SUCCESSFULLY ")
from sqlalchemy import func, case
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext

from .llm_agent import generate_career_reasoning
from .database import SessionLocal
from .models import Feedback, Prediction, User

from sqlalchemy import func
from collections import Counter

# ===============================
# LOAD MODEL ARTIFACTS
# ===============================
model = joblib.load("model/career_model.pkl")
mlb = joblib.load("model/mlb.pkl")
le_target = joblib.load("model/target_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

EXPECTED_FEATURES = model.feature_names_in_

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(
    title="Career Recommendation API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
        ],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SAFE ENCODING MAPS
# ===============================
UG_COURSE_MAP = {
    "b.tech": 0,
    "b.sc": 1,
    "bca": 2,
    "b.com": 3
}

UG_SPEC_MAP = {
    "computer science": 0,
    "it": 1,
    "electronics": 2,
    "mechanical": 3,
    "mathematics": 4
}

# ===============================
# REQUEST MODELS
# ===============================
class CareerInput(BaseModel):
    user_id: int
    age: int
    ug_course: str
    ug_specialization: str
    cgpa: float
    skills: List[str]
    interests: List[str]
    certifications: List[str]
    experience_years: int
    working_status: int


class CareerPrediction(BaseModel):
    career: str
    probability: float


class PredictionResponse(BaseModel):
    prediction_id: int
    top_1: CareerPrediction
    top_3: List[CareerPrediction]
    reasoning: str


class FeedbackInput(BaseModel):
    prediction_id: int
    selected_career: str
    satisfied: bool
    comments: str | None = None

class UserInput(BaseModel):
    age: int
    ug_course: str
    ug_specialization: str
    cgpa: float
    experience_years: int
    working_status: int
class ChatInput(BaseModel):
    user_id: int
    message: str

class RegisterInput(BaseModel):
    username: str
    password: str

class LoginInput(BaseModel):
    username: str
    password: str

class ChatResponse(BaseModel):
    reply: str

class SignupInput(BaseModel):
    username: str
    password: str
    age: int
    ug_course: str
    ug_specialization: str
    cgpa: float
    experience_years: int
    working_status: int


class LoginInput(BaseModel):
    username: str
    password: str

# ===============================
# PREPROCESS FUNCTION
# ===============================
def preprocess_input(user: CareerInput):
    ug_course = UG_COURSE_MAP.get(user.ug_course.strip().lower())
    ug_spec = UG_SPEC_MAP.get(user.ug_specialization.strip().lower())

    if ug_course is None or ug_spec is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid UG course or specialization"
        )

    base_df = pd.DataFrame([{
        "age": user.age,
        "ug_course": ug_course,
        "ug_specialization": ug_spec,
        "cgpa": user.cgpa,
        "experience_years": user.experience_years,
        "working_status": user.working_status
    }])

    base_df["cgpa"] = scaler.transform(base_df[["cgpa"]])

    combined = [
        s.strip().lower()
        for s in (user.skills + user.interests + user.certifications)
        if s.strip().lower() in mlb.classes_
    ]
    if not combined:
        raise HTTPException(
            status_code=400,
            detail="None of the provided skills/interests are recognized by the model."
        )

    multi_hot = mlb.transform([combined])
    multi_df = pd.DataFrame(multi_hot, columns=mlb.classes_)

    X = pd.concat([base_df, multi_df], axis=1)

    # 🔒 Ensure feature order matches training
    X = X.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    return X


# ===============================
# FEEDBACK AGGREGATION (AGENT MEMORY)
# ===============================
def get_feedback_adjustments(db):
    feedback_stats = (
        db.query(
            Feedback.selected_career,
            func.avg(
                case(
                    (Feedback.satisfied == True, 1),
                    else_=-1
                )
            ).label("score")
        )
        .group_by(Feedback.selected_career)
        .all()
    )

    return {career: score for career, score in feedback_stats}

def get_user_feedback_history(db, user_id: int):
    return (
        db.query(Feedback)
        .join(Prediction, Feedback.prediction_id == Prediction.prediction_id)
        .filter(Prediction.user_id == user_id)
        .all()
    )

def get_user_preference_profile(db, user_id: int):
    feedbacks = get_user_feedback_history(db, user_id)

    liked = [
        f.selected_career
        for f in feedbacks
        if f.satisfied
    ]

    disliked = [
        f.selected_career
        for f in feedbacks
        if not f.satisfied
    ]

    return {
        "liked_careers": Counter(liked),
        "disliked_careers": Counter(disliked)
    }

def get_user_prediction_history(db, user_id: int, limit: int = 5):
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

def build_user_memory_context(db, user_id: int):
    predictions = get_user_prediction_history(db, user_id, limit=3)
    preferences = get_user_preference_profile(db, user_id)

    memory_lines = []

    if predictions:
        memory_lines.append("Recent career recommendations:")
        for p in predictions:
            memory_lines.append(
                f"- {p.top_1_career} (confidence {p.top_1_probability:.2f})"
            )

    if preferences["liked_careers"]:
        memory_lines.append("\nCareers the user liked:")
        for career, count in preferences["liked_careers"].items():
            memory_lines.append(f"- {career} ({count} times)")

    if preferences["disliked_careers"]:
        memory_lines.append("\nCareers the user disliked:")
        for career, count in preferences["disliked_careers"].items():
            memory_lines.append(f"- {career} ({count} times)")

    return "\n".join(memory_lines) if memory_lines else "No prior history available."

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/register")
def register_user(user: UserInput):
    db = SessionLocal()
    try:
        new_user = User(
            age=user.age,
            ug_course=user.ug_course,
            ug_specialization=user.ug_specialization,
            cgpa=user.cgpa,
            experience_years=user.experience_years,
            working_status=user.working_status
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "User registered successfully",
            "user_id": new_user.user_id
        }

    finally:
        db.close()

@app.post("/auth/signup")
def signup(user: SignupInput):

    db = SessionLocal()

    existing = db.query(User).filter(
        User.username == user.username
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    hashed_pw = hash_password(user.password)

    new_user = User(
        username=user.username,
        password_hash=hashed_pw,
        age=user.age,
        ug_course=user.ug_course,
        ug_specialization=user.ug_specialization,
        cgpa=user.cgpa,
        experience_years=user.experience_years,
        working_status=user.working_status
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User created successfully",
        "user_id": new_user.user_id
    }

@app.post("/auth/register")

def register_user_auth(user: RegisterInput):

    db = SessionLocal()

    hashed_pw = hash_password(user.password)

    new_user = User(
        username=user.username,
        password_hash=hashed_pw
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"user_id": new_user.user_id}

@app.post("/auth/login")
def login(user: LoginInput):

    db = SessionLocal()

    db_user = db.query(User).filter(
        User.username == user.username
    ).first()

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username"
        )

    if not verify_password(user.password, db_user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid password"
        )

    return {
        "message": "Login successful",
        "user_id": db_user.user_id
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_career(user_input: CareerInput):

    db = SessionLocal()

    try:
        X = preprocess_input(user_input)
        probs = model.predict_proba(X)[0]

        feedback_adjustments = get_feedback_adjustments(db)

        user = db.query(User).filter(User.user_id == user_input.user_id).first()
        if not user:
            raise HTTPException(status_code=400, detail="Invalid user id")

        ALPHA = 0.05
        top_idx = np.argsort(probs)[::-1][:3]

        results = []
        for i in top_idx:
            career = le_target.inverse_transform([i])[0]
            base_prob = float(probs[i])

            adjustment = float(feedback_adjustments.get(career, 0.0))
            final_prob = base_prob + ALPHA * adjustment
            final_prob = max(0.0, min(final_prob, 1.0))

            results.append(
                CareerPrediction(
                    career=career,
                    probability=round(final_prob, 3)
                )
            )

        explanation = generate_career_reasoning(
            skills=user_input.skills,
            interests=user_input.interests,
            cgpa=user_input.cgpa,
            experience_years=user_input.experience_years,
            top_1=results[0].career,
            top_1_prob=results[0].probability,
            top_3=[r.dict() for r in results]
        )

        prediction = Prediction(
            user_id=user_input.user_id,
            top_1_career=results[0].career,
            top_1_probability=results[0].probability,
            top_3=[r.dict() for r in results]
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        return {
            "prediction_id": prediction.prediction_id,
            "top_1": results[0],
            "top_3": results,
            "reasoning": explanation
        }

    except HTTPException:
        raise

    except Exception as e:
        print("PREDICT ERROR:", e)   # very important for debugging
        raise HTTPException(status_code=500, detail=str(e))



# ===============================
# FEEDBACK ENDPOINT
# ===============================
@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    db = SessionLocal()
    
    try:
        pred=db.query(Prediction).filter(Prediction.prediction_id == feedback.prediction_id).first()
        if not pred:
            raise HTTPException(status_code=400, detail="Prediction ID not found")
        fb = Feedback(
            prediction_id=feedback.prediction_id,
            selected_career=feedback.selected_career,
            satisfied=feedback.satisfied,
            comments=feedback.comments
        )

        db.add(fb)
        db.commit()
        db.refresh(fb)

        return {
            "message": "Feedback stored successfully",
            "feedback_id": fb.feedback_id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()
pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

@app.post("/chat", response_model=ChatResponse)
def chat_with_user(chat: ChatInput):
    db = SessionLocal()
    try:
        # Validate user
        from .models import User
        user = db.query(User).filter(User.user_id == chat.user_id).first()
        if not user:
            raise HTTPException(status_code=400, detail="Invalid user_id")

        # Build memory context
        memory_context = build_user_memory_context(db, chat.user_id)

        # Construct LLM prompt
        prompt = f"""
You are a professional career advisor AI.

USER MEMORY:
{memory_context}

USER QUESTION:
{chat.message}

Respond with personalized, practical career advice.
Avoid repeating past dislikes.
Build upon careers the user liked.
"""

        reply = generate_career_reasoning(
            skills=[],
            interests=[],
            cgpa=user.cgpa,
            experience_years=user.experience_years,
            top_1="",
            top_1_prob=0.0,
            top_3=[],
            custom_prompt=prompt   # 🔥 small modification
        )

        return {"reply": reply}

    finally:
        db.close()


@app.get("/user/{user_id}/history")
def get_user_history(user_id: int):
    db = SessionLocal()
    try:
        predictions = get_user_prediction_history(db, user_id)
        preferences = get_user_preference_profile(db, user_id)

        return {
            "recent_predictions": [
                {
                    "prediction_id": p.prediction_id,
                    "top_1_career": p.top_1_career,
                    "top_1_probability": p.top_1_probability,
                    "created_at": p.created_at
                }
                for p in predictions
            ],
            "preferences": preferences
        }

    finally:
        db.close()


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health():
    return {"status": "API running"}
