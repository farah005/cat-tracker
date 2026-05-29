from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.orm import Chat
from app.models.schemas import ChatCreate, ChatOut

router = APIRouter(prefix="/cats", tags=["cats"])


@router.get("/", response_model=List[ChatOut])
def list_cats(db: Session = Depends(get_db)):
    return db.query(Chat).order_by(Chat.id).all()


@router.get("/{chat_id}", response_model=ChatOut)
def get_cat(chat_id: int, db: Session = Depends(get_db)):
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Cat {chat_id} not found")
    return cat


@router.post("/", response_model=ChatOut, status_code=201)
def create_cat(payload: ChatCreate, db: Session = Depends(get_db)):
    cat = Chat(**payload.model_dump())
    db.add(cat)
    db.commit()
    db.refresh(cat)
    return cat


@router.delete("/{chat_id}", status_code=204)
def delete_cat(chat_id: int, db: Session = Depends(get_db)):
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Cat {chat_id} not found")
    db.delete(cat)
    db.commit()
