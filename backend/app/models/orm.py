from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from app.database import Base


class Chat(Base):
    __tablename__ = "chats"

    id         = Column(Integer, primary_key=True, index=True)
    nom        = Column(String(100), nullable=False)
    race       = Column(String(100))
    couleur    = Column(String(50))
    poids_kg   = Column(Float)
    lat_home   = Column(Float, nullable=False, default=48.8566)
    lon_home   = Column(Float, nullable=False, default=2.3522)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    positions = relationship("Position", back_populates="chat", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"

    id               = Column(BigInteger, primary_key=True, index=True)
    chat_id          = Column(Integer, ForeignKey("chats.id"), nullable=False)
    ts               = Column(DateTime(timezone=True), nullable=False)
    latitude         = Column(Float, nullable=False)
    longitude        = Column(Float, nullable=False)
    geom             = Column(Geometry("POINT", srid=4326))
    vitesse_ms       = Column(Float)
    distance_home_m  = Column(Float)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    chat = relationship("Chat", back_populates="positions")
