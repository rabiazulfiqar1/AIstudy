import sqlalchemy
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, TEXT
from sqlalchemy import func, Column, CheckConstraint, UniqueConstraint, Index

metadata = sqlalchemy.MetaData()

# -------------------------------
# Video Cache Table
# -------------------------------
video_cache = sqlalchemy.Table(
    "video_cache",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=sqlalchemy.text("gen_random_uuid()")),
    Column("cache_key", sqlalchemy.String(64), nullable=False),
    Column("cache_type", sqlalchemy.String(20), nullable=False),
    Column("data", JSONB, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("last_accessed", TIMESTAMP(timezone=True), server_default=func.now()),
    
    UniqueConstraint("cache_key", "cache_type", name="unique_cache_entry")
)

# -------------------------------
# Users Table
# -------------------------------
users = sqlalchemy.Table(
    "users",
    metadata,
    Column("user_id", UUID(as_uuid=True), primary_key=True, server_default=sqlalchemy.text("gen_random_uuid()")),
    Column("username", TEXT, nullable=False, unique=True),
    Column("full_name", TEXT, nullable=False),
    Column("email", TEXT, nullable=False, unique=True),
    Column("organization", TEXT),
    Column("field_of_study", TEXT),
    Column("phone", TEXT),
    Column("profile_pic", TEXT),
    Column("status", TEXT, nullable=False, server_default="active"),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
    
    CheckConstraint("status IN ('active', 'inactive')", name="check_users_status")
)

# Partial unique index on phone
Index(
    "unique_phone_not_null",
    users.c.phone,
    unique=True,
    postgresql_where=users.c.phone.isnot(None)
)
