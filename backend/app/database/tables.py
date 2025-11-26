import sqlalchemy
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, TEXT, ARRAY
from sqlalchemy import func, Column, CheckConstraint, UniqueConstraint, Index, ForeignKey
from pgvector.sqlalchemy import Vector

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
    Column("user_id", UUID(as_uuid=True), primary_key=True),
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

# ============================================
# USER PROFILES TABLE
# ============================================

user_profiles = sqlalchemy.Table(
    "user_profiles",
    metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("skill_level", TEXT, nullable=False),
    Column("interests", ARRAY(TEXT)),  # Array of interests: ['web-dev', 'ml', 'data-science']
    Column("bio", TEXT),
    Column("github_username", TEXT),
    Column("preferred_project_types", ARRAY(TEXT)),  # ['open-source', 'research', 'personal']
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
    
    CheckConstraint("skill_level IN ('beginner', 'intermediate', 'advanced')", name="check_skill_level")
)

Index("idx_user_profiles_skill_level", user_profiles.c.skill_level)

# ============================================
# SKILLS TABLE
# ============================================

skills = sqlalchemy.Table(
    "skills",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("name", TEXT, nullable=False, unique=True),
    Column("category", TEXT, nullable=False),  # 'language', 'framework', 'tool', 'domain'
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
)

Index("idx_skills_name", skills.c.name)
Index("idx_skills_category", skills.c.category)

# ============================================
# USER SKILLS (Junction Table)
# ============================================

user_skills = sqlalchemy.Table(
    "user_skills",
    metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("skill_id", sqlalchemy.Integer, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True),
    Column("proficiency", sqlalchemy.Integer, nullable=False),  # 1-5 scale
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    
    CheckConstraint("proficiency BETWEEN 1 AND 5", name="check_proficiency_range")
)

Index("idx_user_skills_user", user_skills.c.user_id)
Index("idx_user_skills_skill", user_skills.c.skill_id)

# ============================================
# PROJECTS TABLE
# ============================================

projects = sqlalchemy.Table(
    "projects",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("title", TEXT, nullable=False),
    Column("description", TEXT, nullable=False),
    Column("repo_url", TEXT),
    Column("difficulty", TEXT, nullable=False),
    Column("topics", ARRAY(TEXT), nullable=False),  # ['react', 'web-dev', 'api']
    Column("estimated_hours", sqlalchemy.Integer),
    Column("source", TEXT, nullable=False),  # 'github', 'kaggle', 'curated'
    Column("stars", sqlalchemy.Integer, default=0),
    Column("language", TEXT),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    
    CheckConstraint("difficulty IN ('beginner', 'intermediate', 'advanced')", name="check_difficulty"),
    CheckConstraint("source IN ('github', 'kaggle', 'curated', 'manual', 'kaggle_competition', 'kaggle_dataset')", name="check_source")
)

Index("idx_projects_title", projects.c.title)
Index("idx_projects_difficulty", projects.c.difficulty)
Index("idx_projects_source", projects.c.source)
# GIN index for array search in PostgreSQL
Index("idx_projects_topics", projects.c.topics, postgresql_using="gin")

# ============================================
# PROJECT SKILLS (Junction Table)
# ============================================

project_skills = sqlalchemy.Table(
    "project_skills",
    metadata,
    Column("project_id", sqlalchemy.Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("skill_id", sqlalchemy.Integer, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True),
    Column("is_required", sqlalchemy.Boolean, default=True),
    
    UniqueConstraint("project_id", "skill_id", name="unique_project_skill")
)

Index("idx_project_skills_project", project_skills.c.project_id)
Index("idx_project_skills_skill", project_skills.c.skill_id)

# ============================================
# EMBEDDINGS CACHE 
# ============================================

project_embeddings = sqlalchemy.Table(
    "project_embeddings",
    metadata,
    Column("project_id", sqlalchemy.Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    Column("embedding", Vector(384)),  # 384-dim for all-MiniLM-L6-v2
    Column("model_version", TEXT, default="all-MiniLM-L6-v2"),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
)

# Add index for faster similarity search (IVFFlat or HNSW)
# IVFFlat is good for < 1M vectors
Index('idx_project_embeddings_vector', 
      project_embeddings.c.embedding, 
      postgresql_using='ivfflat',
      postgresql_ops={'embedding': 'vector_cosine_ops'})

# ============================================
# USER ACTIVITY LOG
# ============================================
user_activity_log = sqlalchemy.Table(
    "user_activity_log",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE")),
    Column("activity_type", TEXT, nullable=False),
    Column("details", JSONB),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now())
)
# ============================================
# USER PROJECT INTERACTIONS
# ============================================

user_project_interactions = sqlalchemy.Table(
    "user_project_interactions",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("project_id", sqlalchemy.Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
    Column("interaction_type", TEXT, nullable=False),  # 'viewed', 'bookmarked', 'started', 'completed'
    Column("rating", sqlalchemy.Integer),  # 1-5 star rating
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    
    CheckConstraint("interaction_type IN ('viewed', 'bookmarked', 'started', 'completed')", name="check_interaction_type"),
    CheckConstraint("rating IS NULL OR rating BETWEEN 1 AND 5", name="check_rating_range")
)

Index("idx_interactions_user", user_project_interactions.c.user_id)
Index("idx_interactions_project", user_project_interactions.c.project_id)
Index("idx_interactions_type", user_project_interactions.c.interaction_type)
Index("idx_interactions_created", user_project_interactions.c.created_at)

# ============================================
# PROJECT PLANS
# ============================================

project_plans = sqlalchemy.Table(
    "project_plans",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("project_id", sqlalchemy.Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
    Column("plan_json", JSONB, nullable=False),  # Stores generated project plan
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    
    UniqueConstraint("user_id", "project_id", name="unique_user_project_plan")
)

Index("idx_project_plans_user", project_plans.c.user_id)
Index("idx_project_plans_project", project_plans.c.project_id)

# ============================================
# RECOMMENDATION FEEDBACK (Optional - for evaluation)
# ============================================

recommendation_feedback = sqlalchemy.Table(
    "recommendation_feedback",
    metadata,
    Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("project_id", sqlalchemy.Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
    Column("method", TEXT, nullable=False),  # 'content', 'collaborative', 'hybrid'
    Column("helpful", sqlalchemy.Boolean, nullable=False),
    Column("comment", TEXT),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    
    CheckConstraint("method IN ('content', 'collaborative', 'hybrid')", name="check_method")
)

Index("idx_feedback_user", recommendation_feedback.c.user_id)
Index("idx_feedback_project", recommendation_feedback.c.project_id)
Index("idx_feedback_method", recommendation_feedback.c.method)
