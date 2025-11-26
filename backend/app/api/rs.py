"""
Recommender System API with Vector Similarity Search
File: app/api/v1/recommendations.py

Enhanced with semantic embeddings using pgvector + HuggingFace
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func, or_
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

from sqlalchemy import cast #noqa
from pgvector.sqlalchemy import Vector #noqa

from app.database.sql_engine import get_db_session
from app.database.tables import (
    users, user_profiles, skills, user_skills,
    projects, project_skills, user_project_interactions, 
    project_plans, project_embeddings # noqa
)

# For embeddings
from sentence_transformers import SentenceTransformer

from sqlalchemy import Table, Column, Integer, Numeric, Text, text
from app.database.tables import metadata

router = APIRouter()

# ============================================
# EMBEDDING MODEL (Singleton)
# ============================================

class EmbeddingService:
    """Singleton service for generating embeddings"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Loaded embedding model: all-MiniLM-L6-v2")
        return cls._instance
    
    # Inside .encode(), SentenceTransformer does:
    # Tokenize your text
    # Run MiniLM transformer (gets token embeddings)
    # Apply mean pooling over token embeddings
    # (Optionally) normalize
    # Return a final vector of size 384
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

# Initialize service
embedding_service = EmbeddingService()

popular_projects_view = Table(
    'popular_projects_by_difficulty',
    metadata,
    Column('difficulty', Text),
    Column('project_count', Integer),
    Column('avg_stars', Numeric),
    Column('avg_hours', Numeric),
    extend_existing=True
)

top_skill_combinations_view = Table(
    'top_skill_combinations',
    metadata,
    Column('skill_1', Integer),
    Column('skill_2', Integer),
    Column('skill_1_name', Text),
    Column('skill_2_name', Text),
    Column('project_count', Integer),
    extend_existing=True
)

# ============================================
# PYDANTIC MODELS
# ============================================

class UserSkillCreate(BaseModel):
    skill_id: int
    proficiency: int  # 1-5

class UserProfileCreate(BaseModel):
    skill_level: str  # 'beginner', 'intermediate', 'advanced'
    interests: List[str]
    bio: Optional[str] = None
    github_username: Optional[str] = None
    preferred_project_types: Optional[List[str]] = None
    skills: List[UserSkillCreate]

class ProjectResponse(BaseModel):
    id: int
    title: str
    description: str
    repo_url: Optional[str]
    difficulty: str
    topics: List[str]
    estimated_hours: int
    source: str
    stars: int
    language: str

class RecommendationResponse(BaseModel):
    project_id: int
    title: str
    description: str
    repo_url: Optional[str]
    difficulty: str
    topics: List[str]
    estimated_hours: int
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    reason: str
    semantic_similarity: Optional[float] = None  
    source: str  # NEW: 'github', 'kaggle_competition', 'kaggle_dataset', 'curated'


class InteractionCreate(BaseModel):
    project_id: int
    interaction_type: str  # 'viewed', 'bookmarked', 'started', 'completed'
    rating: Optional[int] = None  # 1-5

# ============================================
# HELPER FUNCTIONS
# ============================================

async def get_user_profile_data(user_id: str, db: AsyncSession) -> Optional[dict]:
    """Get complete user profile with skills"""
    
    # Get profile
    result = await db.execute(
        select(user_profiles).where(user_profiles.c.user_id == user_id)
    )
    profile_row = result.first()
    
    if not profile_row:
        return None
    
# _mapping gives access to columns by name instead of index (providing a dict-like interface, easier to read)
    profile_data = dict(profile_row._mapping) 
    
    # Get user skills
    result = await db.execute(
        select(user_skills, skills.c.name, skills.c.category)
        .join(skills, user_skills.c.skill_id == skills.c.id)
        .where(user_skills.c.user_id == user_id)
    )
    
    profile_data['skills'] = [
        {
            'skill_id': row.skill_id,
            'name': row.name,
            'category': row.category,
            'proficiency': row.proficiency
        }
        for row in result.fetchall()
    ]
    
    return profile_data

async def get_all_projects_with_skills(db: AsyncSession) -> List[dict]:
    """Get all projects with their required skills"""
    
    # Get all projects
    result = await db.execute(select(projects))
    projects_list = []
    
    for project_row in result.fetchall():
        project_data = dict(project_row._mapping)
        
        # Get skills for this project
        skills_result = await db.execute(
            select(project_skills, skills.c.name)
            .join(skills, project_skills.c.skill_id == skills.c.id)
            .where(project_skills.c.project_id == project_data['id'])
        )
        
        project_data['project_skills'] = [
            {'name': row.name, 'is_required': row.is_required}
            for row in skills_result.fetchall()
        ]
        
        projects_list.append(project_data)
    
    return projects_list

def build_user_query_text(user_profile: dict) -> str:
    """Build semantic search query from user profile"""
    parts = []
    
    # Skills with proficiency weights
    skill_names = [s['name'] for s in user_profile['skills']]
    if skill_names:
        parts.append(f"Skills: {', '.join(skill_names)}")
    
    # Interests
    if user_profile.get('interests'):
        parts.append(f"Interests: {', '.join(user_profile['interests'])}")
    
    # Skill level
    parts.append(f"Level: {user_profile.get('skill_level', 'intermediate')}")
    
    # Bio if available
    if user_profile.get('bio'):
        parts.append(user_profile['bio'][:200])
    
    return ". ".join(parts)

# ============================================
# VECTOR SIMILARITY FUNCTIONS
# ============================================

async def find_similar_projects_vector(
    user_query: str,
    db: AsyncSession,
    limit: int = 20,
    difficulty_filter: Optional[str] = None,
    source_filter: Optional[List[str]] = None  
) -> List[dict]:
    """
    Find similar projects using numpy cosine similarity
    """
    import numpy as np
    
    # Generate query embedding
    query_embedding = embedding_service.encode(user_query)
    query_emb = np.array(query_embedding)
    
    # Build query to get all projects with embeddings
    stmt = (
        select(
            projects,
            project_embeddings.c.embedding
        )
        .select_from(projects)
        .join(project_embeddings, projects.c.id == project_embeddings.c.project_id)
    )
    
    # Add filters
    if difficulty_filter:
        stmt = stmt.where(projects.c.difficulty == difficulty_filter)
    
    if source_filter:
        stmt = stmt.where(projects.c.source.in_(source_filter))
    
    result = await db.execute(stmt)
    
    # Calculate similarities in Python
    projects_with_similarity = []
    for row in result:
        project_emb = np.array(row.embedding)
        
        # Cosine similarity
        similarity = np.dot(query_emb, project_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(project_emb)
        )
        
        project_data = {
            'id': row.id,
            'title': row.title,
            'description': row.description,
            'repo_url': row.repo_url,
            'difficulty': row.difficulty,
            'topics': row.topics,
            'estimated_hours': row.estimated_hours,
            'source': row.source,
            'stars': row.stars,
            'language': row.language,
            'semantic_similarity': float(similarity)
        }
        projects_with_similarity.append(project_data)
    
    # Sort by similarity (highest first) and limit
    projects_with_similarity.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    
    return projects_with_similarity[:limit]


async def ensure_project_embedding(project_id: int, db: AsyncSession):
    """Generate and store embedding for a project if not exists"""
    
    # Check if embedding exists
    result = await db.execute(
        select(project_embeddings).where(project_embeddings.c.project_id == project_id)
    )
    
    if result.first():
        return  # Already exists
    
    # Get project data
    result = await db.execute(
        select(projects).where(projects.c.id == project_id)
    )
    project = result.first()
    
    if not project:
        return
    
    # Build embedding text
    embedding_text = f"{project.title}. {project.description}. "
    embedding_text += f"Topics: {', '.join(project.topics)}. "
    embedding_text += f"Language: {project.language}. "
    embedding_text += f"Difficulty: {project.difficulty}"
    
    # Generate embedding
    embedding = embedding_service.encode(embedding_text)
    
    # Store embedding
    await db.execute(
        insert(project_embeddings).values(
            project_id=project_id,
            embedding=embedding,
            model_version="all-MiniLM-L6-v2"
        )
    )

# ============================================
# HYBRID RECOMMENDATION ENGINE
# ============================================

def calculate_skill_match_score(user_profile: dict, project: dict) -> tuple:
    """
    Calculate skill-based match score (traditional approach)
    Returns: (score, matching_skills, missing_skills)
    """
    
    user_skill_names = {s['name'] for s in user_profile['skills']}
    project_skill_names = {ps['name'] for ps in project.get('project_skills', [])}
    
    if not project_skill_names:
        return 0.0, [], []
    
    matching_skills = user_skill_names & project_skill_names
    missing_skills = project_skill_names - user_skill_names
    
    # Skill match score (0-1)
    skill_score = len(matching_skills) / len(project_skill_names) if project_skill_names else 0
    
    return skill_score, list(matching_skills), list(missing_skills)

def calculate_hybrid_score(
    user_profile: dict,
    project: dict,
    semantic_similarity: float
) -> tuple:
    """
    Combine semantic similarity with traditional features
    
    Returns: (final_score, matching_skills, missing_skills, reason)
    """
    
    reasons = []
    
    # 1. Semantic Similarity (50% weight)
    semantic_score = semantic_similarity
    
    # 2. Skill Match (25% weight)
    skill_score, matching_skills, missing_skills = calculate_skill_match_score(user_profile, project)
    
    if matching_skills:
        reasons.append(f"Matches {len(matching_skills)} of your skills")
    
    # 3. Difficulty Match (15% weight)
    user_level = user_profile.get('skill_level', 'intermediate')
    project_level = project.get('difficulty', 'intermediate')
    
    difficulty_scores = {
        ('beginner', 'beginner'): 1.0,
        ('beginner', 'intermediate'): 0.3,
        ('intermediate', 'beginner'): 0.8,
        ('intermediate', 'intermediate'): 1.0,
        ('intermediate', 'advanced'): 0.4,
        ('advanced', 'intermediate'): 0.7,
        ('advanced', 'advanced'): 1.0,
    }
    
    difficulty_score = difficulty_scores.get((user_level, project_level), 0.5)
    
    if difficulty_score >= 0.8:
        reasons.append("Appropriate difficulty level")
    
    # 4. Growth Opportunity (10% weight)
    growth_score = 0.0
    if 1 <= len(missing_skills) <= 3:
        growth_score = 1.0
        reasons.append(f"Learn {len(missing_skills)} new skill(s)")
    
    # Weighted combination
    final_score = (
        0.50 * semantic_score +
        0.25 * skill_score +
        0.15 * difficulty_score +
        0.10 * growth_score
    )
    
    # Add semantic match reason
    if semantic_similarity > 0.7:
        reasons.insert(0, "Highly relevant to your profile")
    elif semantic_similarity > 0.5:
        reasons.insert(0, "Good match for your interests")
    
    source = project.get('source', 'unknown')
    if source == 'kaggle_competition':
        reasons.append("🏆 Kaggle competition")
    elif source == 'kaggle_dataset':
        reasons.append("📊 Data analysis project")
    
    reason = " • ".join(reasons) if reasons else "Potential match"
    
    return final_score, matching_skills, missing_skills, reason

# ============================================
# API ENDPOINTS
# ============================================

@router.get("/")
async def root():
    return {
        "message": "Recommender System API with Vector Search",
        "version": "2.1",
        "features": ["semantic_search", "hybrid_scoring"],
        "sources": ["github", "kaggle_competitions", "kaggle_datasets", "curated"]
    }

# ------ USER PROFILE ------

@router.post("/users/{user_id}/profile")
async def create_user_profile(
    user_id: str,
    profile: UserProfileCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create or update user profile"""
    print(f"🔍 POST profile request for user_id: {user_id}")
    print(f"🔍 Profile data: {profile}")
    
    try:
        user_uuid = UUID(user_id)
        print(f"✅ Converted to UUID: {user_uuid}")
    except ValueError as e:
        print(f"❌ UUID conversion failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    try:
        # Check if user exists
        result = await db.execute(
            select(users).where(users.c.user_id == user_uuid)
        )
        if not result.first():
            print(f"❌ User not found: {user_uuid}")
            raise HTTPException(status_code=404, detail="User not found")
        
        print("✅ User exists")
        
        profile_data = {
            'user_id': user_uuid,
            'skill_level': profile.skill_level,
            'interests': profile.interests or [],
            'bio': profile.bio or '',
            'github_username': profile.github_username or '',
            'preferred_project_types': profile.preferred_project_types or [],
            'updated_at': datetime.utcnow()
        }
        
        # Check if profile exists
        result = await db.execute(
            select(user_profiles).where(user_profiles.c.user_id == user_uuid)
        )
        
        existing_profile = result.first()
        
        if existing_profile:
            print("🔄 Updating existing profile")
            await db.execute(
                update(user_profiles)
                .where(user_profiles.c.user_id == user_uuid)
                .values(**profile_data)
            )
        else:
            print("➕ Creating new profile")
            await db.execute(insert(user_profiles).values(**profile_data))
        
        # Delete existing skills
        await db.execute(
            delete(user_skills).where(user_skills.c.user_id == user_uuid)
        )
        print("🗑️ Deleted existing skills")
        
        # Insert new skills
        if profile.skills:
            print(f"➕ Inserting {len(profile.skills)} skills")
            for skill in profile.skills:
                print(f"  Skill: {skill.skill_id} (type: {type(skill.skill_id)}), proficiency: {skill.proficiency}")
                
                # Handle skill_id as either int or string
                if isinstance(skill.skill_id, int):
                    skill_id_value = skill.skill_id  # Keep as int
                elif isinstance(skill.skill_id, str):
                    try:
                        skill_id_value = UUID(skill.skill_id)  # Try UUID
                    except ValueError:
                        skill_id_value = int(skill.skill_id)  # Or convert to int
                else:
                    skill_id_value = skill.skill_id
                
                await db.execute(
                    insert(user_skills).values(
                        user_id=user_uuid,
                        skill_id=skill_id_value,  # ← Use the converted value
                        proficiency=skill.proficiency
                    )
                )
        
        await db.commit()
        print("✅ Profile saved successfully")
        
        return {"message": "Profile created successfully"}
    
    except Exception as e:
        await db.rollback()
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")

@router.get("/users/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get user profile"""
    print(f"🔍 GET profile request for user_id: {user_id}")
    
    try:
        user_uuid = UUID(user_id)
        print(f"✅ Converted to UUID: {user_uuid}")
    except ValueError as e:
        print(f"❌ UUID conversion failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Query the profile directly
    print(f"🔍 Querying database for user_id: {user_uuid}")
    result = await db.execute(
        select(user_profiles).where(user_profiles.c.user_id == user_uuid)
    )
    profile_row = result.first()
    
    print(f"🔍 Query result: {profile_row}")
    
    if not profile_row:
        print(f"❌ No profile found for user_id: {user_uuid}")
        raise HTTPException(status_code=404, detail="Profile not found")
    
    print("✅ Profile found!")
    
    # Get associated skills
    skills_result = await db.execute(
        select(
            user_skills.c.skill_id,
            user_skills.c.proficiency,
            skills.c.name
        )
        .select_from(user_skills)
        .join(skills, user_skills.c.skill_id == skills.c.id) 
        .where(user_skills.c.user_id == user_uuid)
    )
    skills_rows = skills_result.all()
    print(f"🔍 Found {len(skills_rows)} skills")
    
    # Build response
    profile_dict = dict(profile_row._mapping)
    profile_dict['skills'] = [
        {
            'skill_id': str(row.skill_id),
            'name': row.name,
            'proficiency': str(row.proficiency)
        }
        for row in skills_rows
    ]
    
    return profile_dict

# ------ PROJECTS ------

@router.get("/projects/search")
async def search_projects(
    q: Optional[str] = None,
    difficulty: Optional[str] = None,
    source: Optional[str] = Query(None, regex="^(github|kaggle_competition|kaggle_dataset|curated)$"), 
    use_semantic: bool = True,
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Search projects with optional semantic search
    
    Parameters:
    - q: Search query
    - difficulty: Filter by difficulty (beginner/intermediate/advanced)
    - source: Filter by source (github/kaggle_competition/kaggle_dataset/curated)
    - use_semantic: Use vector similarity (True) or keyword search (False)
    - limit: Maximum results
    """
    
    source_filter = [source] if source else None
    
    if use_semantic and q:
        # Use vector search
        projects_list = await find_similar_projects_vector(
            user_query=q,
            db=db,
            limit=limit,
            difficulty_filter=difficulty,
            source_filter=source_filter
        )
    else:
        # Traditional search
        query = select(projects)
        
        if difficulty:
            query = query.where(projects.c.difficulty == difficulty)
            
        if source:
            query = query.where(projects.c.source == source)
        
        if q:
            search_term = f"%{q}%"
            query = query.where(
                or_(
                    projects.c.title.ilike(search_term),
                    projects.c.description.ilike(search_term)
                )
            )
        
        query = query.limit(limit)
        result = await db.execute(query)
        projects_list = [dict(row._mapping) for row in result.fetchall()]
    
    return {
        "projects": projects_list,
        "count": len(projects_list),
        "search_type": "semantic" if (use_semantic and q) else "keyword",
        "filters": {
            "difficulty": difficulty,
            "source": source
        }
    }

@router.get("/projects/{project_id}")
async def get_project_detail(
    project_id: int,
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """Get project details with optional skill match analysis"""
    
    # Get project
    result = await db.execute(
        select(projects).where(projects.c.id == project_id)
    )
    project_row = result.first()
    
    if not project_row:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_data = dict(project_row._mapping)
    
    # Get project skills
    result = await db.execute(
        select(project_skills, skills.c.name, skills.c.category)
        .join(skills, project_skills.c.skill_id == skills.c.id)
        .where(project_skills.c.project_id == project_id)
    )
    
    project_data['skills'] = [
        {
            'name': row.name,
            'category': row.category,
            'is_required': row.is_required
        }
        for row in result.fetchall()
    ]
    
    # If user_id provided, calculate match
    if user_id:
        user_profile = await get_user_profile_data(user_id, db)
        if user_profile:
            # Get semantic similarity
            user_query = build_user_query_text(user_profile)
            similar = await find_similar_projects_vector(user_query, db, limit=100)
            
            semantic_sim = 0.0
            for p in similar:
                if p['id'] == project_id:
                    semantic_sim = p['semantic_similarity']
                    break
            
            # Calculate hybrid score
            score, matching, missing, reason = calculate_hybrid_score(
                user_profile,
                {
                    'project_skills': [{'name': s['name']} for s in project_data['skills']],
                    'topics': project_data['topics'],
                    'difficulty': project_data['difficulty'],
                    'source': project_data['source']
                },
                semantic_sim
            )
            
            project_data['match_analysis'] = {
                'score': round(score * 100, 1),
                'matching_skills': matching,
                'missing_skills': missing,
                'reason': reason,
                'semantic_similarity': round(semantic_sim * 100, 1)
            }
    
    return project_data

# ------ STATS ------

@router.get("/admin/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db_session)):
    """Get system statistics"""
    
    # Project counts
    result = await db.execute(select(func.count()).select_from(projects))
    total_projects = result.scalar()
    
    # Embeddings count
    result = await db.execute(select(func.count()).select_from(project_embeddings))
    total_embeddings = result.scalar()
    
    # By difficulty
    result = await db.execute(
        select(projects.c.difficulty, func.count())
        .group_by(projects.c.difficulty)
    )
    by_difficulty = {row[0]: row[1] for row in result}
    # by source
    result = await db.execute(
        select(projects.c.source, func.count())
        .group_by(projects.c.source)
    )
    by_source = {row[0]: row[1] for row in result}
    
    # User profiles count
    result = await db.execute(select(func.count()).select_from(user_profiles))
    total_users = result.scalar()
    
    # Interactions count
    result = await db.execute(select(func.count()).select_from(user_project_interactions))
    total_interactions = result.scalar()
    
    return {
        "projects": {
            "total": total_projects,
            "with_embeddings": total_embeddings,
            "embedding_coverage": round(total_embeddings / total_projects * 100, 1) if total_projects > 0 else 0,
            "by_difficulty": by_difficulty,
            "by_source": by_source 
        },
        "users": {
            "total_profiles": total_users
        },
        "interactions": {
            "total": total_interactions
        },
        "embedding_model": "all-MiniLM-L6-v2"
    }
    
# ------ KAGGLE-SPECIFIC ENDPOINTS ------

@router.get("/kaggle/competitions")
async def get_kaggle_competitions(
    difficulty: Optional[str] = None,
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db_session)
):
    """Get Kaggle competition projects"""
    
    query = select(projects).where(projects.c.source == 'kaggle_competition')
    
    if difficulty:
        query = query.where(projects.c.difficulty == difficulty)
    
    query = query.order_by(projects.c.stars.desc()).limit(limit)
    
    result = await db.execute(query)
    competitions = [dict(row._mapping) for row in result.fetchall()]
    
    return {
        "competitions": competitions,
        "count": len(competitions),
        "filter": {"difficulty": difficulty}
    }

@router.get("/kaggle/datasets")
async def get_kaggle_datasets(
    difficulty: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db_session)
):
    """Get Kaggle dataset analysis projects"""
    
    query = select(projects).where(projects.c.source == 'kaggle_dataset')
    
    if difficulty:
        query = query.where(projects.c.difficulty == difficulty)
    
    if topic:
        # Search in topics array
        query = query.where(projects.c.topics.contains([topic]))
    
    query = query.order_by(projects.c.stars.desc()).limit(limit)
    
    result = await db.execute(query)
    datasets = [dict(row._mapping) for row in result.fetchall()]
    
    return {
        "datasets": datasets,
        "count": len(datasets),
        "filters": {"difficulty": difficulty, "topic": topic}
    }

@router.get("/sources")
async def get_available_sources(db: AsyncSession = Depends(get_db_session)):
    """Get all available project sources with counts"""
    
    result = await db.execute(
        select(projects.c.source, func.count())
        .group_by(projects.c.source)
    )
    
    sources = {}
    for row in result:
        source_name = row[0]
        count = row[1]
        
        # Add friendly descriptions
        descriptions = {
            'github': 'Open source repositories',
            'kaggle_competition': 'Data science competitions',
            'kaggle_dataset': 'Dataset analysis projects',
            'curated': 'Hand-picked learning projects'
        }
        
        sources[source_name] = {
            'count': count,
            'description': descriptions.get(source_name, 'Unknown source')
        }
    
    return {
        "sources": sources,
        "total_sources": len(sources)
    }