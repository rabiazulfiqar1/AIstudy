"""
Data Generator with Embedding Generation - GitHub & Kaggle Support
Works with your async engine setup

Usage:
    python -m app.scripts.generate_data quick
    python -m app.scripts.generate_data full
    python -m app.scripts.generate_data kaggle
    python -m app.scripts.generate_data embeddings
    python -m app.scripts.generate_data stats
"""

import asyncio
import requests
from typing import List, Dict, Optional
import time
from sqlalchemy import select, insert, delete, func
from sentence_transformers import SentenceTransformer
from app.core.config import config

from app.database.sql_engine import get_db 
from app.database.tables import (
    skills, projects, project_skills, project_embeddings,
    users, user_profiles, user_skills #noqa
)

# GitHub Configuration
GITHUB_TOKEN = config.GITHUB_TOKEN
KAGGLE_USERNAME = getattr(config, 'KAGGLE_USERNAME', None)
KAGGLE_KEY = getattr(config, 'KAGGLE_KEY', None)

# ============================================
# EMBEDDING SERVICE
# ============================================

class EmbeddingGenerator:
    """Service for generating embeddings"""
    
    def __init__(self):
        print("🤖 Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def build_project_text(self, project: Dict) -> str:
        """Build text representation of project for embedding"""
        parts = []
        
        # Title (most important)
        parts.append(project.get('title', ''))
        
        # Description
        if project.get('description'):
            parts.append(project['description'][:300])
        
        # Topics
        if project.get('topics'):
            parts.append(f"Topics: {', '.join(project['topics'])}")
        
        # Language
        if project.get('language'):
            parts.append(f"Language: {project['language']}")
        
        # Difficulty
        if project.get('difficulty'):
            parts.append(f"Difficulty: {project['difficulty']}")
        
        return ". ".join(parts)

# Initialize generator
embedding_generator = EmbeddingGenerator()

# ============================================
# INFERENCE LOGIC
# ============================================

def infer_difficulty(repo_data: Dict) -> str:
    """Intelligently infer project difficulty"""
    score = 0
    max_score = 20  # Maximum possible raw score before normalization

    # 1. STARS - (popularity, weak proxy)
    stars = repo_data.get('stargazers_count', 0)
    if stars > 10000:
        score += 2  
    elif stars > 1000:
        score += 1 
    elif stars > 100:
        score += 0.5  
    
    # 2. FORKS - Strong indicator of complexity (people fork to understand/modify)
    forks = repo_data.get('forks_count', 0)
    if forks > 5000:
        score += 2
    elif forks > 1000:
        score += 1
    elif forks > 100:
        score += 0.5
    
    # 3. TOPICS - Very good indicators
    topics = [t.lower() for t in repo_data.get('topics', [])]
    
    beginner_indicators = [
        'beginner', 'tutorial', 'learning', 'starter', 'first-timers',
        'good-first-issue', 'beginner-friendly', 'intro', 'basic',
        'education', 'course', 'bootcamp', 'example', 'sample'
    ]
    # for intermediate since it's neutral, we won't adjust score
    # intermediate_indicators = [
    #     'intermediate', 'project', 'full-stack', 'web-app', 'api',
    #     'rest', 'crud', 'database', 'authentication'
    # ]
    advanced_indicators = [
        'advanced', 'production', 'enterprise', 'scalable', 'distributed',
        'microservices', 'kubernetes', 'system-design', 'architecture',
        'machine-learning', 'deep-learning', 'compiler', 'os', 'kernel',
        'blockchain', 'cryptography', 'high-performance', 'concurrent',
    ]
    
    if any(ind in topics for ind in beginner_indicators):
        score -= 2
    # if any(ind in topics for ind in intermediate_indicators):
    #     score += 0  # Neutral
    if any(ind in topics for ind in advanced_indicators):
        score += 3
    
    # 4. DESCRIPTION analysis
    description = (repo_data.get('description') or '').lower()
    if any(word in description for word in ['beginner', 'learn', 'tutorial', 'intro', 'simple', 'easy']):
        score -= 1
    if any(word in description for word in ['advanced', 'complex', 'enterprise', 'production', 'professional']):
        score += 2
    
    # 5. LANGUAGE complexity
    language = repo_data.get('language')
    if language:
        language_lower = language.lower()
        # Very complex languages
        if language_lower in ['rust', 'c++', 'c', 'assembly', 'haskell', 'ocaml', 'scala', 'erlang']:
            score += 2
        # Complex languages
        elif language_lower in ['java', 'go', 'kotlin', 'swift', 'typescript']:
            score += 1
        # Beginner-friendly languages
        elif language_lower in ['python', 'javascript', 'ruby', 'php']:
            score += 0  # Neutral - language alone doesn't determine difficulty
        # Markup/styling (typically beginner)
        elif language_lower in ['html', 'css', 'markdown']:
            score -= 1
    
    
    # 6. OPEN ISSUES - High issue count might indicate complexity or active development
    open_issues = repo_data.get('open_issues_count', 0)
    if open_issues > 500:
        score += 1
    elif open_issues > 100:
        score += 0.5
    
    # 7. CONTRIBUTORS - More contributors often means more complex codebase
    # Note: This might not be in basic repo_data, may need separate API call
    contributors = repo_data.get('contributors_count', 0)
    if contributors > 100:
        score += 2
    elif contributors > 20:
        score += 1
    
    # 8. DEFAULT BRANCH COMMIT COUNT (if available)
    # More commits = more mature/complex project
    
    # 9. REPO SIZE OR LINES OF CODE 
    if 'lines_of_code' in repo_data:
        loc = repo_data['lines_of_code']
        if loc > 50000:
            score += 3
        elif loc > 10000:
            score += 2
        elif loc < 1000:
            score -= 1
    else:
        size_kb = repo_data.get('size', 0)
        if size_kb > 10000:
            score += 2
        elif size_kb < 100:
            score -= 1

    
    # ----------------------------
    # Normalize score to 1-10
    # ----------------------------
    normalized_score = max(1, min(10, round((score / max_score) * 10)))
    
    # SCORING THRESHOLDS (adjusted for new scoring system)
    if normalized_score <= 3:
        difficulty = 'beginner'
    elif normalized_score <= 6:
        difficulty = 'intermediate'
    else:
        difficulty = 'advanced'
        
    return difficulty

def estimate_hours(difficulty: str, repo_data: Dict) -> int:
    """Estimate project completion hours"""
    base_hours = {
        'beginner': 15,
        'intermediate': 40,
        'advanced': 80
    }
    
    hours = base_hours.get(difficulty, 40)
    
    if 'lines_of_code' in repo_data:
        loc = repo_data['lines_of_code']
        if loc > 50000:
            hours *= 1.5
        elif loc > 10000:
            hours *= 1.2
        elif loc < 1000:
            hours *= 0.7
    else:
        size_kb = repo_data.get('size', 0)
        if size_kb > 50000:
            hours *= 1.5
        elif size_kb > 10000:
            hours *= 1.2
        elif size_kb < 100:
            hours *= 0.7
    
    if repo_data.get('has_issues') and repo_data.get('open_issues_count', 0) > 10:
        hours *= 1.1
        
    contributors = repo_data.get('contributors_count', 0)
    if contributors > 50:
        hours *= 1.2
    elif contributors > 10:
        hours *= 1.1
        
     # -----------------------
    # 4. Adjust for CI/CD and Docker
    # -----------------------
    has_docker = repo_data.get('has_docker', False)
    has_ci_cd = repo_data.get('has_ci_cd', False)  # e.g., GitHub Actions, Jenkins, Travis CI

    if has_docker: # Docker projects → +10% time (setup & environment complexity)
        hours *= 1.1  
    if has_ci_cd: # CI/CD pipelines → +10% time (build/test/deploy complexity)
        hours *= 1.1 

    hours = min(hours, 200)
    
    return round(hours)

SKILL_KEYWORDS = {
    # Programming Languages
    'Python': ['python', 'py', 'django', 'flask', 'fastapi', 'pytorch', 'tensorflow'],
    'JavaScript': ['javascript', 'js', 'node', 'nodejs', 'npm'],
    'TypeScript': ['typescript', 'ts'],
    'Java': ['java', 'spring', 'springboot'],
    'C++': ['c++', 'cpp'],
    'C': ['c'],
    'C#': ['c#', 'dotnet', '.net'],
    'Go': ['go', 'golang'],
    'Rust': ['rust'],
    'Kotlin': ['kotlin'],
    'Swift': ['swift'],
    'PHP': ['php', 'laravel', 'symfony'],
    'Ruby': ['ruby', 'rails'],
    'Scala': ['scala', 'akka'],
    'Haskell': ['haskell'],

    # Frontend Frameworks
    'React': ['react', 'reactjs', 'react-native'],
    'Next.js': ['nextjs', 'next.js', 'next'],
    'Vue.js': ['vue', 'vuejs', 'vue.js'],
    'Angular': ['angular', 'ng'],
    'Svelte': ['svelte', 'sveltekit'],
    'Ember.js': ['ember', 'emberjs'],

    # Backend Frameworks
    'Node.js': ['nodejs', 'node.js', 'express'],
    'FastAPI': ['fastapi'],
    'Django': ['django'],
    'Flask': ['flask'],
    'Spring Boot': ['springboot', 'spring'],
    'Ruby on Rails': ['rails', 'ruby on rails'],
    'Laravel': ['laravel'],

    # Databases
    'PostgreSQL': ['postgresql', 'postgres', 'psql'],
    'MySQL': ['mysql', 'mariadb'],
    'MongoDB': ['mongodb', 'mongo'],
    'SQLite': ['sqlite', 'sqlite3'],
    'Redis': ['redis'],
    'Elasticsearch': ['elasticsearch'],

    # Machine Learning / AI
    'Machine Learning': ['machine-learning', 'ml', 'scikit-learn', 'sklearn'],
    'Deep Learning': ['deep-learning', 'neural-network', 'keras', 'tensorflow', 'pytorch'],
    'NLP': ['nlp', 'natural-language', 'transformers', 'huggingface'],
    'Computer Vision': ['computer-vision', 'cv', 'opencv', 'yolov5', 'pillow', 'detectron'],
    'Reinforcement Learning': ['reinforcement-learning', 'rl', 'gym', 'stable-baselines'],

    # DevOps / CI/CD / Containers
    'Docker': ['docker', 'container'],
    'Kubernetes': ['kubernetes', 'k8s'],
    'Jenkins': ['jenkins'],
    'GitHub Actions': ['github-actions'],
    'Travis CI': ['travis', 'travis-ci'],
    'CircleCI': ['circleci'],
    'Ansible': ['ansible'],
    'Terraform': ['terraform'],

    # Cloud Platforms
    'AWS': ['aws', 'amazon-web-services'],
    'GCP': ['gcp', 'google-cloud', 'google-cloud-platform'],
    'Azure': ['azure', 'microsoft-azure'],
    'Heroku': ['heroku'],
    'DigitalOcean': ['digitalocean'],

    # API / Web Development
    'API Development': ['api', 'rest', 'graphql', 'restful'],
    'Web Development': ['web', 'webapp', 'website', 'frontend', 'backend', 'fullstack'],
    'HTML': ['html', 'html5'],
    'CSS': ['css', 'scss', 'sass', 'tailwind', 'bootstrap'],

    # Version Control
    'Git': ['git', 'github', 'gitlab', 'bitbucket'],

    # Data Tools
    'Pandas': ['pandas'],
    'NumPy': ['numpy', 'np'],
    'Matplotlib': ['matplotlib', 'plt'],
    'Seaborn': ['seaborn'],
    'Plotly': ['plotly'],
    'OpenCV': ['opencv', 'cv2'],
}

def extract_skills(repo_data: Dict) -> List[str]:
    """Extract relevant skills from repository data"""
    skills_found = set()
    
    # Safely handle None values
    name = repo_data.get('name') or ''
    description = repo_data.get('description') or ''
    topics = repo_data.get('topics') or []
    language = repo_data.get('language') or ''
    
    searchable_text = ' '.join([
        name,
        description,
        ' '.join(topics),
        language
    ]).lower()
    
    if language:
        skills_found.add(language)
    
    # .items() returns an iterator over (key, value) pairs in the dictionary.
    for skill_name, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in searchable_text:
                skills_found.add(skill_name)
                break
            
    # Optional fallback if nothing is detected
    if not skills_found and topics:
        if 'web' in ' '.join(topics).lower():
            skills_found.add('Web Development')
        if any(t.lower() in ['ml', 'ai', 'data'] for t in topics):
            skills_found.add('Machine Learning')
    
    return list(skills_found)

# ============================================
# KAGGLE API INTEGRATION
# ============================================

def fetch_kaggle_competitions(max_results: int = 20) -> List[Dict]:
    """Fetch Kaggle competitions"""
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        print("⚠️ Kaggle credentials not configured")
        return []
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        competitions = api.competitions_list(page=1)[:max_results]
        
        results = []
        for comp in competitions:
            
            # Handle tags - can be string or list
            tags = []
            if hasattr(comp, 'tags'):
                if isinstance(comp.tags, str):
                    tags = [t.strip() for t in comp.tags.split(',') if t.strip()]
                elif isinstance(comp.tags, list):
                    tags = comp.tags
            
            results.append({
                'id': comp.id if hasattr(comp, 'id') else comp.ref,
                'title': comp.title,
                'description': comp.description if comp.description else f"Kaggle competition: {comp.title}",
                'url': comp.url if hasattr(comp, 'url') else f"https://www.kaggle.com/c/{comp.ref}",
                'reward': comp.reward if hasattr(comp, 'reward') else '',
                'deadline': str(comp.deadline) if hasattr(comp, 'deadline') and comp.deadline else None,
                'category': comp.category if hasattr(comp, 'category') else 'general',
                'tags': tags
            })
        
        print(f"✅ Found {len(results)} competitions")
        
        return results
    except ImportError:
        print("⚠️ Kaggle package not installed. Run: pip install kaggle")
        return []
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        return []

def fetch_kaggle_datasets(max_results: int = 20) -> List[Dict]:
    """Fetch popular Kaggle datasets for project ideas"""
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        print("⚠️ Kaggle credentials not configured")
        return []
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        datasets = api.dataset_list(sort_by='votes', page=1)[:max_results]
        
        results = []
        for ds in datasets:
            
            ref = ds.ref if hasattr(ds, 'ref') else ''
            title = ds.title if hasattr(ds, 'title') else 'Untitled Dataset'
            subtitle = ds.subtitle if hasattr(ds, 'subtitle') else ''
            description = subtitle if subtitle else f"Kaggle dataset: {title}"
                    
            # Handle different attribute names
            votes = 0
            if hasattr(ds, 'voteCount'):
                votes = ds.voteCount
            elif hasattr(ds, 'votes'):
                votes = ds.votes
            elif hasattr(ds, 'usabilityRating'):
                votes = int(ds.usabilityRating * 100)  # Convert rating to pseudo-votes
            
            downloads = 0
            if hasattr(ds, 'downloadCount'):
                downloads = ds.downloadCount
            elif hasattr(ds, 'downloads'):
                downloads = ds.downloads
            
            # Handle tags
            tags = []
            if hasattr(ds, 'tags'):
                if isinstance(ds.tags, str):
                    tags = [t.strip() for t in ds.tags.split(',') if t.strip()]
                elif isinstance(ds.tags, list):
                    tags = ds.tags
            
            results.append({
                'ref': ref,
                'title': title,
                'description': description[:500],
                'url': f"https://www.kaggle.com/datasets/{ref}",
                'votes': votes,
                'downloads': downloads,
                'tags': tags
            })
        
        print(f"✅ Found {len(results)} datasets")
        
        return results
    except ImportError:
        print("⚠️ Kaggle package not installed. Run: pip install kaggle")
        return []
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        return []

def process_kaggle_competition(comp: Dict) -> Dict:
    """Process Kaggle competition into project format"""
    # Infer difficulty based on reward and category
    reward = comp.get('reward', '')
    difficulty = 'intermediate'
    
    if '$' in reward:
        try:
            amount = int(''.join(filter(str.isdigit, reward)))
            if amount > 50000:
                difficulty = 'advanced'
            elif amount > 10000:
                difficulty = 'intermediate'
            else:
                difficulty = 'beginner'
        except: #noqa
            pass
    
    category = comp.get('category', '').lower()
    if 'getting started' in category or 'playground' in category:
        difficulty = 'beginner'
    elif 'featured' in category or 'research' in category:
        difficulty = 'advanced'
    
    # Extract skills from tags and description
    topics = comp.get('tags', [])
    extracted_skills = ['Python', 'Data Science', 'Machine Learning']
    
    # For competitions
    topics = [t.name if hasattr(t, 'name') else str(t) for t in comp.get('tags', [])]
    
    searchable = ' '.join([comp.get('title', ''), comp.get('description', ''), ' '.join(topics)]).lower()
    
    if any(word in searchable for word in ['nlp', 'text', 'language']):
        extracted_skills.append('NLP')
    if any(word in searchable for word in ['image', 'vision', 'cv', 'detection']):
        extracted_skills.append('Computer Vision')
    if any(word in searchable for word in ['time', 'forecast', 'series']):
        extracted_skills.append('Time Series')
    if any(word in searchable for word in ['deep', 'neural', 'cnn', 'lstm']):
        extracted_skills.append('Deep Learning')
    
    # one line dictionary lookup : Create a dictionary, Immediately use the key difficulty (difficulty = beginner or intermediate or advanced), Return the value
    hours = {
        'beginner': 20,
        'intermediate': 50,
        'advanced': 100
    }[difficulty]
    
    return {
        'title': comp['title'],
        'description': comp['description'][:500],
        'repo_url': comp.get('url'),
        'difficulty': difficulty,
        'topics': topics[:10],
        'estimated_hours': hours,
        'source': 'kaggle_competition',
        'stars': 0,
        'language': 'Python',
        'extracted_skills': list(set(extracted_skills))
    }

def process_kaggle_dataset(dataset: Dict) -> Dict:
    """Process Kaggle dataset into project format"""
    # Most datasets are intermediate difficulty
    votes = dataset.get('votes', 0)
    downloads = dataset.get('downloads', 0)
    
    if votes > 1000 or downloads > 10000:
        difficulty = 'intermediate'
    elif votes > 100 or downloads > 1000:
        difficulty = 'beginner'
    else:
        difficulty = 'beginner'
    
    extracted_skills = ['Python', 'Data Science', 'Pandas']
    
    # FIX: Extract string names from tag objects
    tags = dataset.get('tags', [])
    tag_strings = []
    for t in tags:
        if hasattr(t, 'name'):
            tag_strings.append(t.name)
        elif hasattr(t, 'ref'):
            tag_strings.append(t.ref)
        elif isinstance(t, str):
            tag_strings.append(t)
        else:
            tag_strings.append(str(t))
    
    searchable = ' '.join([
        dataset.get('title', ''), 
        dataset.get('description', ''), 
        ' '.join(tag_strings)
    ]).lower()
    
    if any(word in searchable for word in ['nlp', 'text']):
        extracted_skills.extend(['NLP', 'Machine Learning'])
    if any(word in searchable for word in ['image', 'vision']):
        extracted_skills.extend(['Computer Vision', 'Deep Learning'])
    if any(word in searchable for word in ['visualization', 'plot']):
        extracted_skills.extend(['Matplotlib', 'Seaborn'])
    
    title = f"Analysis: {dataset['title']}"
    description = f"Analyze and visualize the {dataset['title']} dataset. {dataset.get('description', '')}"
    
    return {
        'title': title[:100],
        'description': description[:500],
        'repo_url': dataset.get('url'),
        'difficulty': difficulty,
        'topics': tag_strings[:10],  # Now these are strings!
        'estimated_hours': 25,
        'source': 'kaggle_dataset',
        'stars': dataset.get('votes', 0),
        'language': 'Python',
        'extracted_skills': list(set(extracted_skills))
    }

# ============================================
# CURATED DATA 
# ============================================

CURATED_PROJECTS = [
    {
        'title': 'Personal Portfolio Website',
        'description': 'Build a responsive portfolio website to showcase your projects using modern web technologies',
        'repo_url': None,
        'difficulty': 'beginner',
        'topics': ['web-development', 'html', 'css', 'javascript', 'portfolio'],
        'estimated_hours': 12,
        'extracted_skills': ['HTML/CSS', 'JavaScript', 'Web Development']  # RENAMED from 'skills'
    },
    {
        'title': 'RESTful API with Authentication',
        'description': 'Create a secure REST API with JWT authentication, CRUD operations, and database integration',
        'repo_url': None,
        'difficulty': 'intermediate',
        'topics': ['backend', 'api', 'authentication', 'database'],
        'estimated_hours': 35,
        'extracted_skills': ['Node.js', 'API Development', 'PostgreSQL']
    },
    {
        'title': 'Real-time Chat Application',
        'description': 'Build a real-time chat app with WebSockets, user authentication, and message persistence',
        'repo_url': None,
        'difficulty': 'intermediate',
        'topics': ['websockets', 'real-time', 'chat', 'full-stack'],
        'estimated_hours': 45,
        'extracted_skills': ['Node.js', 'React', 'MongoDB', 'Web Development']
    },
    {
        'title': 'To-Do List with React',
        'description': 'Create an interactive to-do list application with local storage and filtering',
        'repo_url': None,
        'difficulty': 'beginner',
        'topics': ['react', 'frontend', 'javascript'],
        'estimated_hours': 10,
        'extracted_skills': ['React', 'JavaScript']
    },
    {
        'title': 'E-commerce Product Catalog',
        'description': 'Develop a full-stack e-commerce product catalog with search, filters, and shopping cart',
        'repo_url': None,
        'difficulty': 'advanced',
        'topics': ['e-commerce', 'full-stack', 'database'],
        'estimated_hours': 80,
        'extracted_skills': ['React', 'Node.js', 'PostgreSQL', 'API Development']
    },
    {
        'title': 'Sentiment Analysis Tool',
        'description': 'Build a tool that analyzes sentiment in text using natural language processing',
        'repo_url': None,
        'difficulty': 'intermediate',
        'topics': ['nlp', 'machine-learning', 'text-analysis'],
        'estimated_hours': 30,
        'extracted_skills': ['Python', 'NLP', 'Machine Learning']
    }
]

GITHUB_SEARCH_QUERIES = [
    "good-first-issue language:python stars:>100",
    "beginner-friendly language:javascript stars:>100",
    "react-project stars:>200",
    "machine-learning python stars:>200",
    "fastapi tutorial stars:>50",
    "nextjs fullstack stars:>100",
    "docker kubernetes stars:>100",
]

# ============================================
# DATABASE OPERATIONS 
# ============================================

async def seed_skills():
    """Seed initial skills into database"""
    skill_data = [
        # Languages
        {'name': 'Python', 'category': 'language'},
        {'name': 'JavaScript', 'category': 'language'},
        {'name': 'TypeScript', 'category': 'language'},
        {'name': 'Java', 'category': 'language'},
        {'name': 'Go', 'category': 'language'},
        
        # Frontend
        {'name': 'React', 'category': 'framework'},
        {'name': 'Next.js', 'category': 'framework'},
        {'name': 'Vue.js', 'category': 'framework'},
        {'name': 'HTML/CSS', 'category': 'framework'},
        
        # Backend
        {'name': 'Node.js', 'category': 'framework'},
        {'name': 'FastAPI', 'category': 'framework'},
        {'name': 'Django', 'category': 'framework'},
        {'name': 'Flask', 'category': 'framework'},
        
        # Databases
        {'name': 'PostgreSQL', 'category': 'tool'},
        {'name': 'MongoDB', 'category': 'tool'},
        {'name': 'Redis', 'category': 'tool'},
        
        # ML/AI
        {'name': 'Machine Learning', 'category': 'domain'},
        {'name': 'Deep Learning', 'category': 'domain'},
        {'name': 'NLP', 'category': 'domain'},
        {'name': 'Computer Vision', 'category': 'domain'},
        
        # DevOps
        {'name': 'Docker', 'category': 'tool'},
        {'name': 'Kubernetes', 'category': 'tool'},
        {'name': 'AWS', 'category': 'tool'},
        {'name': 'Git', 'category': 'tool'},
        
        # Other
        {'name': 'API Development', 'category': 'domain'},
        {'name': 'Web Development', 'category': 'domain'},
    ]
    
    async with get_db() as db:
        result = await db.execute(select(func.count()).select_from(skills))
        count = result.scalar()
        
        if count > 0:
            print(f"⏭️  Skills already seeded ({count} skills exist)")
            return
        
        await db.execute(insert(skills).values(skill_data))
        print(f"✅ Seeded {len(skill_data)} skills")

async def get_skill_id_map() -> Dict[str, int]:
    """Get all skill IDs as a map for efficient lookup"""
    async with get_db() as db:
        result = await db.execute(select(skills.c.id, skills.c.name))
        return {row[1]: row[0] for row in result.fetchall()}

async def project_exists(title: str, repo_url: Optional[str]) -> bool:
    """Check if project already exists"""
    async with get_db() as db:
        if repo_url:
            result = await db.execute(
                select(projects.c.id).where(projects.c.repo_url == repo_url)
            )
            if result.first():
                return True
        
        result = await db.execute(
            select(projects.c.id).where(projects.c.title == title)
        )
        return result.first() is not None

async def insert_project_with_relations(project_data: Dict, skill_id_map: Dict[str, int], generate_embedding: bool = True) -> Optional[int]:
    """
    Insert project with skills and embeddings in a SINGLE transaction
    """
    extracted_skills = project_data.pop('extracted_skills', [])
    
    if await project_exists(project_data['title'], project_data.get('repo_url')):
        print(f"⏭️  Skipping: {project_data['title']}")
        return None

    async with get_db() as db:
        try:
            # 1. Insert project
            result = await db.execute(
                insert(projects).values(**project_data).returning(projects.c.id)
            )
            project_id = result.scalar()
            
            print(f"✅ Inserted: {project_data['title']} (ID: {project_id}, {project_data['difficulty']})")
            
            # 2. Link skills (in same transaction)
            linked_count = 0
            for skill_name in extracted_skills:
                skill_id = skill_id_map.get(skill_name)
                
                if skill_id:
                    try:
                        # Check if already exists
                        check_result = await db.execute(
                            select(project_skills).where(
                                (project_skills.c.project_id == project_id) &
                                (project_skills.c.skill_id == skill_id)
                            )
                        )
                        
                        if not check_result.first():
                            await db.execute(
                                insert(project_skills).values(
                                    project_id=project_id,
                                    skill_id=skill_id,
                                    is_required=True
                                )
                            )
                            linked_count += 1
                    except Exception as e:
                        print(f"  ⚠️ Error linking {skill_name}: {e}")
            
            if linked_count > 0:
                print(f"  → Linked {linked_count} skills")
            
            # 3. Generate embedding (in same transaction)
            if generate_embedding:
                try:
                    # Check if embedding already exists
                    check_result = await db.execute(
                        select(project_embeddings).where(
                            project_embeddings.c.project_id == project_id
                        )
                    )
                    
                    if not check_result.first():
                        # Build embedding text
                        embedding_text = embedding_generator.build_project_text(project_data)
                        
                        # Generate embedding
                        embedding = embedding_generator.encode(embedding_text)
                        
                        # Store embedding
                        await db.execute(
                            insert(project_embeddings).values(
                                project_id=project_id,
                                embedding=embedding,
                                model_version="all-MiniLM-L6-v2"
                            )
                        )
                        print("  🤖 Generated embedding (384-dim)")
                except Exception as e:
                    print(f"  ⚠️ Error generating embedding: {e}")
            
            return project_id
            
        except Exception as e:
            print(f"❌ Error: {project_data['title']}: {e}")
            return None

# ============================================
# BATCH EMBEDDING GENERATION
# ============================================

async def generate_missing_embeddings():
    """Generate embeddings for projects that don't have them"""
    
    print("\n🤖 Generating missing embeddings...")
    print("=" * 60)
    
    async with get_db() as db:
        # Find projects without embeddings
        result = await db.execute(
            select(projects.c.id, projects.c.title, projects.c.description, 
                   projects.c.topics, projects.c.language, projects.c.difficulty)
            .where(
                ~projects.c.id.in_( # ~condition   # means NOT in subquery
                    select(project_embeddings.c.project_id)
                )
            )
        )
        
        
        # Converts each Row into a dictionary-like object so we can safely
        # access column values using their names (e.g., row['id']) instead of numeric indexes.

        # missing_projects = [dict(row._mapping) for row in result.fetchall()]
        missing_projects = result.mappings().all()
        
        if not missing_projects:
            print("✅ All projects have embeddings!")
            return
        
        print(f"Found {len(missing_projects)} projects without embeddings")
        
        # Generate embeddings in batch
        texts = []
        project_ids = []
        
        for project in missing_projects:
            project_data = {
                'title': project['title'],
                'description': project['description'],
                'topics': project['topics'],
                'language': project['language'],
                'difficulty': project['difficulty']
            }
            texts.append(embedding_generator.build_project_text(project_data))
            project_ids.append(project['id'])
        
        print("\nGenerating embeddings...")
        embeddings = embedding_generator.encode_batch(texts)
        
        # Store embeddings
        print("\nStoring embeddings...")
        # zip() function combines two or more iterables (like lists or tuples) into pairs
        for project_id, embedding in zip(project_ids, embeddings):
            async with get_db() as db:
                try:
                    await db.execute(
                        insert(project_embeddings).values(
                            project_id=project_id,
                            embedding=embedding,
                            model_version="all-MiniLM-L6-v2"
                        )
                    )
                    print(f"✅ Project {project_id}")
                except Exception as e:
                    print(f"❌ Project {project_id}: {e}")
        
        print(f"\n✅ Generated {len(embeddings)} embeddings")

# ============================================
# GITHUB FETCHER
# ============================================

# per_page is the number of results GitHub should return per API request.
def fetch_github_projects(query: str, per_page: int = 10) -> List[Dict]:
    """Fetch repos from GitHub"""
    if not GITHUB_TOKEN:
        print("⚠️ No GitHub token")
        return []
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }
    
# Example Rsesponse: {
#   "total_count": 12345,
#   "incomplete_results": false,
#   "items": [
#     { "id": 1, "name": "repo1", ... },
#     { "id": 2, "name": "repo2", ... },
#     ...
#   ]
# }

# items: list of repos

    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()["items"]
        elif response.status_code == 403:
            print("⚠️ Rate limit. Waiting 60s...")
            time.sleep(60)
            return fetch_github_projects(query, per_page)
        else:
            print(f"❌ Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def process_github_repo(repo: Dict) -> Dict:
    """Process GitHub repo into our format"""
    difficulty = infer_difficulty(repo)
    estimated_hours = estimate_hours(difficulty, repo)
    extracted_skills = extract_skills(repo)
    
    # Safely handle None values
    description = repo.get('description')
    if not description:
        language = repo.get('language') or 'programming'
        description = f"A {language} project"
    
    topics = repo.get('topics') or []
    # Filter out common non-technical topics
    topics = [t for t in topics if t not in ['awesome', 'hacktoberfest']]
    
    return {
        'title': repo.get('name', 'Unnamed Project'),
        'description': description[:500],
        'repo_url': repo.get('html_url'),
        'difficulty': difficulty,
        'topics': topics[:10],
        'estimated_hours': estimated_hours,
        'source': 'github',
        'stars': repo.get('stargazers_count', 0),
        'language': repo.get('language') or 'Unknown',
        'extracted_skills': extracted_skills
    }

# ============================================
# MAIN GENERATION PIPELINE
# ============================================

async def generate_kaggle_data(competitions: bool = True, datasets: bool = True, generate_embeddings: bool = True):
    """Generate Kaggle projects"""
    print("\n🏆 Fetching Kaggle projects...")
    print("=" * 60)
    
    await seed_skills()
    skill_id_map = await get_skill_id_map()
    
    total_inserted = 0
    total_skipped = 0
    
    if competitions:
        print("\n🏅 Fetching Kaggle Competitions...")
        comps = fetch_kaggle_competitions(max_results=15)
        
        for comp in comps:
            processed = process_kaggle_competition(comp)
            result = await insert_project_with_relations(processed, skill_id_map, generate_embedding=generate_embeddings)
            if result:
                total_inserted += 1
            else:
                total_skipped += 1
    
    if datasets:
        print("\n📊 Fetching Kaggle Datasets...")
        dsets = fetch_kaggle_datasets(max_results=15)
        
        for ds in dsets:
            processed = process_kaggle_dataset(ds)
            result = await insert_project_with_relations(processed, skill_id_map, generate_embedding=generate_embeddings)
            if result:
                total_inserted += 1
            else:
                total_skipped += 1
    
    print("\n" + "=" * 60)
    print("📊 KAGGLE SUMMARY")
    print("=" * 60)
    print(f"✅ Projects inserted: {total_inserted}")
    print(f"⏭️  Projects skipped: {total_skipped}")

async def generate_all_data(use_github: bool = True, use_kaggle: bool = True, github_per_query: int = 10, generate_embeddings: bool = True):
    """Main data generation pipeline"""
    print("🚀 Starting data generation...")
    print("=" * 60)
    
    await seed_skills()
    
    print("\n📋 Loading skill mappings...")
    skill_id_map = await get_skill_id_map()
    print(f"✅ Loaded {len(skill_id_map)} skills")
    
    total_inserted = 0
    total_skipped = 0
    
    print("\n📝 Adding curated projects...")
    for proj in CURATED_PROJECTS:
        project_data = {
            **proj,
            'source': 'curated',
            'stars': 0,
            'language': 'Multiple',
        }
        result = await insert_project_with_relations(project_data, skill_id_map, generate_embedding=generate_embeddings)
        if result:
            total_inserted += 1
        else:
            total_skipped += 1
    
    if use_github:
        print("\n🐙 Fetching GitHub projects...")
        for i, query in enumerate(GITHUB_SEARCH_QUERIES, 1):
            print(f"\n[{i}/{len(GITHUB_SEARCH_QUERIES)}] {query}")
            
            repos = fetch_github_projects(query, per_page=github_per_query)
            
            for repo in repos:
                processed = process_github_repo(repo)
                result = await insert_project_with_relations(processed, skill_id_map, generate_embedding=generate_embeddings)
                if result:
                    total_inserted += 1
                else:
                    total_skipped += 1
            
            time.sleep(2)
    
    if use_kaggle:
        print("\n🏆 Fetching Kaggle projects...")
        
        print("\n🏅 Kaggle Competitions...")
        comps = fetch_kaggle_competitions(max_results=10)
        for comp in comps:
            processed = process_kaggle_competition(comp)
            result = await insert_project_with_relations(processed, skill_id_map, generate_embedding=generate_embeddings)
            if result:
                total_inserted += 1
            else:
                total_skipped += 1
        
        print("\n📊 Kaggle Datasets...")
        dsets = fetch_kaggle_datasets(max_results=10)
        for ds in dsets:
            processed = process_kaggle_dataset(ds)
            result = await insert_project_with_relations(processed, skill_id_map, generate_embedding=generate_embeddings)
            if result:
                total_inserted += 1
            else:
                total_skipped += 1
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Projects inserted: {total_inserted}")
    print(f"⏭️  Projects skipped: {total_skipped}")
    
    await show_stats()

async def show_stats():
    """Show database statistics"""
    async with get_db() as db:
        result = await db.execute(select(func.count()).select_from(projects))
        total = result.scalar()
        
        if total == 0:
            print("No projects in database")
            return
        
        result = await db.execute(select(func.count()).select_from(project_embeddings))
        total_embeddings = result.scalar()
        
        result = await db.execute(
            select(projects.c.difficulty, func.count())
            .group_by(projects.c.difficulty)
        )
        by_difficulty = {row[0]: row[1] for row in result}
        
        result = await db.execute(
            select(projects.c.source, func.count())
            .group_by(projects.c.source)
        )
        by_source = {row[0]: row[1] for row in result}
        
        print("\n📊 Database Stats:")
        print(f"  Total projects: {total}")
        print(f"  With embeddings: {total_embeddings} ({total_embeddings/total*100:.1f}%)")
        
        print("\n📊 By Difficulty:")
        for diff in ['beginner', 'intermediate', 'advanced']:
            count = by_difficulty.get(diff, 0)
            print(f"  {diff.capitalize()}: {count}")
        
        print("\n📊 By Source:")
        for source, count in by_source.items():
            print(f"  {source}: {count}")

async def clean_database():
    """Remove all projects"""
    confirm = input("⚠️ Delete ALL projects? Type 'yes': ")
    if confirm.lower() == 'yes':
        try:
            async with get_db() as db:
                await db.execute(delete(project_embeddings))
                await db.execute(delete(project_skills))
                await db.execute(delete(projects))
            print("✅ Database cleaned")
        except Exception as e:
            print(f"❌ Database error: {e}")
            print("\n💡 Make sure PostgreSQL is running and connection details are correct")
    else:
        print("❌ Cancelled")

async def test_connection():
    """Test database connection"""
    try:
        async with get_db() as db:
            result = await db.execute(select(func.count()).select_from(skills))
            count = result.scalar()
            print("✅ Database connection successful!")
            print(f"   Skills table has {count} records")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if PostgreSQL is running")
        print("   2. Verify connection details in .env file")
        print("   3. Ensure database exists and tables are created")
        return False

# ============================================
# CLI
# ============================================

async def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            await test_connection()
        elif command == "clean":
            await clean_database()
        elif command == "stats":
            if await test_connection():
                await show_stats()
        elif command == "quick":
            if await test_connection():
                await generate_all_data(use_github=False, use_kaggle=False, generate_embeddings=True)
        elif command == "full":
            if await test_connection():
                await generate_all_data(use_github=True, use_kaggle=True, github_per_query=15, generate_embeddings=True)
        elif command == "kaggle":
            if await test_connection():
                await generate_kaggle_data(competitions=True, datasets=True, generate_embeddings=True)
        elif command == "kaggle-comps":
            if await test_connection():
                await generate_kaggle_data(competitions=True, datasets=False, generate_embeddings=True)
        elif command == "kaggle-datasets":
            if await test_connection():
                await generate_kaggle_data(competitions=False, datasets=True, generate_embeddings=True)
        elif command == "embeddings":
            if await test_connection():
                await generate_missing_embeddings()
        elif command == "no-embeddings":
            if await test_connection():
                await generate_all_data(use_github=True, use_kaggle=False, github_per_query=10, generate_embeddings=False)
        else:
            print("Unknown command. Available commands:")
            print("  test              - Test database connection")
            print("  clean             - Remove all projects")
            print("  stats             - Show database statistics")
            print("  quick             - Only curated projects with embeddings")
            print("  full              - GitHub + Kaggle + curated with embeddings")
            print("  kaggle            - Only Kaggle projects (competitions + datasets)")
            print("  kaggle-comps      - Only Kaggle competitions")
            print("  kaggle-datasets   - Only Kaggle datasets")
            print("  embeddings        - Generate missing embeddings")
            print("  no-embeddings     - Generate data without embeddings")
    else:
        if await test_connection():
            await generate_all_data(use_github=True, use_kaggle=False, github_per_query=10, generate_embeddings=True)

if __name__ == "__main__":
    asyncio.run(main())