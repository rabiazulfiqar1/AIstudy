const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api'

interface Skill {
  skill_id: string
  name?: string
  proficiency: string
}

interface UserProfileData {
  bio?: string
  skillLevel?: string
  interests?: string[]
  githubUsername?: string
  preferredProjectTypes?: string[]
  skills?: Skill[]
}

interface UserProfileResponse {
  user_id: string
  skill_level?: string
  interests?: string[]
  bio?: string
  github_username?: string
  preferred_project_types?: string[]
  skills?: Array<{
    skill_id: string
    name: string
    proficiency: string
  }>
}

interface ProjectDetail {
  id: number;
  title: string;
  description: string;
  repo_url?: string;
  difficulty: string;
  topics: string[];
  estimated_hours?: number;
  source: string;
  stars?: number;
  language?: string;
  skills: Array<{
    name: string;
    category: string;
    is_required: boolean;
  }>;
  match_analysis?: {
    score: number;
    matching_skills: string[];
    missing_skills: string[];
    reason: string;
    semantic_similarity: number;
  };
}

interface SearchProject {
  id: number;
  title: string;
  description: string;
  difficulty: string;
  source: string;
  match_score?: number;
  participants?: number;
  prize?: string;
}

interface SearchResponse {
  projects: SearchProject[];
  count: number;
  search_type: string;
  filters: {
    difficulty?: string;
    source?: string;
  };
}

// Get user profile from backend
export async function getUserProfile(userId: string): Promise<UserProfileResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/users/${userId}/profile`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      if (response.status === 404) {
        return null // Profile doesn't exist yet
      }
      throw new Error('Failed to fetch profile')
    }

    return await response.json()
  } catch (error) {
    console.error('Error fetching user profile:', error)
    return null
  }
}

// Create or update user profile
export async function updateUserProfile(userId: string, profileData: UserProfileData): Promise<boolean> {
  try {
    // Transform frontend data to match backend schema
    const payload = {
      skill_level: profileData.skillLevel,
      interests: profileData.interests || [],
      bio: profileData.bio || '',
      github_username: profileData.githubUsername || '',
      preferred_project_types: profileData.preferredProjectTypes || [],
      skills: (profileData.skills || []).map(skill => ({
        skill_id: skill.skill_id,
        proficiency: skill.proficiency,
      })),
    }

    const response = await fetch(`${API_BASE_URL}/users/${userId}/profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      const errorData = await response.json()
      console.error('Failed to update profile:', errorData)
      throw new Error(errorData.detail || 'Failed to update profile')
    }

    return true
  } catch (error) {
    console.error('Error updating user profile:', error)
    throw error
  }
}

// Transform backend profile data to frontend format
export function transformProfileFromBackend(backendProfile: UserProfileResponse) {
  return {
    bio: backendProfile.bio || '',
    skillLevel: backendProfile.skill_level || 'intermediate',
    interests: backendProfile.interests || [],
    githubUsername: backendProfile.github_username || '',
    preferredProjectTypes: backendProfile.preferred_project_types || [],
    skills: (backendProfile.skills || []).map(skill => ({
      id: skill.skill_id,
      name: skill.name,
      proficiency: parseInt(skill.proficiency) || 3,
    })),
  }
}

/**
 * Search projects with filters
 */
export async function searchProjects(
  query?: string,
  difficulty?: string,
  source?: string,
  useSemanticSearch: boolean = true,
  limit: number = 20
): Promise<SearchProject[]> {
  try {
    const params = new URLSearchParams();
    
    if (query) params.append('q', query);
    if (difficulty) params.append('difficulty', difficulty.toLowerCase());
    if (source) params.append('source', source.toLowerCase());
    params.append('use_semantic', useSemanticSearch.toString());
    params.append('limit', limit.toString());

    const response = await fetch(`${API_BASE_URL}/projects/search?${params}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const data: SearchResponse = await response.json();
    return data.projects;
  } catch (error) {
    console.error('Error searching projects:', error);
    throw error;
  }
}

/**
 * Get project details by ID
 */
export async function getProjectDetail(
  projectId: string | number,
  userId?: string
): Promise<ProjectDetail> {
  try {
    const params = new URLSearchParams();
    if (userId) params.append('user_id', userId);

    const response = await fetch(
      `${API_BASE_URL}/projects/${projectId}${params.toString() ? `?${params}` : ''}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Project not found');
      }
      throw new Error(`Failed to fetch project: ${response.statusText}`);
    }

    const data: ProjectDetail = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching project detail:', error);
    throw error;
  }
}

/**
 * Get Kaggle competitions
 */
export async function getKaggleCompetitions(
  query?: string,
  limit: number = 20
): Promise<SearchProject[]> {
  return searchProjects(query, undefined, 'kaggle_competition', false, limit);
}

/**
 * Get Kaggle datasets
 */
export async function getKaggleDatasets(
  query?: string,
  limit: number = 20
): Promise<SearchProject[]> {
  return searchProjects(query, undefined, 'kaggle_dataset', false, limit);
}

/**
 * Get competition/dataset detail
 * This reuses the project detail endpoint since Kaggle items are stored as projects
 */
export async function getKaggleItemDetail(
  itemId: string | number,
  userId?: string
): Promise<ProjectDetail> {
  return getProjectDetail(itemId, userId);
}