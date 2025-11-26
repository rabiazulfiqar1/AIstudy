-- ============================================
-- 5. Batch Generate Project Embeddings
-- ============================================
-- Helper function to check which projects need embeddings

CREATE OR REPLACE FUNCTION get_projects_needing_embeddings()
RETURNS TABLE(
    project_id INTEGER,
    embedding_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id,
        CONCAT(
            p.title, '. ',
            p.description, '. ',
            'Topics: ', ARRAY_TO_STRING(p.topics, ', '), '. ',
            'Language: ', p.language, '. ',
            'Difficulty: ', p.difficulty
        ) as text
    FROM projects p
    LEFT JOIN project_embeddings pe ON p.id = pe.project_id
    WHERE pe.project_id IS NULL;
END;
$$;

-- ============================================
-- 6. Clean Old Interactions
-- ============================================
-- Archive or remove very old view interactions to keep table lean

CREATE OR REPLACE FUNCTION clean_old_interactions(
    p_days_threshold INTEGER DEFAULT 180
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old 'viewed' interactions only (keep bookmarks, started, completed)
    DELETE FROM user_project_interactions
    WHERE interaction_type = 'viewed'
      AND created_at < NOW() - (p_days_threshold || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$;

-- ============================================
-- 7. Get User Activity Summary
-- ============================================
-- Comprehensive activity statistics for a user

CREATE OR REPLACE FUNCTION get_user_activity_summary(
    p_user_id UUID
)
RETURNS TABLE(
    total_interactions BIGINT,
    projects_viewed BIGINT,
    projects_bookmarked BIGINT,
    projects_started BIGINT,
    projects_completed BIGINT,
    avg_rating NUMERIC,
    total_learning_hours INTEGER,
    skills_count BIGINT,
    most_active_category TEXT,
    recent_activity_7d BIGINT,
    completion_rate NUMERIC
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH interaction_stats AS (
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE interaction_type = 'viewed') as viewed,
            COUNT(*) FILTER (WHERE interaction_type = 'bookmarked') as bookmarked,
            COUNT(*) FILTER (WHERE interaction_type = 'started') as started,
            COUNT(*) FILTER (WHERE interaction_type = 'completed') as completed,
            AVG(rating) FILTER (WHERE rating IS NOT NULL) as avg_rat,
            COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as recent
        FROM user_project_interactions
        WHERE user_id = p_user_id
    ),
    learning_stats AS (
        SELECT 
            COALESCE(SUM(p.estimated_hours), 0) as hours
        FROM user_project_interactions upi
        JOIN projects p ON upi.project_id = p.id
        WHERE upi.user_id = p_user_id
          AND upi.interaction_type IN ('started', 'completed')
    ),
    skill_stats AS (
        SELECT 
            COUNT(*) as skill_count,
            (
                SELECT s.category
                FROM user_skills us
                JOIN skills s ON us.skill_id = s.id
                WHERE us.user_id = p_user_id
                GROUP BY s.category
                ORDER BY COUNT(*) DESC
                LIMIT 1
            ) as top_category
        FROM user_skills
        WHERE user_id = p_user_id
    )
    SELECT 
        ist.total,
        ist.viewed,
        ist.bookmarked,
        ist.started,
        ist.completed,
        ROUND(ist.avg_rat, 2),
        lst.hours::INTEGER,
        sst.skill_count,
        sst.top_category,
        ist.recent,
        CASE 
            WHEN ist.started > 0 THEN 
                ROUND((ist.completed::NUMERIC / ist.started) * 100, 1)
            ELSE 0
        END
    FROM interaction_stats ist, learning_stats lst, skill_stats sst;
END;
$$;