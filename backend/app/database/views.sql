-- views.sql

DO $$ BEGIN

    -- Popular projects by difficulty
    CREATE OR REPLACE VIEW popular_projects_by_difficulty AS
    SELECT 
        difficulty,
        COUNT(*) AS project_count,
        AVG(stars) AS avg_stars,
        AVG(estimated_hours) AS avg_hours
    FROM projects
    GROUP BY difficulty;

    -- Most recommended skill combinations
    CREATE OR REPLACE VIEW top_skill_combinations AS
    SELECT 
        ps1.skill_id AS skill_1,
        ps2.skill_id AS skill_2,
        s1.name AS skill_1_name,
        s2.name AS skill_2_name,
        COUNT(DISTINCT ps1.project_id) AS project_count
    FROM project_skills ps1
    JOIN project_skills ps2 
        ON ps1.project_id = ps2.project_id 
        AND ps1.skill_id < ps2.skill_id
    JOIN skills s1 
        ON ps1.skill_id = s1.id
    JOIN skills s2 
        ON ps2.skill_id = s2.id
    GROUP BY ps1.skill_id, ps2.skill_id, s1.name, s2.name
    ORDER BY project_count DESC
    LIMIT 20;

END $$;
