-- triggers.sql

-- Auto-update last_accessed on cache reads
CREATE OR REPLACE FUNCTION update_cache_access()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.last_accessed = NOW();
    RETURN NEW;
END;
$func$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_cache_access
BEFORE UPDATE ON video_cache
FOR EACH ROW
EXECUTE FUNCTION update_cache_access();


-- Trigger on profile updates
CREATE OR REPLACE FUNCTION log_profile_update()
RETURNS TRIGGER AS $func$
BEGIN
    INSERT INTO user_activity_log (user_id, activity_type, details)
    VALUES (
        NEW.user_id,
        'profile_updated',
        json_build_object(
            'skill_level', NEW.skill_level,
            'interests_count', array_length(NEW.interests, 1)
        )
    );
    RETURN NEW;
END;
$func$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_profile_update
AFTER UPDATE ON user_profiles
FOR EACH ROW
EXECUTE FUNCTION log_profile_update();
