-- users per interest, kept by trigger so cascaded deletes (user erasure) stay correct; drives IDF
ALTER TABLE interests ADD COLUMN user_count integer NOT NULL DEFAULT 0;
UPDATE interests i SET user_count = (SELECT count(*) FROM user_interests ui WHERE ui.interest_id = i.id);

CREATE FUNCTION interest_user_count() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE interests SET user_count = user_count + 1 WHERE id = NEW.interest_id;
    ELSE
        UPDATE interests SET user_count = user_count - 1 WHERE id = OLD.interest_id;
    END IF;
    RETURN NULL;
END $$;

CREATE TRIGGER user_interests_count AFTER INSERT OR DELETE ON user_interests
FOR EACH ROW EXECUTE FUNCTION interest_user_count();
