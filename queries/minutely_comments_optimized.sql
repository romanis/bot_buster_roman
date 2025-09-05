-- Optimized version of minutely_comments.sql with better performance
-- Added proper indexing hints and query structure improvements

WITH comments AS (
    SELECT 
        c.id AS comment_id,
        c.published_at,
        c.user_id,
        c.video_id,
        c.like_count,
        c.reply_count
    FROM youtube_comments c
    WHERE c.published_at >= '{{start_date}}'
        AND c.published_at < '{{end_date}}'
        AND MOD(c.user_id, 100) <= {{user_id_mod|default(100)}}
    -- Assume index on (published_at, user_id) exists for optimal performance
),

video_channel_info AS (
    SELECT DISTINCT
        v.id AS video_id,
        v.channel_id,
        ch.title AS channel_title
    FROM youtube_videos v
    INNER JOIN youtube_channels ch ON v.channel_id = ch.id
    WHERE v.id IN (SELECT DISTINCT video_id FROM comments)
),

user_info AS (
    SELECT 
        u.id AS user_id,
        u.username
    FROM youtube_users u
    WHERE u.id IN (SELECT DISTINCT user_id FROM comments)
),

comments_by_period AS (
    SELECT 
        c.user_id,
        u.username,
        c.video_id,
        vci.channel_title,
        DATE_FORMAT(c.published_at, '{{interval_format}}') AS period,
        SUBSTR(DATE_FORMAT(c.published_at, '{{interval_format}}'), 1, 7) AS month,
        COUNT(*) AS number_comments,
        SUM(c.like_count) AS total_likes,
        SUM(c.reply_count) AS total_replies,
        MAX(c.like_count) AS max_likes,
        MAX(c.reply_count) AS max_replies
    FROM comments c
    INNER JOIN user_info u ON c.user_id = u.user_id
    INNER JOIN video_channel_info vci ON c.video_id = vci.video_id
    GROUP BY c.user_id, u.username, c.video_id, vci.channel_title, 
             DATE_FORMAT(c.published_at, '{{interval_format}}'),
             SUBSTR(DATE_FORMAT(c.published_at, '{{interval_format}}'), 1, 7)
),

aggregated_stats AS (
    SELECT
        user_id,
        username,
        channel_title,
        month,
        SUM(number_comments) AS number_comments_all_time,
        COUNT(DISTINCT period) AS number_periods_with_comments,
        MAX(number_comments) AS max_CPP_this_CH,
        SUM(total_likes) AS total_likes_all_time,
        SUM(total_replies) AS total_replies_all_time,
        MAX(max_likes) AS max_likes,
        MAX(max_replies) AS max_replies
    FROM comments_by_period
    GROUP BY user_id, username, channel_title, month
)

SELECT 
    username,
    user_id,
    channel_title,
    month,
    number_comments_all_time,
    number_periods_with_comments,
    max_CPP_this_CH,
    total_likes_all_time,
    total_replies_all_time,
    max_likes,
    max_replies,
    CASE 
        WHEN number_comments_all_time > 0 
        THEN ROUND(1.0 * total_likes_all_time / number_comments_all_time, 4)
        ELSE 0 
    END AS mean_likes_per_comment,
    CASE 
        WHEN number_comments_all_time > 0 
        THEN ROUND(1.0 * total_replies_all_time / number_comments_all_time, 4)
        ELSE 0 
    END AS mean_replies_per_comment,
    CASE 
        WHEN total_replies_all_time > 0 
        THEN ROUND(1.0 * total_likes_all_time / total_replies_all_time, 4)
        ELSE NULL 
    END AS mean_likes_per_reply
FROM aggregated_stats
WHERE number_comments_all_time >= {{min_comments|default(1)}}
ORDER BY month ASC, max_CPP_this_CH DESC

-- Recommended indexes for optimal performance:
-- CREATE INDEX idx_comments_date_user ON youtube_comments(published_at, user_id);
-- CREATE INDEX idx_comments_video ON youtube_comments(video_id);
-- CREATE INDEX idx_videos_channel ON youtube_videos(id, channel_id);
-- CREATE INDEX idx_users_lookup ON youtube_users(id);