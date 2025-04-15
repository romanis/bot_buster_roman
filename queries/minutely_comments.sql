

with comments as (
    select 
        c.id as comment_id
        , c.published_at
        , c.user_id
        , c.video_id
        , c.like_count
        , c.reply_count
    from youtube_comments c
    where TRUE
        AND mod(user_id, 100) <= 100
        AND c.published_at >= '{{start_date}}'
        AND c.published_at < '{{end_date}}'
)

, comments_by_minute as(
    select 
        c.comment_id
        , u.username
        , u.id as user_id
        , c.video_id
        , ch.title as channel_title
        , date_format(c.published_at, '{{interval_format}}') as period
        , count(*) as number_comments
        , sum(c.like_count) as total_likes
        , sum(c.reply_count) as total_replies
        , max(c.like_count) as max_likes
        , max(c.reply_count) as max_replies
        -- , count(*) OVER (PARTITION BY c.user_id, date_format(c.published_at, '{{interval_format}}')) as all_comments_per_period
        -- , count(*) OVER (PARTITION BY c.user_id, date_format(c.published_at, '{{interval_format}}'), v.channel_id) as comments_per_period_this_channel
    from comments c
    left join youtube_videos v on TRUE
        AND c.video_id = v.id
    LEFT JOIN youtube_channels ch on TRUE
        AND v.channel_id = ch.id
    left join youtube_users u on TRUE
        AND c.user_id = u.id
    group by 1,2,3,4,5,6
)
, all_comments_per_period_channel AS (
    SELECT
        user_id
        , username
        , channel_title
        , SUBSTR(period, 1,7) as month
        , period
        , sum(number_comments) AS number_comments
        , sum(total_likes) AS total_likes
        , sum(total_replies) AS total_replies
        , max(max_likes) as max_likes
        , max(max_replies) as max_replies
        
    FROM comments_by_minute
    GROUP BY 1,2,3,4,5
)


select 
    c.username
    , c.user_id
    , channel_title as channel_title
    , c.month
    , sum(c.number_comments) as number_comments_all_time
    , count(DISTINCT c.period) as number_periods_with_comments
    -- , max(c.all_comments_per_period) as max_CPP_all_CH
    , max(c.number_comments) as max_CPP_this_CH
    , sum(c.total_likes) as total_likes_all_time
    , sum(c.total_replies) as total_replies_all_time
    , max(c.max_likes) as max_likes
    , max(c.max_replies) as max_replies
    , 1.0*sum(total_likes)/sum(number_comments) as mean_likes_per_comment
    , 1.0*sum(total_replies)/sum(number_comments) as mean_replies_per_comment
    , 1.0*sum(total_likes)/sum(total_replies) as mean_likes_per_reply
from all_comments_per_period_channel c 
where TRUE
group by 1,2,3,4
order by 4 ASC,7 DESC
