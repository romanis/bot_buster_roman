

with comments as (
    select 
        c.id as comment_id
        , c.published_at
        , c.user_id
        , c.video_id
        , c.like_count
        , c.reply_count
        , c.text
    from youtube_comments c
    where TRUE
        AND mod(user_id, 100) <= 100
        AND c.published_at >= '{{start_date}}'
        AND c.published_at < '{{end_date}}'
        AND c.user_id in ({{user_ids}})
)

select * from comments
