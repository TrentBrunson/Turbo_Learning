-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE TEX_PBP (
    playId varchar   NOT NULL,
    gameId int   NOT NULL,
    year int   NOT NULL,
    week int   NOT NULL,
    offenseAbbr varchar   NOT NULL,
    defenseAbbr varchar   NOT NULL,
    quarter int   NOT NULL,
    clock time   NOT NULL,
    type varchar   NOT NULL,
    down int   NOT NULL,
    distance int   NOT NULL,
    yardLine int   NOT NULL,
    yardsGained int   NOT NULL,
    CONSTRAINT pk_TEX_PBP PRIMARY KEY (
        playId
     )
);

CREATE TABLE TEX_Opp_Score (
    playId varchar   NOT NULL,
    homeAbbr varchar   NOT NULL,
    awayAbbr varchar   NOT NULL,
    homeScore int   NOT NULL,
    awayScore int   NOT NULL,
    CONSTRAINT pk_TEX_Opp_Score PRIMARY KEY (
        playId
     )
);

ALTER TABLE TEX_Opp_Score ADD CONSTRAINT fk_TEX_Opp_Score_playId FOREIGN KEY(playId)
REFERENCES TEX_PBP (playId);

select * from TEX_PBP;
select * from TEX_Opp_Score;

drop table TEX_PBP;
drop table TEX_Opp_Score;

commit;
rollback;

