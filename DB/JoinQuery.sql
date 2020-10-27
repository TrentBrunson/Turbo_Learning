SELECT * FROM Tex_Opp_score;
SELECT * FROM TEX_PBP;

-- Add Columns for Texas Score and Opponent Score
ALTER TABLE Tex_Opp_Score
ADD TexScore INT, 
ADD OppScore INT;

-- Puts Texas Score in appropiate column
Update Tex_Opp_Score
SET TexScore = 
	CASE homeAbbr
    	WHEN 'TEX' THEN homeScore
    	ELSE awayScore
END;

-- Puts Opponent score in appropiate column
Update Tex_Opp_Score
SET OppScore = 
	CASE homeAbbr
		WHEN 'TEX' THEN awayScore
		ELSE homeScore
END;

-- Join new columns into Play by Play Table
Select pbp.*,
	s.TexScore,
	s.OppScore
INTO Tex_combined_final
FROM tex_pbp as pbp
LEFT JOIN tex_opp_score as s
ON pbp.playId = s.playId;

SELECT * from Tex_combined_final;

DROP TABLE;

rollback;
commit;