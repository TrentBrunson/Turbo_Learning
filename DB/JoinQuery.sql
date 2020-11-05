SELECT * FROM TEX_PBP;
SELECT * FROM Tex_Opp_score;

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

-- Final Table
SELECT * FROM Tex_combined_final;
ALTER TABLE Tex_combined_final ADD CONSTRAINT pk_Tex_combined_final 
	PRIMARY KEY (playId);

rollback;
commit;