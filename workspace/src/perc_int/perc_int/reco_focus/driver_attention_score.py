class DriverAttentionScorer:
    
    def compute_score(self,
                initial_score,
                penalty):
        final_score = initial_score - (1 * penalty)
        score = max(0.0, min(1.0, final_score))
        #return score
        #print(score)
        if score >= 0.5 :
            return 100
        elif score >= 0.3 :
            return 50
        else:
            return 0