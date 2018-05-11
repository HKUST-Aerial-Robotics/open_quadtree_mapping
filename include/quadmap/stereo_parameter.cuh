// search range
#define MAX_DEP 100
#define MIN_DEP 0.5
#define MIN_GRAIDIENT 5
#define MIN_INV_DEPTH 0.01
#define MAX_INV_DEPTH 2.0
#define STEP_INV_DEPTH (MAX_INV_DEPTH-MIN_INV_DEPTH)/63.0
#define IDEPTH_INITIAL 1.0
#define VARIANCE_MAX 4.0

#define MAX_DIFF_CONSTANT (40.0f*40.0f)
#define MAX_DIFF_GRAD_MULT (0.5f*0.5f)
#define REG_DIST_VAR (0.075f*0.075f)
#define VAL_SUM_MIN_FOR_CREATE (24) // minimal summed validity over 5x5 region to create a new hypothesis for non-blacklisted pixel (hole-filling)
#define VAL_SUM_MIN_FOR_KEEP (24) // minimal summed validity over 5x5 region to keep hypothesis (regularization)
#define VAL_SUM_MIN_FOR_UNBLACKLIST (100) // if summed validity surpasses this, a pixel is un-blacklisted.
#define MIN_BLACKLIST -1
#define MAX_SEARCH 100
#define MAX_ERROR_STEREO (1300.0f) // maximal photometric error for stereo to be successful (sum over 9 squared intensity differences)
#define MIN_DISTANCE_ERROR_STEREO (1.5f) // minimal multiplicative difference to second-best match to not be considered ambiguous.

#define VALIDITY_COUNTER_INC 5		// validity is increased by this on sucessfull stereo
#define VALIDITY_COUNTER_DEC 5		// validity is decreased by this on failed stereo
#define VALIDITY_COUNTER_INITIAL_OBSERVE 5	// initial validity for first observations
#define VALIDITY_COUNTER_MAX (5.0f)		// validity will never be higher than this
#define VALIDITY_COUNTER_MAX_VARIABLE (250.0f)		// validity will never be higher than this

#define SUCC_VAR_INC_FAC (1.01f) // before an ekf-update, the variance is increased by this factor.
#define FAIL_VAR_INC_FAC 1.1f // after a failed stereo observation, the variance is increased by this factor.

//for depth extract
// #define PRIOR_COST_SCALE 0.02
// #define TRUNCATE_COST 5.0
#define PRIOR_COST_SCALE 0.1
#define TRUNCATE_COST 0.5

#define update_debug false

//for depth upsample
#define use_fabs_distence false
#define upsample_sigma 400.0
#define upsample_lambda 10.0