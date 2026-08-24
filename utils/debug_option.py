DEBUG = False
DEBUG_GRADIENT = False
LOG_GPU_MEMORY = False

# SP numerical-equivalence check: in DEBUG mode with SP=1, generate 1/N of the noise
# and repeat it N times. Set this to the target SP size you want to compare against
# (e.g. 3 to compare SP=3 vs SP=1).
DEBUG_SP_VERIFY_SIZE = 3