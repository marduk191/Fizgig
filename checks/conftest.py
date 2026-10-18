import os
import sys

# the checks import Fizgig from the repo's src/, without installing it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
