"""
To handle multiple globals_?.py.
"""
import argparse
import os.path as osp
import imp
import sys
print 'ddgdgdgdgdgdg', sys.argv

dirname = osp.abspath(osp.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='%s/globals.py' % dirname, help=
        """choose the globals_?.py file. defualt=globals.py""")

#ignore unknown arguments
args, unknown = parser.parse_known_args() 

args = vars(args)
config = args['config']
print 'Loading globas from config file:', config

# import this for use global variables
# from config_loader import globals as g_
globals = imp.load_source('globals', config)
