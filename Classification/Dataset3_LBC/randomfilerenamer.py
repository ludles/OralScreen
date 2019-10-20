#!/usr/bin/python3

# Copyright © 2015 Ole Rickowski
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.

import os,sys,random,string

# Print help
def usage():
	print("randomrenamer: Renames every File in a given folder to a random string. The file extension is kept.")
	print("Usage:", str(sys.argv[0]), "<Folder>")

# Print help if not executed correctly
def checkusage():
	if len(sys.argv) == 2 and os.path.isdir(str(sys.argv[1])):
		dorename(correctpath(str(sys.argv[1])))
	else:
		usage()

# Append / to path if not given
def correctpath(path):
	if not path.endswith("/"):
		path=path+"/"
	return path

# Rename those files
def dorename(path):
	print("Begin renaming...")
	# Rename everything in target folder
	for this in os.listdir(path):
		# print("Renaming:", this)
		#Is target a file?
		if os.path.isfile(path+this):
			#Get a random name including the previous file extension
			rand=createrandname(path,os.path.splitext(this)[1])
			#Do the rename
			os.rename(path+this, path+rand)
			# print("\t->", rand)
		else:
			#Not a file
			print("\tSkipped:", "'"+this+"'", "Target is not a file")
	print("Finished!")

# Create a random file name and make sure the target file wouldn't exist
def createrandname(path,ext):
	check=False
	while check == False:
		rand="".join(random.sample(string.ascii_lowercase+string.digits,20))+ext
		check=True if not os.path.exists(path+rand) else False
	return rand

# Check the usage and continue...
checkusage()
