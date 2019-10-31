#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ConvertToMP4.py
# 
# 
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
# 
# 
import glob
import os
from optparse import OptionParser

class ConvertToMP4:
    def __init__(self):
        self.webmFiles = []
       
    def get_all_wbm_files_from_directory(self, directory_name):
        print("Scanning directory: " + directory_name)
        if len(directory_name) == 0:
            print('ERROR: Source directory name cannot be NIL/NULL.')
            print('Provide a valid directory name to proceed!')
            return
        os.chdir(directory_name)
        for files in os.listdir('.'):
            if files.endswith('.webm'):
                print('convert file: ' + files)
                self.webmFiles.append(files)

    def convert_wbm_file_to_mp4_file(self):
        # check to see if the list is empty, if not proceed
        if len(self.webmFiles) <= 0:
            print("No files to convert!")
            return
        for webm_file in self.webmFiles:
            mp4_file = webm_file.replace('.webm','.mp4')
            cmd_string = 'ffmpeg -i "' + webm_file + '" "' + mp4_file + '"'
            print('converting ' + webm_file + ' to ' + mp4_file)
            os.system(cmd_string)

def main():
    usage = "usage: %prog -d <source directory for webm files>"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--sourcedirectory", action="store",
        type="string", dest="sourcedirectory", default="./",
          help="source directory where all the webm files are stored!")
    (options, args) = parser.parse_args()
    webm_to_mp4 = ConvertToMP4()
    webm_to_mp4.get_all_wbm_files_from_directory(options.sourcedirectory)
    webm_to_mp4.convert_wbm_file_to_mp4_file()
    return 0

if __name__ == '__main__':
    main()
