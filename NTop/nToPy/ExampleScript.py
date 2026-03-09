import ntopcl
import os

# string of the current directory grabbed using the os module 
Current_Directory = os.path.dirname(os.path.realpath('__file__'))
Current_Directory = '"'+ Current_Directory + '"'



# EXAMPLE 1
# ntopcl.numtext(GUI,SAVE,Inputs,nTop file)

Inputs = {
       "output file path" : Current_Directory,
       "Length"     :       4.000, 
       "Width"      :       5.000,
       "Height"     :       3.000
       }
#ntopcl.numtext(0,0,Inputs,'Example_NUMTEXT.ntop')



# EXAMPLE 2
# ntopcl.jsontemplate(nTop file)
# Creates input and output JSON template files
#ntopcl.jsontemplate('Example_JSONTEMPLATE.ntop')



# EXAMPLE 3
# ntopcl.json inputs: (GUI, SAVE, input json file, output json file, nTop file)
# Runs nTop, without GUI or saving, based on an input from in.json, outputting to out.json 
#ntopcl.json(0,0,'in.json','out.json','Example_JSON.ntop')