# def login(email,password):
#     
#     import os
#     import subprocess
#     
#     
#     exePath = r"C:/Program Files/nTopology/nTop Platform/nTopCL.exe"
#     arguments = [exePath]
#     #arguments = [r"C:/Program Files/nTopology/nTop Platform/nTopCL.exe -u"]
#     arguments.append(email)
#     arguments.append("-w ")
#     arguments.append(password)
#     print(arguments)
#     subprocess.call(arguments)
#     #subprocess.call("C:/Program Files/nTopology/nTop Platform/ntopcl.exe -u",email," -w",password)
#     
def jsontemplate(ntopfile):
    
    import subprocess
    exePath = r'ntopcl.exe'
    ## Put together string that calls nTop with input and output JSON databases
    arguments = [exePath]
    arguments.append("-t")
    arguments.append(ntopfile)
    subprocess.call(arguments)
    
    
def json(GUI,save,jsonin,jsonout,ntopfile):
    # ======================================= #
    # ========== nTop Platform API ========== #
    # ======== =json input method============ #
    # https://support.ntopology.com/hc/en-us/articles/360052703693-Running-nTop-Command-Line-in-Python-scripts
    
    import os
    import subprocess


    # ================================ Inputs ================================ #
    
    # WARNING:  There must be NO SPACES in the directory/filename that your .nTop file resides in.
    # WARNING:  There must be NO SPACES in the directory/filename that your results are being output to.
    # exePath describes the path to where your nTop Platform application has been installed.
    # nTopFilePath describes the path to where your target nTop file is.
    #   • If it is in the same folder as this script, just the file name is required.
    # Current_Directory can be used to locate the output files from nTop Platform.
    Current_Directory = os.path.dirname(os.path.realpath('__file__'))

    if GUI:
        exePath = r'ntop.exe'
    else:
        exePath = r'ntopcl.exe'
        


    ## Put together string that calls nTop with input and output JSON databases
    arguments = [exePath]
        
    if save:
        arguments.append("-s -j")
    else:
        arguments.append("-j")
        
    arguments.append(jsonin)
    

    arguments.append("-o")
    arguments.append(jsonout)

    
    arguments.append(ntopfile)
    #arguments.
        
    arguments = " ".join(arguments)

    print('Starting...')
    print('')
    # run nTop
    print(arguments)
    subprocess.call(arguments)
    
def numtext(GUI,save,Inputs,nTopFile):
    
    import os
    import subprocess
    
    Current_Directory = os.path.dirname(os.path.realpath('__file__'))

    if GUI:
        exePath = r'ntop.exe'
    else:
        exePath = r'ntopcl.exe'
    
    Argument_Values = [exePath]
    
    Argument_String = [r"%s"]

    # The formmating below automatically adds the unit 'mm' to the numeric inputs
    Real_Input   = r"-i %0.6f"
    String_Input = r"-i %s"

    for key in Inputs:
        if type(Inputs[key]) is float:
            Argument_String.append(Real_Input)
            Argument_Values.append(Inputs[key])
        if type(Inputs[key]) is str:
            Argument_String.append(String_Input)
            Argument_Values.append(Inputs[key])
        else:
            pass
        
    if save:
        Argument_String.append("-s")  
        
    Argument_String.append(r"%s")
    Argument_Values.append(nTopFile)
    
    AS = " ".join(Argument_String)

    arguments = (AS % (*Argument_Values,))
    print(arguments)
    subprocess.call(arguments)

