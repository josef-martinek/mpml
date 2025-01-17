import subprocess
import os
import shutil
import logging


# Function to get the current Git commit hash
def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return commit_hash
    except Exception as e:
        raise RuntimeError(f"Unable to retrieve Git commit hash: {e}")
    

def save_commit_hash(folder, git_commit):
    os.makedirs(folder, exist_ok=True)
    git_commit_file = folder + "git_commit.txt"
    with open(git_commit_file, "w") as f:
        f.write(git_commit)


def copy_settings_to_output(output_folder, settings_folder, main_file):
    """
    Copies the contents of the settings folder into the output folder.
    
    Parameters:
        output_folder (str): The directory where the settings should be copied to.
        settings_folder (str): The directory containing the settings to be copied.
    """
    try:

        # Ensure the output folder exists
        if os.path.exists(output_folder):
            # Warn the user and delete all contents in the output folder
            logging.warning(f"Output folder '{output_folder}' exists, all contents will be deleted.")
            shutil.rmtree(output_folder)  # Remove all contents of the folder          
        os.makedirs(output_folder, exist_ok=True)
        
        # Check if the settings folder exists
        if not os.path.exists(settings_folder):
            raise RuntimeError(f"Settings folder '{settings_folder}' does not exist.")
        
        # Copy the settings folder into the output folder
        settings_output_path = os.path.join(output_folder, settings_folder)
        main_file_output_path = os.path.join(output_folder, main_file)
        
        shutil.copytree(settings_folder, settings_output_path)
        shutil.copyfile(main_file, main_file_output_path)
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while copying the settings: {e}")
    

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def clear_fenics_cache():
    fenics_cache = os.path.expanduser("~/.cache/fenics")
    if os.path.exists(fenics_cache):
        shutil.rmtree(fenics_cache)