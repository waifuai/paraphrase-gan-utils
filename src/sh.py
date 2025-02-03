import subprocess
import os

def sh(command):
  """
  Runs a shell command
  :param command: A string of the shell command to execute. eg "ls -lat data/*"
  :return: None
  """
  subprocess.call(command.split(" "), cwd=os.path.dirname(os.path.realpath(__file__)))

sh("ls -lat")
