import subprocess
with open('err.txt', 'w') as f:
    subprocess.run(['cargo', 'check'], stderr=f, stdout=f)
