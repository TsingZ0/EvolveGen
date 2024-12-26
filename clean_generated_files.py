import shutil
  
directory_path = 'dataset/generated/'
  
# Forcefully delete the directory and its contents
try:
    shutil.rmtree(directory_path)
    print(f'{directory_path} deleted.')
except:
    print(f'{directory_path} already deleted.')


directory_path = 'dataset/train/'
  
# Forcefully delete the directory and its contents
try:
    shutil.rmtree(directory_path)
    print(f'{directory_path} deleted.')
except:
    print(f'{directory_path} already deleted.')