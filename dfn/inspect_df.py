import df
import df.enhance
import inspect

with open("api_inspect.txt", "w") as f:
    f.write("DF DIR:\n")
    f.write(str(dir(df)) + "\n\n")
    f.write("DF.ENHANCE DIR:\n")
    f.write(str(dir(df.enhance)) + "\n\n")
    
    # Check for anything looking like 'stream', 'realtime', 'state'
    f.write("Streaming candidates:\n")
    for name, obj in inspect.getmembers(df):
        if 'stream' in name.lower() or 'real' in name.lower():
            f.write(f"df.{name}\n")
            
    for name, obj in inspect.getmembers(df.enhance):
        if 'stream' in name.lower() or 'real' in name.lower():
            f.write(f"df.enhance.{name}\n")
