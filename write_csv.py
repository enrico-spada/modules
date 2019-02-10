def write_csv(file, outdir : str):
    filename = outdir.split("/")[-1]
    if outdir.find("/"):
        import os
        directory = "/".join(outdir.split("/")[ : -1])
        if not os.path.exists(directory):
                os.makedirs(directory)
    file.to_csv(outdir)
