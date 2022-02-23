import os
import shutil
import argparse

def get_dirs(basedir):
    subfolders = [f.name for f in os.scandir(basedir) if f.is_dir()]
    return subfolders


parser = argparse.ArgumentParser()
parser.add_argument('rootdirs', nargs='+')

args = parser.parse_args()

for root_dir in args.rootdirs:
    temp_dir = root_dir+"_tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for facet_name in get_dirs(root_dir):
        facet_dir = os.path.join(root_dir, facet_name)
        for surface_name in get_dirs(facet_dir):
            surface_dir = os.path.join(facet_dir, surface_name)
            if os.path.exists(os.path.join(surface_dir, "surface")):
                for run_name in get_dirs(surface_dir):
                    run_dir = os.path.join(surface_dir, run_name)
                    if "surface" in run_name and run_name != "surface":
                        continue
                    if not os.path.exists(os.path.join(run_dir, "OUTCAR")):
                        print("No OUTCAR file detected in %s, skipping this run..." % run_dir)
                    else:
                        out_dir = os.path.join(temp_dir, facet_name, surface_name, run_name)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        shutil.copy(os.path.join(run_dir, "OUTCAR"), os.path.join(out_dir, "OUTCAR"))
            else:
                print("No surface folder detected in %s, skipping this surface..." % surface_dir)

    shutil.make_archive(root_dir, "zip", temp_dir)
    shutil.rmtree(temp_dir)
