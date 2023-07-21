#!/usr/bin/env python3
"""Tune script
"""

import yaml
import torch
import json
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import time
import pandas as pd
import sys
sys.path.insert(0, '/Users/selene/Documents/forest_drought_forecasting/modelling/')
from earthnet_models_pytorch.utils import parse_setting


class Tuner:

    def __init__(self, slurm = False):
        self.slurm = slurm
        self.trial_paths = []

    @staticmethod
    def update_dict(in_dict, keys, val):
        if len(keys) > 0:
            out = Tuner.update_dict(in_dict[keys[0]], keys[1:], val)
            in_dict[keys[0]] = out
        else:
            in_dict = val
        return in_dict

    def create_tune_cfgs(self, params_file, setting_file):
        
        with open(setting_file, 'r') as fp:
            setting_dict = yaml.load(fp, Loader = yaml.FullLoader)

        with open(params_file, 'r') as fp:
            params_dict = yaml.load(fp, Loader = yaml.FullLoader)
        

        outpath = Path(params_file).parent/"grid"

        outpath.mkdir(parents = True, exist_ok = True)
        
        flat_params = []

        for k, v in params_dict.items():
            if k=='Model':
                for param_name, param_vals in params_dict[k].items():
                    if isinstance(param_vals, list):
                        param_list = []
                        for val in param_vals:
                            param_list.append({"param": f"Model.{param_name}", "val": val})
                        flat_params.append(param_list)
    
        trials = list(itertools.product(*flat_params))
        
        trial_paths = []
        for trial in trials:
            trial_name = []
            for elem in trial:
                access = elem["param"].split(".")
                trial_name.append(f"({access[-1]}={elem['val']})")
                access = [int(key) if key.isdigit() else key for key in access]
                setting_dict = self.update_dict(setting_dict, access, elem['val'])
            
            trial_name = "_".join(trial_name)

            trial_path = outpath/(trial_name+".yaml")
            
            with open(trial_path, 'w') as fp:
                yaml.dump(setting_dict, fp)

            trial_paths.append(trial_path)

        self.trial_paths = trial_paths

        return trial_paths

    def select_trials(self, mode = "add"):
        
        new_trials = [] if mode == "add" else self.trial_paths
        for trial_path in self.trial_paths:
            setting_dict = parse_setting(trial_path)
            out_dir = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]
            if out_dir.exists():
                if mode == "overwrite":
                    shutil.rmtree(out_dir)
                    print(f"Overwriting {out_dir}")
                elif mode == "add":
                    print(f"Skipping {trial_path}")
            else:
                if mode == "add":
                    new_trials.append(trial_path)
        self.trial_paths = new_trials



    def start_one_trial(self, trial_path):
        # TODO export CUDA_VISIBLE_DEVICES ???
        
        script_path = Path(sys.path[1])
        
        if not self.slurm:
            cmd = [f"{script_path}/train.py" , trial_path]
            pass
        else:
            cmd = ["sbatch", f"{script_path/'slurmrun.sh'}", trial_path, "train"] #f"{script_path/'slurmrun.sh'} {trial_path} train"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        
        return out

    def run_trials(self):
        
        managers = []
        for trial_path in self.trial_paths:
            managers.append(self.start_one_trial(trial_path))
            #if not self.slurm and len(managers)>1:
                #managers[-1].wait()

        errors = ""

        start_time = time.time()

        if self.slurm:
            time.sleep(15)
            jobs = []
            for idx, manager in enumerate(managers):
                message, err = manager.communicate()
                job = str(message).split(" ")[-1][:-3] # Extracting Job ID that is outputed when running sbatch
                jobs.append(job)
                print(f"Detected SLURM Job: {job} for trial {self.trial_paths[idx]}")

        done = [0]*len(managers)
        while not all(done):
            time.sleep(60)
            for idx, manager in enumerate(managers):
                if not self.slurm:
                    manager.stdout.read()
                    manager.poll()
                    if manager.returncode is not None:
                        if manager.returncode == 0:
                            done[idx] = True
                        else:
                            done[idx] = True
                            errors += f"{self.trial_paths[idx]} failed with returncode {manager.returncode}\n"
                else:
                    # trial_path = self.trial_paths[idx]
                    # setting_dict = parse_setting(trial_path)
                    # out_dir = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]
                    # results_file = out_dir/"validation_scores.json"
                    # if results_file.exists(): #TODO needs to wait till this file contains EPOCH max epoch.....
                    #     done[idx] = True
                    job = jobs[idx]
                    cmd = ["sacct", "-j", f"{job}.0", "-o", "state"]
                    check = subprocess.run(cmd, stdout = subprocess.PIPE)
                    state = str(check.stdout).split("\\n")[-2].strip() # Extracting State from SLURM sacct command...
                    if state == "COMPLETED":
                        done[idx] = True
                    elif state == "CANCELLED":
                        print(f"SLURM Job {job} Cancelled!")
                        errors += f"SLURM Job {job} Cancelled!\n"
                        done[idx] = True
                    elif state == "FAILED":
                        print(f"SLURM Job {job} Failed!")
                        errors += f"SLURM Job {job} Failed!\n"
                        done[idx] = True
    
            current_time = time.time()
            seconds_elapsed = current_time - start_time
            hours, rest = divmod(seconds_elapsed, 3600)

            if hours > 48:
                print("timeout")
                #break

        return errors
    
    def aggregate_results(self):

        all_scores = []
        best_scores = []

        for trial_path in self.trial_paths:
            setting_dict = parse_setting(trial_path)
            out_dir = Path(setting_dict["Logger"]["save_dir"])/setting_dict["Logger"]["name"]/setting_dict["Logger"]["version"]
            with open(out_dir/"validation_scores.json", 'r') as fp:
                scores = json.load(fp)
    
            score = min([s["RMSE_drought"] for s in scores])

            best_scores.append(score)

            all_scores += [{**s, **{"trial": trial_path}} for s in scores]

        best_score = (max(best_scores) if setting_dict["Setting"] == "en21-std" else min(best_scores))
        best_trial = [self.trial_paths[i] for i, s in enumerate(best_scores) if (s == best_score)][0]

        cfg_path = Path(trial_path).parent.parent/"best.yaml"

        best_setting = parse_setting(best_trial)

        print(f"Best trial {best_trial} with score {best_score}.")

        with open(cfg_path, 'w') as fp:
            yaml.dump(best_setting, fp)
        
        scores_df = pd.DataFrame(all_scores)

        scores_df.to_csv(Path(best_setting["Logger"]["save_dir"])/best_setting["Logger"]["name"]/"tune_results.csv")
    

    @classmethod
    def tune(cls, params_file, setting_file, slurm = False, overwrite = False):
        
        self = cls(slurm = slurm)

        print("Creating Configs...")
        self.create_tune_cfgs(params_file, setting_file)

        print("Selecting Trials...")
        old_trials = self.trial_paths
        self.select_trials(mode = ("overwrite" if overwrite else "add"))

        if len(self.trial_paths) > 0:
            print("Running Trials...")
            errors = self.run_trials()
            print(errors)

            self.trial_paths = old_trials
            print("Aggregating Results...")
            self.aggregate_results()
        else:
            print("No trials to run!")


if __name__=="__main__":
    import fire
    fire.Fire(Tuner.tune)    

