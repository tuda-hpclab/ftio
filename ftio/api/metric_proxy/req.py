"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

# import json
import requests


class MetricProxy:
    def __init__(self, url="http://localhost:1337"):
        self.url = url

    def jobs(self) -> dict:
        return requests.get(f"{self.url}/profiles").json()

    def profile(self, jobid) -> dict:
        return requests.get(f"{self.url}/profiles/get?jobid={jobid}").json()

    def metric(self, jobid) -> dict:
        return requests.get(f"{self.url}/trace/metrics/?job={jobid}").json()

    def trace(self, jobid) -> dict:
        response = requests.get(f"{self.url}/trace/json?jobid={jobid}").json()
        if not response.get("success", True):
            operation = response.get("operation", "Unknown operation")
            raise ValueError(f"Job not found: {operation}")
        else:
            return response

    def trace_metric(self, jobid, metric) -> dict:
        return requests.get(
            f"{self.url}/trace/plot/?jobid={jobid}&filter={requests.utils.quote(metric)}"
        ).json()

    def jsonl(self, jobid) -> str:
        return requests.get(f"{self.url}/model/download?jobid={jobid}").text

    def models(self, jobid) -> dict:
        return requests.get(f"{self.url}/model/get?jobid={jobid}").json()

    def online_models(self, jobid) -> dict:
        """Get models during runtime

        Args:
            jobid (str): job id

        Returns:
            dict: json file
        """
        return requests.get(f"{self.url}/model/ftio?jobid={jobid}").json()

    def plotmodel(self, jobid, metric, end, start=0, step=1):
        return requests.get(
            f"{self.url}/model/plot?jobid={jobid}&metric={requests.utils.quote(metric)}&start={start}&end={end}&step={step}"
        ).json()


# # Client OBJECT
# mp = MetricProxy()

# # Get a LIST of all JOBs
# jobs = mp.jobs()
# #print(json.dumps(jobs, indent=4))

# # Get a JSONL for this JOB
# id_of_first_job = jobs[0]["jobid"]
# jsonl = mp.jsonl(id_of_first_job)
# #print(jsonl)

# # Get the model formulas and associated errors
# models = mp.models(id_of_first_job)
# #print(json.dumps(models, indent=4))

# # Now get the profile of a given JOB
# profile = mp.profile(id_of_first_job)
# #print(json.dumps(profile, indent=4))

# #trace for cpu
# proxy_cpu_load_average_percent =mp.trace_metric(id_of_first_job,"proxy_cpu_load_average_percent")

# first_counter_name = profile["counters"][0]["name"]

# plot = mp.plotmodel(id_of_first_job, first_counter_name, 64)
# print(json.dumps(plot, indent=4))
