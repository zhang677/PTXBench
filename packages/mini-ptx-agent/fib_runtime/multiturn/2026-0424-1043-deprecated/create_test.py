import csv

with open("../template_compile_measure_cuda.txt") as f:
    COMPILE_SCRIPT_CONTENT = f.read()

with open("mha_problems.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    definition_name = rows[-1]["definition_name"]
    workload_uuid = rows[-1]["workload_uuid"]
    with open(f"{definition_name}_{workload_uuid}.py", "w") as out:
        out.write(COMPILE_SCRIPT_CONTENT.replace("<definition_name>", definition_name).replace("<workload_uuid>", workload_uuid))
