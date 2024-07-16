"""
230424
"""

import os
import pandas as pd
import ray
from multiprocessing import Process, freeze_support, Manager

import subprocess
from subprocess import PIPE
import signal

ray.init(num_cpus=24)
DATA_DIR = "../data"


def execute(idx, exec_file, input, output, msg_file):
    # preprocessing for execute

    if not os.path.exists(f"tmp"):
        os.makedirs(f"tmp")

    input_file = f"tmp/input_{idx}.txt"
    output_file = f"tmp/output_{idx}.txt"
    gold_file = f"tmp/gold_{idx}.txt"
    if os.path.exists(input_file):
        os.system("rm {}".format(input_file))
    if os.path.exists(output_file):
        os.system("rm {}".format(output_file))
    if os.path.exists(gold_file):
        os.system("rm {}".format(gold_file))

    input_s = input.split("\\n")
    with open(input_file, "w") as f:
        for i_s in input_s:
            f.write(i_s.strip() + " ")

    gold_s = output.split("\\n")
    with open(gold_file, "w") as f:
        for g_s in gold_s:
            f.write(g_s.strip() + " ")

    try:
        # cmd = f'{exec_file} < {input_file} > {output_file} 2> {msg_file}'
        cmd = f"{exec_file} < {input_file}"
        p = subprocess.Popen(
            cmd, shell=True, stdout=PIPE, stderr=PIPE, start_new_session=True
        )
        stdout, stderr = p.communicate(timeout=2)

        try:
            if stderr != b"":
                err_file = f"tmp/err_{idx}.txt"
                with open(err_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
                return "RE"

            pred = stdout.decode("utf-8")
            with open(output_file, "w") as f:
                f.write(pred)
        except UnicodeDecodeError:
            return "TLE"

        with open(gold_file) as f:
            gold = f.readlines()
            gold = " ".join(gold)

        pred = " ".join(pred.split())
        gold = " ".join(gold.split())

        if pred.strip() != gold.strip():
            return "WA"
        else:
            return "AC"

    except subprocess.TimeoutExpired:
        # os.kill(p.pid, 9)
        # p.kill()
        # p.terminate()
        # cmd = 'kill -9 $(ps -ef | grep ' + exec_file + ' | awk \'{print $2}\' )'
        # os.system(cmd)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        return "TLE"


@ray.remote
def judge(idx, detail, dataset, beam_size, incorrect_code, predict_stmt):

    # init
    result_to_top = []
    pid, cid, iid, i_code = incorrect_code
    i_code_lines = i_code.split("||| ")
    for beam in range(beam_size):
        tmp_idx, stmt = predict_stmt[beam]
        no_stmt = stmt.strip().split(" ", 1)[0]
        # code repair
        is_repair = False
        repair_code = ""
        for line in i_code_lines:
            line = line.strip()
            line_parse = line.split(" ", 1)
            if len(line_parse) < 2:
                break
            no_line, code = line_parse
            # code overwrite
            if no_stmt == no_line:
                is_repair = True
                if line == stmt:
                    is_repair = False
                line = stmt
                s_line_parse = line.split(" ", 1)
                if len(s_line_parse) < 2:
                    code = ""
                else:
                    code = s_line_parse[1]
            repair_code += code + "|||"

        # check is_repair  코드가 수정이 안된경우 : 수정할 수 없는 라인을 예측한 경우, 예측한 라인과 원래 라인이 같은 경우
        if not is_repair:
            result_to_top.append("WA")
            continue

        # compile
        source_file = f"tmp/source_{idx}.cpp"
        exec_file = f"tmp/exec_2slee_{idx}.exe"
        if os.path.exists(source_file):
            os.system("rm {}".format(source_file))
        if os.path.exists(exec_file):
            os.system("rm {}".format(exec_file))

        with open(source_file, "w") as f:
            lines = repair_code.split("|||")
            for line in lines:
                f.write(line.strip() + "\n")

        # check compile error
        ce_msg_file = f"tmp/ce_{idx}.txt"
        cmd = "g++ -static -O2 -std=c++17"  # compile option 제한
        os.system(f"{cmd} {source_file} -o {exec_file} 2> {ce_msg_file}")
        # os.system('g++ ' + source_file + ' -o ' + exec_file + ' 2> ' + ce_msg_file)
        if not os.path.exists(exec_file):
            result_to_top.append("CE")
            continue

        # make samples
        public_sample = f"../../data/samples_{dataset}/{pid}.txt"
        private_sample = f"../../data/private_samples_{dataset}/{pid}.txt"
        generated_sample = f"../../data/generated_samples_{dataset}/{pid}.txt"

        inputs, outputs = [], []

        def add_samples(sample_path, inputs, outputs):
            with open(sample_path, "r") as f:
                lines = f.readlines()
                for l in range(0, len(lines), 2):
                    input = lines[l].split('"')
                    if len(input) < 2:
                        break
                    input = input[1]
                    output = lines[l + 1].split('"')[1]

                    inputs.append(input)
                    outputs.append(output)
            return

        add_samples(public_sample, inputs, outputs)
        add_samples(private_sample, inputs, outputs)
        add_samples(generated_sample, inputs, outputs)

        # execute all samples with multiprocessing
        n_sample = len(inputs)
        cnt_RE, cnt_TLE, cnt_WA, cnt_AC = 0, 0, 0, 0
        for i in range(n_sample):
            exec_msg_file = f"tmp/exec_{idx}.txt"
            ret = execute(idx, exec_file, inputs[i], outputs[i], exec_msg_file)

            cnt_RE += ret == "RE"
            cnt_TLE += ret == "TLE"
            cnt_WA += ret == "WA"
            cnt_AC += ret == "AC"

        if cnt_RE != 0:
            result_to_top.append("RE")
        elif cnt_TLE != 0:
            result_to_top.append("TLE")
        elif cnt_WA != 0:
            result_to_top.append("WA")
        else:
            result_to_top.append("AC")

    return pid, cid, iid, result_to_top


def compile_and_execute(detail, dataset, beam_size, csv_file):

    # data load
    incorrect_code_path = f"../ksc/single_line_r_{dataset}.txt"
    incorrect_code = pd.read_csv(incorrect_code_path, sep="\t")
    incorrect_code = incorrect_code[
        ["PID", "CID", "IID", "Incorrect_code"]
    ].values.tolist()
    predict_stmt_path = f"model/cpp/checkpoint-{detail}/{dataset}_0.output"  # model/cpp_epoch_beam/checkpoint-best-bleu/test_0.output
    predict_stmt = pd.read_csv(
        predict_stmt_path, sep="\t", header=None
    ).values.tolist()  # idx, stmt

    # judge(compile and execute) with multi-threading
    jobs = []
    n_pred = len(incorrect_code)
    for i in range(n_pred):
        jobs.append(
            judge.remote(
                i,
                detail,
                dataset,
                beam_size,
                incorrect_code[i],
                predict_stmt[beam_size * i : beam_size * (i + 1)],
            )
        )

    result_to_csv = ray.get(jobs)
    df = pd.DataFrame(result_to_csv, columns=["PID", "CID", "IID", "RESULT"])
    df.to_csv(csv_file, index=False)


def get_ret(csv_file, top):

    df = pd.read_csv(csv_file)
    df = df["RESULT"].values.tolist()
    s = df[0][1:-1]
    sl = s.split(",")
    sl = [i.strip()[1:-1] for i in sl]

    n = len(df)

    sums = [0] * 4
    for i in range(n):
        ret = [0] * 4
        sl = df[i][1:-1].split(",")
        sl = [i.strip()[1:-1] for i in sl]
        for j in range(10):
            if j >= top:
                continue

            ret[0] += sl[j] == "CE"
            ret[1] += (sl[j] == "RE") | (sl[j] == "TLE")
            ret[2] += sl[j] == "WA"
            ret[3] += sl[j] == "AC"

        if ret[3]:  # ac
            sums[3] += 1
        elif ret[2]:  # wa
            sums[2] += 1
        elif ret[1]:  # re
            sums[1] += 1
        else:  # ce
            sums[0] += 1
    sums = [round(s / n * 100, 1) for s in sums]
    sums.reverse()
    return sums


def eval_localization_precision(detail, dataset, top, beam_size=10):
    """
    라인넘버 예측이 잘 되었는지 확인하는 모듈
    """

    DIR = f"model/cpp/checkpoint-{detail}"
    fgold = DIR + "/" + dataset + "_0.gold"
    foutput = DIR + "/" + dataset + "_0.output"

    ret = 0.0
    with open(fgold, "r") as g, open(foutput, "r") as o:
        lines_gold = g.readlines()
        lines_output = o.readlines()

        localization = 0
        for i, line_g in enumerate(lines_gold):
            line_no_g = line_g.split("\t")[1].split(" ", 1)[0]
            for j in range(beam_size):
                if j >= top:
                    break

                idx = beam_size * i + j
                line_o = lines_output[idx]
                line_no_o = line_o.split("\t")[1].split(" ", 1)[0]

                if line_no_g == line_no_o:
                    localization += 1
                    break

        ret = round(localization / len(lines_gold) * 100, 1)

    return ret


if __name__ == "__main__":

    # details = ["best-bleu", "last", "best-EM", "best-edit-sim", "best-ppl"]
    # details = ['last', 'best-EM', 'best-edit-sim', 'best-ppl']
    # details = ['best-bleu']
    details = ["last"]
    dataset = "test"
    beam_size = 10

    for detail in details:

        csv_file = f"result/{detail}_{dataset}.csv"

        # testing
        # compile_and_execute(detail, dataset, beam_size, csv_file)

        # eval
        print(f"checkpoint-{detail}/{dataset}")
        print("[AC, WA, RE, CE] localization")
        for top in [1, 3, 5, 10]:
            AC = get_ret(csv_file, top)
            FL = eval_localization_precision(detail, dataset, top)
            print(AC, FL)


ray.shutdown()
