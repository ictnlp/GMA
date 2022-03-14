#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
import pdb
import torch
import numpy as np
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import warnings

warnings.filterwarnings("ignore")


def main(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print("| loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(":"),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    rw = []
    step_dict = {}
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()
            hypos, d, srcs = task.inference_step(
                generator, models, sample, prefix_tokens
            )
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)
            rw.extend([d201(d[i], srcs[i]) for i in range(len(d))])

            for i in range(len(d)):
                istep = int(d[i][0] + 1)
                if istep in step_dict.keys():
                    step_dict[istep] += 1
                else:
                    step_dict[istep] = 1

                for j in range(1, len(d[i])):
                    istep = int(
                        (min(d[i][j], srcs[i] - 1) - min(d[i][j - 1], srcs[i] - 1))
                    )

                    if istep in step_dict.keys():
                        step_dict[istep] += 1
                    else:
                        step_dict[istep] = 1

            for i, sample_id in enumerate(sample["id"].tolist()):
                has_target = sample["target"] is not None

                # Remove padding
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
                target_tokens = None
                if has_target:
                    target_tokens = (
                        utils.strip_pad(sample["target"][i, :], tgt_dict.pad())
                        .int()
                        .cpu()
                    )

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(
                        sample_id
                    )
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(
                        sample_id
                    )
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(
                            target_tokens, args.remove_bpe, escape_unk=True
                        )

                if not args.quiet:
                    if src_dict is not None:
                        print("S-{}\t{}".format(sample_id, src_str))
                    if has_target:
                        print("T-{}\t{}".format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][: args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"].int().cpu()
                        if hypo["alignment"] is not None
                        else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print("H-{}\t{}\t{}".format(sample_id, hypo["score"], hypo_str))
                        print(
                            "P-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    map(
                                        lambda x: "{:.4f}".format(x),
                                        hypo["positional_scores"].tolist(),
                                    )
                                ),
                            )
                        )

                        if args.print_alignment:
                            print(
                                "A-{}\t{}".format(
                                    sample_id,
                                    " ".join(
                                        map(lambda x: str(utils.item(x)), alignment)
                                    ),
                                )
                            )

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(
                                target_str, add_if_not_exist=True
                            )
                        if hasattr(scorer, "add_string"):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += sample["nsentences"]

    print(
        "| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        print(
            "| Generate {} with beam={}: {}".format(
                args.gen_subset, args.beam, scorer.result_string()
            )
        )

    cw, ap, al, dal = compute_delay(rw, is_weight_ave=True)
    print("CW score: ", cw)
    print("AP score: ", ap)
    print("AL score: ", al)
    print("DAL score: ", dal)
    print(step_dict)
    return scorer


def d201(d, src):
    # print("+++",d)
    s = "0 " * int(d[0]) + "1 "
    for i in range(1, len(d)):
        s = s + "0 " * int((min(d[i], src) - min(d[i - 1], src))) + "1 "
    if src > d[-1]:
        s = s + "0 " * int(src - d[-1])
    return s


def compute_delay(rw, is_weight_ave=False):
    CWs, ALs, APs, DALs, Lsrc = [], [], [], [], []
    for line in rw:
        line = line.strip()
        al_ans = RW2AL(line)
        dal_ans = RW2DAL(line)
        ap_ans = RW2AP(line)
        cw_ans = RW2CW(line)
        if al_ans is not None:
            ALs.append(al_ans)
            DALs.append(dal_ans)
            APs.append(ap_ans)
            CWs.append(cw_ans)
            Lsrc.append(line.count("0"))

    CW = np.average(CWs) if is_weight_ave else np.average(CWs, weights=Lsrc)
    AL = np.average(ALs) if is_weight_ave else np.average(ALs, weights=Lsrc)
    DAL = np.average(DALs) if is_weight_ave else np.average(DALs, weights=Lsrc)
    AP = np.average(APs) if is_weight_ave else np.average(APs, weights=Lsrc)
    return CW, AP, AL, DAL


def RW2CW(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0
    c = s.count("01")

    if c == 0:
        return 0
    else:
        return x / c


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AP(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    return sum(curr) / x / y


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
        if i == "1" and count == x:
            break
    y1 = len(curr)
    diag = [(t - 1) / rate for t in range(1, y1 + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr, diag)) / y1


def RW2DAL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    curr1 = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    curr1.append(curr[0])
    for i in range(1, y):
        curr1.append(max(curr[i], curr1[i - 1] + (1 / rate)))

    diag = [(t - 1) / rate for t in range(1, y + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr1, diag)) / y


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
