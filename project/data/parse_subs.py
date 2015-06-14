import sys
import bisect


def main():
    subs_all = {}
    if len(sys.argv) < 2:
        print('Usage: {0} <submissions_file>'.format(sys.argv[0]),
              file=sys.stderr)
        return

    subs_file = sys.argv[1]
    with open(subs_file) as subs_fd:
        for line in subs_fd:
            tokens = [token.strip() for token in line.split('|')]
            group = tokens[0]
            runtime = float(tokens[3])
            rmse = float(tokens[4])
            if group not in subs_all:
                subs_all[group] = []
            subs_all[group].append((runtime, rmse))

    subs_non_dom = {}
    subs_list = []
    for group in subs_all:
        if group == 'TheNonstandardDeviations':
            continue
        subs_non_dom[group] = []
        for sub in subs_all[group]:
            is_dom = False
            for aSub in subs_all[group]:
                is_dom = is_dom or (sub != aSub
                                    and aSub[0] <= sub[0]
                                    and aSub[1] <= sub[1])
            if not is_dom:
                subs_non_dom[group].append(sub)
                subs_list.append(sub)

    subs_list_time = sorted([s[0] for s in subs_list])
    subs_list_rmse = sorted([s[1] for s in subs_list])

    my_sub = (51.059683, 0.94582)
    time_rank = bisect.bisect(subs_list_time, my_sub[0])
    rmse_rank = bisect.bisect(subs_list_rmse, my_sub[1])

    print((time_rank + 2.0 * rmse_rank) / 3.0)


if __name__ == '__main__':
    main()
