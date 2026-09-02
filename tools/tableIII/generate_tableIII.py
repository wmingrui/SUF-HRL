from pathlib import Path
import json
import csv


# Automatically locate repository root
ROOT = Path(__file__).resolve().parents[2]

RESULT = ROOT / "results" / "tableIII"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(path, "r") as f:
        return json.load(f)



def extract_metrics(data, key):

    if key not in data["spatial_metrics"]:
        raise KeyError(
            f"{key} not found in spatial_metrics"
        )

    spatial = data["spatial_metrics"][key]

    row = {
        "Source": key,

        "Error_AUROC": None,
        "Error_AUPR": None,
        "UCE": None,

        "BFUR": spatial["BFUR"]["mean"],
        "DSCG": spatial["DSCG"]["mean"],
        "MSAD": spatial["MSAD"]["mean"],
    }


    # conventional uncertainty metrics
    if (
        "traditional_metrics" in data
        and key in data["traditional_metrics"]
    ):

        trad = data["traditional_metrics"][key]

        row.update(
            {
                "Error_AUROC":
                    trad["Error_AUROC"],

                "Error_AUPR":
                    trad["Error_AUPR"],

                "UCE":
                    trad["UCE"],
            }
        )


    return row



def main():

    # --------------------------------------------------
    # Table III protocol:
    #
    # MSP + Entropy:
    #   computed from baseline checkpoint
    #
    # Learned:
    #   computed from SUF-HRL checkpoint
    #
    # --------------------------------------------------

    baseline_json = (
        RESULT
        / "baseline"
        / "spatial_uncertainty_summary.json"
    )


    sufhrl_json = (
        RESULT
        / "sufhrl"
        / "spatial_uncertainty_summary.json"
    )


    print("=" * 70)
    print("Generating Table III")
    print(f"Repository : {ROOT}")
    print(f"Baseline   : {baseline_json}")
    print(f"SUF-HRL    : {sufhrl_json}")
    print("=" * 70)



    baseline = load_json(
        baseline_json
    )

    sufhrl = load_json(
        sufhrl_json
    )


    rows = []


    # -----------------------------
    # Conventional uncertainty
    # from baseline model
    # -----------------------------

    rows.append(
        extract_metrics(
            baseline,
            "MSP"
        )
    )


    rows.append(
        extract_metrics(
            baseline,
            "Entropy"
        )
    )


    # -----------------------------
    # Learned uncertainty
    # from SUF-HRL model
    # -----------------------------

    rows.append(
        extract_metrics(
            sufhrl,
            "Learned"
        )
    )


    output_csv = (
        RESULT
        / "TableIII_final.csv"
    )


    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)



    print()
    print("=" * 70)
    print("[DONE] Table III generated")
    print(output_csv)
    print("=" * 70)


    print()

    for row in rows:
        print(row)



if __name__ == "__main__":
    main()